"""Fail-closed audit for blind-first human oversight sessions.

The protocol proves internal trace consistency: an independent reviewer decision
must be committed before an AI recommendation is revealed. It does not prove
that the UI, reviewer, or event producer is honest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Final

SCHEMA_VERSION: Final = "blind-first-oversight/v1"
REPORT_VERSION: Final = "blind-first-oversight-report/v1"
EXIT_ACCEPTED: Final = 0
EXIT_REJECTED: Final = 2
EXIT_MALFORMED: Final = 3

_ZERO_DIGEST = "0" * 64
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_EVENT_ORDER = (
    "case_assigned",
    "independent_decision",
    "ai_recommendation_revealed",
    "final_decision",
)
_DECISIONS = frozenset({"approve", "reject", "defer"})


class ArtifactError(ValueError):
    """Raised when an artifact cannot safely enter semantic evaluation."""


@dataclass(frozen=True)
class OversightPolicy:
    policy_id: str = "blind-first/default-v1"
    max_artifact_bytes: int = 131_072
    max_json_depth: int = 12
    max_json_nodes: int = 512
    max_string_chars: int = 2_048
    max_future_skew_seconds: int = 30
    max_independent_decision_seconds: int = 1_800
    max_reveal_delay_seconds: int = 300
    max_final_decision_seconds: int = 1_800
    max_session_seconds: int = 3_600
    allowed_override_reasons: tuple[str, ...] = (
        "new_evidence",
        "policy_constraint",
        "risk_reassessment",
        "model_error_identified",
    )

    def __post_init__(self) -> None:
        _require_identifier(self.policy_id, "policy_id")
        for name in (
            "max_artifact_bytes",
            "max_json_depth",
            "max_json_nodes",
            "max_string_chars",
            "max_future_skew_seconds",
            "max_independent_decision_seconds",
            "max_reveal_delay_seconds",
            "max_final_decision_seconds",
            "max_session_seconds",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.allowed_override_reasons, tuple):
            raise TypeError("allowed_override_reasons must be an immutable tuple")
        if not self.allowed_override_reasons:
            raise ValueError("allowed_override_reasons must not be empty")
        if len(set(self.allowed_override_reasons)) != len(self.allowed_override_reasons):
            raise ValueError("allowed_override_reasons must not contain duplicates")
        for reason in self.allowed_override_reasons:
            _require_identifier(reason, "override reason")


@dataclass(frozen=True)
class OversightReport:
    schema_version: str
    accepted: bool
    reason_codes: tuple[str, ...]
    session_digest: str
    case_digest: str
    reviewer_digest: str
    policy_digest: str
    trace_digest: str
    independent_latency_ms: int
    reveal_delay_ms: int
    final_latency_ms: int
    session_duration_ms: int
    decision_changed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _require_identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ArtifactError(f"{name} must be a bounded identifier")
    return value


def _require_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ArtifactError(f"{name} must be a canonical SHA-256 digest")
    return value


def _require_exact_fields(
    value: object,
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    name: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ArtifactError(f"{name} must be an object")
    fields = set(value)
    missing = required - fields
    unknown = fields - required - optional
    if missing:
        raise ArtifactError(f"{name} is missing fields")
    if unknown:
        raise ArtifactError(f"{name} has unknown fields")
    return value


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ArtifactError("value is not canonical JSON") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _private_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parse_timestamp(value: object, name: str) -> datetime:
    if not isinstance(value, str) or len(value) > 64:
        raise ArtifactError(f"{name} must be a bounded timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ArtifactError(f"{name} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ArtifactError(f"{name} must include a timezone")
    return parsed.astimezone(UTC)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ArtifactError("duplicate JSON field")
        value[key] = item
    return value


def _reject_constant(value: str) -> object:
    raise ArtifactError(f"non-finite JSON number: {value}")


def _check_json_budget(value: object, policy: OversightPolicy) -> None:
    nodes = 0
    stack: list[tuple[object, int]] = [(value, 1)]
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > policy.max_json_nodes:
            raise ArtifactError("JSON node budget exceeded")
        if depth > policy.max_json_depth:
            raise ArtifactError("JSON depth budget exceeded")
        if isinstance(current, str):
            if len(current) > policy.max_string_chars:
                raise ArtifactError("JSON string budget exceeded")
        elif isinstance(current, dict):
            for key, child in current.items():
                if not isinstance(key, str):
                    raise ArtifactError("JSON object keys must be strings")
                if len(key) > policy.max_string_chars:
                    raise ArtifactError("JSON key budget exceeded")
                stack.append((child, depth + 1))
        elif isinstance(current, list):
            for child in current:
                stack.append((child, depth + 1))
        elif current is not None and not isinstance(current, (bool, int, float)):
            raise ArtifactError("unsupported JSON value")
        if isinstance(current, float) and not math.isfinite(current):
            raise ArtifactError("non-finite JSON number")


def parse_trace_json(raw: bytes, policy: OversightPolicy | None = None) -> dict[str, object]:
    policy = policy or OversightPolicy()
    if not isinstance(raw, bytes):
        raise TypeError("raw must be bytes")
    if len(raw) > policy.max_artifact_bytes:
        raise ArtifactError("artifact byte budget exceeded")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except UnicodeDecodeError as exc:
        raise ArtifactError("artifact must be UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactError("artifact must be valid JSON") from exc
    _check_json_budget(value, policy)
    if not isinstance(value, dict):
        raise ArtifactError("artifact root must be an object")
    return value


def seal_event(
    *,
    sequence: int,
    event_type: str,
    occurred_at: str,
    payload: Mapping[str, object],
    previous_digest: str,
) -> dict[str, object]:
    """Create a content-addressed event for a producer-side append-only trace."""

    material: dict[str, object] = {
        "sequence": sequence,
        "event_type": event_type,
        "occurred_at": occurred_at,
        "payload": dict(payload),
        "previous_digest": previous_digest,
    }
    return {**material, "event_digest": _digest(material)}


def _validate_payload(event_type: str, payload: object) -> dict[str, object]:
    if event_type == "case_assigned":
        result = _require_exact_fields(
            payload, required={"case_version"}, name="case_assigned payload"
        )
        _require_identifier(result["case_version"], "case_version")
        return result

    if event_type in {"independent_decision", "final_decision"}:
        optional = {"override_reason"} if event_type == "final_decision" else set()
        result = _require_exact_fields(
            payload,
            required={"decision", "confidence", "decision_digest"},
            optional=optional,
            name=f"{event_type} payload",
        )
        if result["decision"] not in _DECISIONS:
            raise ArtifactError("decision is unsupported")
        confidence = result["confidence"]
        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
        ):
            raise ArtifactError("confidence must be finite and within [0, 1]")
        _require_digest(result["decision_digest"], "decision_digest")
        if "override_reason" in result and result["override_reason"] is not None:
            _require_identifier(result["override_reason"], "override_reason")
        return result

    if event_type == "ai_recommendation_revealed":
        result = _require_exact_fields(
            payload,
            required={"recommendation", "recommendation_digest"},
            name="ai_recommendation_revealed payload",
        )
        if result["recommendation"] not in _DECISIONS:
            raise ArtifactError("recommendation is unsupported")
        _require_digest(result["recommendation_digest"], "recommendation_digest")
        return result

    raise ArtifactError("event_type is unsupported")


def _milliseconds(delta_seconds: float) -> int:
    return round(delta_seconds * 1_000)


def audit_oversight_trace(
    document: Mapping[str, object],
    *,
    now: datetime,
    policy: OversightPolicy | None = None,
) -> OversightReport:
    """Validate one complete blind-first session and return bounded evidence."""

    policy = policy or OversightPolicy()
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    now_utc = now.astimezone(UTC)
    root = _require_exact_fields(
        dict(document),
        required={
            "schema_version",
            "session_id",
            "case_id",
            "reviewer_id",
            "model_revision",
            "prompt_revision",
            "policy_revision",
            "events",
        },
        name="artifact",
    )
    if root["schema_version"] != SCHEMA_VERSION:
        raise ArtifactError("schema_version is unsupported")
    session_id = _require_identifier(root["session_id"], "session_id")
    case_id = _require_identifier(root["case_id"], "case_id")
    reviewer_id = _require_identifier(root["reviewer_id"], "reviewer_id")
    _require_identifier(root["model_revision"], "model_revision")
    _require_identifier(root["prompt_revision"], "prompt_revision")
    policy_revision = _require_identifier(root["policy_revision"], "policy_revision")
    events_value = root["events"]
    if not isinstance(events_value, list) or len(events_value) != len(_EVENT_ORDER):
        raise ArtifactError("events must contain exactly four protocol events")

    reasons: set[str] = set()
    if policy_revision != policy.policy_id:
        reasons.add("POLICY_REVISION_MISMATCH")
    timestamps: list[datetime] = []
    timestamps_by_type: dict[str, datetime] = {}
    payloads_by_type: dict[str, dict[str, object]] = {}
    prior_digest = _ZERO_DIGEST
    for expected_sequence, (expected_type, raw_event) in enumerate(
        zip(_EVENT_ORDER, events_value, strict=True)
    ):
        event = _require_exact_fields(
            raw_event,
            required={
                "sequence",
                "event_type",
                "occurred_at",
                "payload",
                "previous_digest",
                "event_digest",
            },
            name="event",
        )
        sequence = event["sequence"]
        if (
            not isinstance(sequence, int)
            or isinstance(sequence, bool)
            or sequence != expected_sequence
        ):
            reasons.add("INVALID_EVENT_SEQUENCE")
        event_type = event["event_type"]
        if event_type != expected_type:
            reasons.add("INVALID_EVENT_ORDER")
        if not isinstance(event_type, str):
            raise ArtifactError("event_type must be a string")
        payload = _validate_payload(event_type, event["payload"])
        timestamp = _parse_timestamp(event["occurred_at"], "occurred_at")
        timestamps.append(timestamp)
        payloads_by_type[event_type] = payload
        timestamps_by_type[event_type] = timestamp
        previous_digest = _require_digest(event["previous_digest"], "previous_digest")
        event_digest = _require_digest(event["event_digest"], "event_digest")
        if previous_digest != prior_digest:
            reasons.add("EVENT_CHAIN_MISMATCH")
        material = {
            "sequence": event["sequence"],
            "event_type": event["event_type"],
            "occurred_at": event["occurred_at"],
            "payload": event["payload"],
            "previous_digest": event["previous_digest"],
        }
        if event_digest != _digest(material):
            reasons.add("EVENT_DIGEST_MISMATCH")
        prior_digest = event_digest

    if set(payloads_by_type) != set(_EVENT_ORDER):
        raise ArtifactError("events must contain each protocol event exactly once")

    if any(later <= earlier for earlier, later in pairwise(timestamps)):
        reasons.add("NON_MONOTONIC_EVENT_TIME")
    if timestamps[-1] > now_utc:
        skew = (timestamps[-1] - now_utc).total_seconds()
        if skew > policy.max_future_skew_seconds:
            reasons.add("FUTURE_EVENT")

    independent_seconds = (
        timestamps_by_type["independent_decision"] - timestamps_by_type["case_assigned"]
    ).total_seconds()
    reveal_seconds = (
        timestamps_by_type["ai_recommendation_revealed"]
        - timestamps_by_type["independent_decision"]
    ).total_seconds()
    final_seconds = (
        timestamps_by_type["final_decision"] - timestamps_by_type["ai_recommendation_revealed"]
    ).total_seconds()
    session_seconds = (
        timestamps_by_type["final_decision"] - timestamps_by_type["case_assigned"]
    ).total_seconds()
    if independent_seconds > policy.max_independent_decision_seconds:
        reasons.add("INDEPENDENT_DECISION_TIMEOUT")
    if reveal_seconds > policy.max_reveal_delay_seconds:
        reasons.add("REVEAL_DELAY_EXCEEDED")
    if final_seconds > policy.max_final_decision_seconds:
        reasons.add("FINAL_DECISION_TIMEOUT")
    if session_seconds > policy.max_session_seconds:
        reasons.add("SESSION_TIMEOUT")

    independent_decision = payloads_by_type["independent_decision"]["decision"]
    final_decision = payloads_by_type["final_decision"]["decision"]
    changed = independent_decision != final_decision
    override_reason = payloads_by_type["final_decision"].get("override_reason")
    if changed:
        if override_reason is None:
            reasons.add("MISSING_OVERRIDE_REASON")
        elif override_reason not in policy.allowed_override_reasons:
            reasons.add("UNAPPROVED_OVERRIDE_REASON")
    elif override_reason is not None:
        reasons.add("UNEXPECTED_OVERRIDE_REASON")

    return OversightReport(
        schema_version=REPORT_VERSION,
        accepted=not reasons,
        reason_codes=tuple(sorted(reasons)),
        session_digest=_private_digest(session_id),
        case_digest=_private_digest(case_id),
        reviewer_digest=_private_digest(reviewer_id),
        policy_digest=_digest(asdict(policy)),
        trace_digest=_digest(root),
        independent_latency_ms=_milliseconds(independent_seconds),
        reveal_delay_ms=_milliseconds(reveal_seconds),
        final_latency_ms=_milliseconds(final_seconds),
        session_duration_ms=_milliseconds(session_seconds),
        decision_changed=changed,
    )


def _atomic_write_json(path: Path, report: OversightReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    fd, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _parse_now(value: str | None) -> datetime:
    return datetime.now(UTC) if value is None else _parse_timestamp(value, "now")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a blind-first human oversight trace.")
    parser.add_argument("input", type=Path, help="oversight trace JSON")
    parser.add_argument("--output", type=Path, help="atomic report destination")
    parser.add_argument("--now", help="timezone-aware audit time for replay")
    args = parser.parse_args(argv)
    policy = OversightPolicy()
    try:
        with args.input.open("rb") as handle:
            raw = handle.read(policy.max_artifact_bytes + 1)
        document = parse_trace_json(raw, policy)
        report = audit_oversight_trace(document, now=_parse_now(args.now), policy=policy)
        if args.output:
            _atomic_write_json(args.output, report)
        else:
            json.dump(report.to_dict(), sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
    except (ArtifactError, OSError, ValueError) as exc:
        print(f"oversight-protocol: {type(exc).__name__}", file=sys.stderr)
        return EXIT_MALFORMED
    return EXIT_ACCEPTED if report.accepted else EXIT_REJECTED


if __name__ == "__main__":
    raise SystemExit(main())
