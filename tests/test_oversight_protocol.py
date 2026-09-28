from __future__ import annotations

import copy
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.oversight_protocol import (
    ArtifactError,
    OversightPolicy,
    audit_oversight_trace,
    parse_trace_json,
    seal_event,
)

NOW = datetime(2026, 9, 29, 0, 0, tzinfo=UTC)
DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def timestamp(seconds: int) -> str:
    return (NOW - timedelta(minutes=10) + timedelta(seconds=seconds)).isoformat()


def make_trace(
    *,
    event_types: tuple[str, ...] = (
        "case_assigned",
        "independent_decision",
        "ai_recommendation_revealed",
        "final_decision",
    ),
    times: tuple[int, ...] = (0, 60, 65, 120),
    independent: str = "approve",
    final: str = "approve",
    override_reason: str | None = None,
) -> dict[str, object]:
    payloads: tuple[dict[str, object], ...] = (
        {"case_version": "case-v7"},
        {
            "decision": independent,
            "confidence": 0.61,
            "decision_digest": DIGEST_A,
        },
        {"recommendation": "reject", "recommendation_digest": DIGEST_B},
        {
            "decision": final,
            "confidence": 0.82,
            "decision_digest": DIGEST_B,
            **({"override_reason": override_reason} if override_reason is not None else {}),
        },
    )
    events: list[dict[str, object]] = []
    previous = "0" * 64
    for sequence, (event_type, seconds, payload) in enumerate(
        zip(event_types, times, payloads, strict=True)
    ):
        event = seal_event(
            sequence=sequence,
            event_type=event_type,
            occurred_at=timestamp(seconds),
            payload=payload,
            previous_digest=previous,
        )
        events.append(event)
        previous = str(event["event_digest"])
    return {
        "schema_version": "blind-first-oversight/v1",
        "session_id": "session-123",
        "case_id": "case-456",
        "reviewer_id": "reviewer-789",
        "model_revision": "model-v3",
        "prompt_revision": "prompt-v4",
        "policy_revision": "blind-first/default-v1",
        "events": events,
    }


def audit(trace: dict[str, object], policy: OversightPolicy | None = None):
    return audit_oversight_trace(trace, now=NOW, policy=policy)


def reseal(trace: dict[str, object]) -> None:
    previous = "0" * 64
    events = trace["events"]
    assert isinstance(events, list)
    for event in events:
        assert isinstance(event, dict)
        event["previous_digest"] = previous
        sealed = seal_event(
            sequence=event["sequence"],
            event_type=event["event_type"],
            occurred_at=event["occurred_at"],
            payload=event["payload"],
            previous_digest=previous,
        )
        event["event_digest"] = sealed["event_digest"]
        previous = str(event["event_digest"])


def test_accepts_blind_first_session() -> None:
    report = audit(make_trace())
    assert report.accepted
    assert report.reason_codes == ()
    assert report.independent_latency_ms == 60_000
    assert report.reveal_delay_ms == 5_000
    assert report.final_latency_ms == 55_000
    assert report.session_duration_ms == 120_000
    assert not report.decision_changed


def test_accepts_documented_override() -> None:
    report = audit(make_trace(final="reject", override_reason="model_error_identified"))
    assert report.accepted
    assert report.decision_changed


def test_report_is_deterministic_and_private() -> None:
    trace = make_trace()
    first = audit(trace).to_dict()
    second = audit(trace).to_dict()
    assert first == second
    encoded = json.dumps(first)
    assert "session-123" not in encoded
    assert "case-456" not in encoded
    assert "reviewer-789" not in encoded
    assert all(
        len(str(first[key])) == 64
        for key in (
            "session_digest",
            "case_digest",
            "reviewer_digest",
            "policy_digest",
            "trace_digest",
        )
    )


def test_policy_change_changes_evidence() -> None:
    trace = make_trace()
    first = audit(trace)
    second = audit(trace, OversightPolicy(max_session_seconds=4_000))
    assert first.trace_digest == second.trace_digest
    assert first.policy_digest != second.policy_digest


def test_rejects_ai_reveal_before_independent_decision() -> None:
    trace = make_trace()
    events = trace["events"]
    assert isinstance(events, list)
    events[1], events[2] = events[2], events[1]
    events[1]["sequence"] = 1
    events[2]["sequence"] = 2
    reseal(trace)
    report = audit(trace)
    assert not report.accepted
    assert "INVALID_EVENT_ORDER" in report.reason_codes


def test_rejects_policy_revision_drift() -> None:
    trace = make_trace()
    trace["policy_revision"] = "blind-first/old-v0"
    report = audit(trace)
    assert not report.accepted
    assert "POLICY_REVISION_MISMATCH" in report.reason_codes


def test_rejects_non_monotonic_time() -> None:
    trace = make_trace(times=(0, 60, 59, 120))
    report = audit(trace)
    assert not report.accepted
    assert "NON_MONOTONIC_EVENT_TIME" in report.reason_codes


@pytest.mark.parametrize(
    ("times", "policy", "code"),
    [
        (
            (0, 11, 12, 13),
            OversightPolicy(max_independent_decision_seconds=10),
            "INDEPENDENT_DECISION_TIMEOUT",
        ),
        ((0, 1, 12, 13), OversightPolicy(max_reveal_delay_seconds=10), "REVEAL_DELAY_EXCEEDED"),
        ((0, 1, 2, 13), OversightPolicy(max_final_decision_seconds=10), "FINAL_DECISION_TIMEOUT"),
        ((0, 1, 2, 13), OversightPolicy(max_session_seconds=10), "SESSION_TIMEOUT"),
    ],
)
def test_rejects_timing_budget(times: tuple[int, ...], policy: OversightPolicy, code: str) -> None:
    report = audit(make_trace(times=times), policy)
    assert not report.accepted
    assert code in report.reason_codes


def test_rejects_future_event_beyond_skew() -> None:
    trace = make_trace(times=(590, 595, 599, 631))
    report = audit(trace)
    assert not report.accepted
    assert "FUTURE_EVENT" in report.reason_codes


@pytest.mark.parametrize(
    ("final", "reason", "code"),
    [
        ("reject", None, "MISSING_OVERRIDE_REASON"),
        ("reject", "reviewer_mood", "UNAPPROVED_OVERRIDE_REASON"),
        ("approve", "new_evidence", "UNEXPECTED_OVERRIDE_REASON"),
    ],
)
def test_rejects_invalid_override_policy(final: str, reason: str | None, code: str) -> None:
    report = audit(make_trace(final=final, override_reason=reason))
    assert not report.accepted
    assert code in report.reason_codes


def test_detects_event_content_tampering() -> None:
    trace = make_trace()
    events = trace["events"]
    assert isinstance(events, list)
    payload = events[1]["payload"]
    assert isinstance(payload, dict)
    payload["confidence"] = 0.99
    report = audit(trace)
    assert not report.accepted
    assert "EVENT_DIGEST_MISMATCH" in report.reason_codes


def test_detects_chain_tampering() -> None:
    trace = make_trace()
    events = trace["events"]
    assert isinstance(events, list)
    events[2]["previous_digest"] = "f" * 64
    report = audit(trace)
    assert not report.accepted
    assert "EVENT_CHAIN_MISMATCH" in report.reason_codes
    assert "EVENT_DIGEST_MISMATCH" in report.reason_codes


def test_rejects_sequence_change_even_when_resealed() -> None:
    trace = make_trace()
    events = trace["events"]
    assert isinstance(events, list)
    events[2]["sequence"] = 9
    reseal(trace)
    report = audit(trace)
    assert not report.accepted
    assert "INVALID_EVENT_SEQUENCE" in report.reason_codes


@pytest.mark.parametrize(
    "mutator",
    [
        lambda trace: trace.update({"unknown": True}),
        lambda trace: trace.pop("case_id"),
        lambda trace: trace.update({"schema_version": "v0"}),
        lambda trace: trace.update({"session_id": "bad id with spaces"}),
        lambda trace: trace.update({"events": trace["events"][:3]}),
    ],
)
def test_rejects_malformed_root(mutator) -> None:
    trace = make_trace()
    mutator(trace)
    with pytest.raises(ArtifactError):
        audit(trace)


@pytest.mark.parametrize(
    "payload_patch",
    [
        {"confidence": float("nan")},
        {"confidence": True},
        {"confidence": 1.1},
        {"decision": "maybe"},
        {"decision_digest": "ABC"},
        {"extra": "field"},
    ],
)
def test_rejects_malformed_decision_payload(payload_patch: dict[str, object]) -> None:
    trace = make_trace()
    events = trace["events"]
    assert isinstance(events, list)
    payload = events[1]["payload"]
    assert isinstance(payload, dict)
    payload.update(payload_patch)
    with pytest.raises(ArtifactError):
        audit(trace)


def test_requires_timezone_aware_now() -> None:
    with pytest.raises(ValueError):
        audit_oversight_trace(make_trace(), now=datetime.fromisoformat("2026-09-29"))


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":2}',
        b'{"value":NaN}',
        b"[]",
        b"not-json",
        b"\xff",
    ],
)
def test_strict_json_rejects_malformed_artifacts(raw: bytes) -> None:
    with pytest.raises(ArtifactError):
        parse_trace_json(raw)


def test_json_resource_budgets() -> None:
    with pytest.raises(ArtifactError, match="byte budget"):
        parse_trace_json(b"{} " * 20, OversightPolicy(max_artifact_bytes=10))
    with pytest.raises(ArtifactError, match="depth budget"):
        parse_trace_json(b'{"a":{"b":{"c":1}}}', OversightPolicy(max_json_depth=3))
    with pytest.raises(ArtifactError, match="node budget"):
        parse_trace_json(b'{"a":[1,2,3]}', OversightPolicy(max_json_nodes=3))
    with pytest.raises(ArtifactError, match="string budget"):
        parse_trace_json(b'{"a":"long"}', OversightPolicy(max_string_chars=3))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_artifact_bytes": 0},
        {"max_json_depth": True},
        {"max_session_seconds": -1},
        {"allowed_override_reasons": []},
        {"allowed_override_reasons": ()},
        {"allowed_override_reasons": ("new_evidence", "new_evidence")},
        {"allowed_override_reasons": ("bad reason",)},
    ],
)
def test_rejects_invalid_policy(kwargs: dict[str, object]) -> None:
    with pytest.raises((ArtifactError, TypeError, ValueError)):
        OversightPolicy(**kwargs)


def test_seal_event_is_deterministic() -> None:
    kwargs = {
        "sequence": 0,
        "event_type": "case_assigned",
        "occurred_at": timestamp(0),
        "payload": {"case_version": "v1"},
        "previous_digest": "0" * 64,
    }
    assert seal_event(**kwargs) == seal_event(**kwargs)


def test_cli_exit_codes_and_atomic_report(tmp_path: Path) -> None:
    input_path = tmp_path / "trace.json"
    output_path = tmp_path / "report.json"
    input_path.write_text(json.dumps(make_trace()))
    command = [
        sys.executable,
        "-m",
        "app.oversight_protocol",
        str(input_path),
        "--now",
        NOW.isoformat(),
        "--output",
        str(output_path),
    ]
    accepted = subprocess.run(command, check=False, capture_output=True, text=True)
    assert accepted.returncode == 0
    assert json.loads(output_path.read_text())["accepted"] is True

    rejected_trace = make_trace(final="reject")
    input_path.write_text(json.dumps(rejected_trace))
    rejected = subprocess.run(command, check=False, capture_output=True, text=True)
    assert rejected.returncode == 2
    assert json.loads(output_path.read_text())["reason_codes"] == ["MISSING_OVERRIDE_REASON"]


def test_cli_malformed_input_does_not_leak_path(tmp_path: Path) -> None:
    missing = tmp_path / "customer-secret" / "missing.json"
    result = subprocess.run(
        [sys.executable, "-m", "app.oversight_protocol", str(missing)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 3
    assert "FileNotFoundError" in result.stderr
    assert "customer-secret" not in result.stderr


def test_input_mutation_does_not_change_previous_report() -> None:
    trace = make_trace()
    report = audit(trace)
    changed = copy.deepcopy(trace)
    changed["case_id"] = "another-case"
    assert report.case_digest != audit(changed).case_digest
    assert report.to_dict() == audit(trace).to_dict()
