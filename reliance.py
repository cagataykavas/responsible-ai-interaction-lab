from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from statistics import NormalDist

from experiment import HumanResponse, Variant


@dataclass(frozen=True, slots=True)
class SelectiveReliancePolicy:
    confidence_level: float = 0.95
    min_cases_per_ai_outcome: int = 20
    min_correct_reliance_lower: float = 0.70
    max_harmful_reliance_upper: float = 0.30
    min_reliance_gap_lower: float = 0.30
    max_cases: int = 5_000
    max_case_id_bytes: int = 200

    def __post_init__(self) -> None:
        if (
            isinstance(self.confidence_level, bool)
            or not isinstance(self.confidence_level, (int, float))
            or not math.isfinite(self.confidence_level)
            or not 0.5 <= self.confidence_level < 1.0
        ):
            raise ValueError("confidence_level must be in [0.5, 1.0)")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (
                self.min_cases_per_ai_outcome,
                self.max_cases,
                self.max_case_id_bytes,
            )
        ):
            raise ValueError("case budgets must be positive")
        for name, value, lower in (
            ("min_correct_reliance_lower", self.min_correct_reliance_lower, 0.0),
            ("max_harmful_reliance_upper", self.max_harmful_reliance_upper, 0.0),
            ("min_reliance_gap_lower", self.min_reliance_gap_lower, -1.0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not lower <= value <= 1.0
            ):
                raise ValueError(f"{name} must be finite and within [{lower}, 1]")


@dataclass(frozen=True, slots=True)
class RateInterval:
    successes: int
    trials: int
    rate: float
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class SelectiveRelianceReport:
    accepted: bool
    reasons: tuple[str, ...]
    variant: Variant
    cases_seen: int
    cases_evaluated: int
    deferred_cases: int
    ai_correct_cases: int
    ai_incorrect_cases: int
    correct_ai_reliance: RateInterval
    harmful_ai_reliance: RateInterval
    beneficial_override: RateInterval
    reliance_gap: float
    reliance_gap_lower: float
    reliance_gap_upper: float
    evidence_sha256: str


def _wilson(successes: int, trials: int, confidence_level: float) -> RateInterval:
    if trials == 0:
        return RateInterval(successes, trials, 0.0, 0.0, 1.0)
    rate = successes / trials
    z = NormalDist().inv_cdf((1.0 + confidence_level) / 2.0)
    denominator = 1.0 + z * z / trials
    center = (rate + z * z / (2.0 * trials)) / denominator
    margin = (
        z * math.sqrt(rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials)) / denominator
    )
    return RateInterval(
        successes=successes,
        trials=trials,
        rate=rate,
        lower=max(0.0, center - margin),
        upper=min(1.0, center + margin),
    )


def _validate(rows: list[HumanResponse], policy: SelectiveReliancePolicy) -> Variant:
    if not rows:
        raise ValueError("responses must not be empty")
    if len(rows) > policy.max_cases:
        raise ValueError("response case budget exceeded")
    case_ids = [row.case_id for row in rows]
    if any(
        not isinstance(case_id, str)
        or not case_id.strip()
        or len(case_id.encode("utf-8")) > policy.max_case_id_bytes
        for case_id in case_ids
    ):
        raise ValueError("case IDs must be non-empty strings within the byte budget")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("response case IDs must be unique")

    if any(not isinstance(row.variant, Variant) for row in rows):
        raise ValueError("responses must contain exactly one known interaction variant")
    variants = {row.variant for row in rows}
    if len(variants) != 1:
        raise ValueError("responses must contain exactly one known interaction variant")
    for row in rows:
        for name, value in (
            ("ai_label", row.ai_label),
            ("human_label", row.human_label),
            ("true_label", row.true_label),
        ):
            if isinstance(value, bool) or value not in (0, 1):
                raise ValueError(f"case {row.case_id!r} has invalid {name}")
        if not isinstance(row.deferred, bool):
            raise TypeError(f"case {row.case_id!r} has invalid deferred state")
        if row.confidence_seen is not None and (
            isinstance(row.confidence_seen, bool)
            or not isinstance(row.confidence_seen, (int, float))
            or not math.isfinite(row.confidence_seen)
            or not 0.0 <= row.confidence_seen <= 1.0
        ):
            raise ValueError(f"case {row.case_id!r} has invalid confidence_seen")
    return next(iter(variants))


def selective_reliance_audit(
    responses: list[HumanResponse],
    *,
    policy: SelectiveReliancePolicy | None = None,
) -> SelectiveRelianceReport:
    """Gate observed reviewer reliance separately for correct and incorrect AI advice."""

    active_policy = policy or SelectiveReliancePolicy()
    rows = sorted(responses, key=lambda row: row.case_id)
    variant = _validate(rows, active_policy)
    evaluated = [row for row in rows if not row.deferred]
    ai_correct = [row for row in evaluated if row.ai_correct]
    ai_incorrect = [row for row in evaluated if not row.ai_correct]

    correct_reliance = _wilson(
        sum(row.agreed for row in ai_correct),
        len(ai_correct),
        active_policy.confidence_level,
    )
    harmful_reliance = _wilson(
        sum(row.agreed for row in ai_incorrect),
        len(ai_incorrect),
        active_policy.confidence_level,
    )
    beneficial_override = _wilson(
        sum(row.overridden and row.human_correct for row in ai_incorrect),
        len(ai_incorrect),
        active_policy.confidence_level,
    )
    reliance_gap = correct_reliance.rate - harmful_reliance.rate
    gap_lower = correct_reliance.lower - harmful_reliance.upper
    gap_upper = correct_reliance.upper - harmful_reliance.lower

    reasons: list[str] = []
    if len(ai_correct) < active_policy.min_cases_per_ai_outcome:
        reasons.append("insufficient_ai_correct_cases")
    if len(ai_incorrect) < active_policy.min_cases_per_ai_outcome:
        reasons.append("insufficient_ai_incorrect_cases")
    if not reasons:
        if correct_reliance.lower < active_policy.min_correct_reliance_lower:
            reasons.append("correct_reliance_below_policy")
        if harmful_reliance.upper > active_policy.max_harmful_reliance_upper:
            reasons.append("harmful_reliance_above_policy")
        if gap_lower < active_policy.min_reliance_gap_lower:
            reasons.append("reliance_discrimination_below_policy")

    evidence_payload = {
        "schema": "selective-reliance-audit/v1",
        "policy": asdict(active_policy),
        "responses": [
            {
                "case_id": row.case_id,
                "variant": row.variant.value,
                "ai_label": row.ai_label,
                "human_label": row.human_label,
                "true_label": row.true_label,
                "confidence_seen": row.confidence_seen,
                "deferred": row.deferred,
            }
            for row in rows
        ],
    }
    evidence_sha256 = hashlib.sha256(
        json.dumps(evidence_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return SelectiveRelianceReport(
        accepted=not reasons,
        reasons=tuple(reasons),
        variant=variant,
        cases_seen=len(rows),
        cases_evaluated=len(evaluated),
        deferred_cases=len(rows) - len(evaluated),
        ai_correct_cases=len(ai_correct),
        ai_incorrect_cases=len(ai_incorrect),
        correct_ai_reliance=correct_reliance,
        harmful_ai_reliance=harmful_reliance,
        beneficial_override=beneficial_override,
        reliance_gap=reliance_gap,
        reliance_gap_lower=gap_lower,
        reliance_gap_upper=gap_upper,
        evidence_sha256=evidence_sha256,
    )
