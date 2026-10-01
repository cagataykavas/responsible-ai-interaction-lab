"""Pre-registered selective-risk admission for AI recommendations."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

from experiment import Prediction, generate_synthetic_predictions

MAX_CASES = 100_000
MAX_CASE_ID_LENGTH = 256


@dataclass(frozen=True, slots=True)
class SelectiveRiskPolicy:
    """Release limits chosen before evaluating the holdout set."""

    confidence_threshold: float
    minimum_coverage: float
    maximum_selective_risk: float
    maximum_aurc: float


@dataclass(frozen=True, slots=True)
class RiskCoveragePoint:
    minimum_confidence: float
    automated_cases: int
    errors: int
    coverage: float
    selective_risk: float


@dataclass(frozen=True, slots=True)
class SelectiveRiskReport:
    schema_version: int
    input_digest: str
    evidence_id: str
    cases: int
    automated_cases: int
    deferred_cases: int
    coverage: float
    selective_risk: float | None
    deferred_error_capture: float | None
    area_under_risk_coverage: float
    violations: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return not self.violations

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "accepted": self.accepted}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_policy(policy: SelectiveRiskPolicy) -> None:
    for name, value in asdict(policy).items():
        if not _is_number(value) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be a finite number in [0, 1]")


def _validate_predictions(predictions: list[Prediction]) -> None:
    if not predictions:
        raise ValueError("at least one prediction is required")
    if len(predictions) > MAX_CASES:
        raise ValueError(f"prediction count exceeds {MAX_CASES}")

    seen: set[str] = set()
    for row in predictions:
        if not isinstance(row.case_id, str) or not 1 <= len(row.case_id) <= MAX_CASE_ID_LENGTH:
            raise ValueError("case_id must contain 1..256 characters")
        if row.case_id in seen:
            raise ValueError(f"duplicate case_id: {row.case_id}")
        seen.add(row.case_id)
        if type(row.true_label) is not int or row.true_label not in {0, 1}:
            raise ValueError("true_label must be 0 or 1")
        if not _is_number(row.probability) or not math.isfinite(row.probability):
            raise ValueError("probability must be finite")
        if not 0 <= row.probability <= 1:
            raise ValueError("probability must be in [0, 1]")


def _input_digest(predictions: list[Prediction]) -> str:
    rows = sorted(
        (
            hashlib.sha256(row.case_id.encode()).hexdigest(),
            row.true_label,
            row.probability,
        )
        for row in predictions
    )
    payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def risk_coverage_curve(predictions: list[Prediction]) -> tuple[RiskCoveragePoint, ...]:
    """Return a tie-safe curve, admitting confidence groups from high to low."""
    rows = list(predictions)
    _validate_predictions(rows)
    ordered = sorted(rows, key=lambda row: row.confidence, reverse=True)
    total = len(ordered)
    points: list[RiskCoveragePoint] = []
    automated = 0
    errors = 0
    index = 0

    while index < total:
        confidence = ordered[index].confidence
        end = index
        while end < total and ordered[end].confidence == confidence:
            automated += 1
            errors += int(not ordered[end].correct)
            end += 1
        points.append(
            RiskCoveragePoint(
                minimum_confidence=confidence,
                automated_cases=automated,
                errors=errors,
                coverage=automated / total,
                selective_risk=errors / automated,
            )
        )
        index = end

    return tuple(points)


def area_under_risk_coverage(curve: tuple[RiskCoveragePoint, ...]) -> float:
    """Integrate the right-continuous, tie-grouped risk-coverage curve."""
    previous_coverage = 0.0
    area = 0.0
    for point in curve:
        if point.coverage <= previous_coverage:
            raise ValueError("curve coverage must be strictly increasing")
        area += (point.coverage - previous_coverage) * point.selective_risk
        previous_coverage = point.coverage
    if not math.isclose(previous_coverage, 1.0):
        raise ValueError("curve must end at full coverage")
    return area


def audit_selective_risk(
    predictions: list[Prediction],
    policy: SelectiveRiskPolicy,
) -> SelectiveRiskReport:
    """Evaluate one fixed automation threshold without tuning on the holdout set."""
    rows = list(predictions)
    _validate_policy(policy)
    _validate_predictions(rows)
    curve = risk_coverage_curve(rows)
    aurc = area_under_risk_coverage(curve)

    automated = [row for row in rows if row.confidence >= policy.confidence_threshold]
    deferred = [row for row in rows if row.confidence < policy.confidence_threshold]
    automated_errors = sum(not row.correct for row in automated)
    total_errors = sum(not row.correct for row in rows)
    deferred_errors = sum(not row.correct for row in deferred)
    coverage = len(automated) / len(rows)
    selective_risk = automated_errors / len(automated) if automated else None
    error_capture = deferred_errors / total_errors if total_errors else None

    violations: list[str] = []
    if not automated:
        violations.append("no_automated_cases")
    if coverage < policy.minimum_coverage:
        violations.append("coverage_below_minimum")
    if selective_risk is not None and selective_risk > policy.maximum_selective_risk:
        violations.append("selective_risk_above_maximum")
    if aurc > policy.maximum_aurc:
        violations.append("aurc_above_maximum")

    digest = _input_digest(rows)
    evidence_payload = {
        "schema_version": 1,
        "input_digest": digest,
        "policy": asdict(policy),
        "cases": len(rows),
        "automated_cases": len(automated),
        "coverage": coverage,
        "selective_risk": selective_risk,
        "deferred_error_capture": error_capture,
        "area_under_risk_coverage": aurc,
        "violations": violations,
    }
    evidence_id = hashlib.sha256(
        json.dumps(evidence_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return SelectiveRiskReport(
        schema_version=1,
        input_digest=digest,
        evidence_id=evidence_id,
        cases=len(rows),
        automated_cases=len(automated),
        deferred_cases=len(deferred),
        coverage=coverage,
        selective_risk=selective_risk,
        deferred_error_capture=error_capture,
        area_under_risk_coverage=aurc,
        violations=tuple(violations),
    )


def main() -> int:
    predictions = generate_synthetic_predictions(count=2_000, seed=42)
    policy = SelectiveRiskPolicy(
        confidence_threshold=0.80,
        minimum_coverage=0.45,
        maximum_selective_risk=0.10,
        maximum_aurc=0.20,
    )
    report = audit_selective_risk(predictions, policy)
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    return 0 if report.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
