from __future__ import annotations

import math

import pytest

from app.selective_risk import (
    SelectiveRiskPolicy,
    area_under_risk_coverage,
    audit_selective_risk,
    risk_coverage_curve,
)
from experiment import Prediction


def prediction(case_id: str, *, label: int, probability: float) -> Prediction:
    return Prediction(
        case_id=case_id,
        true_label=label,
        probability=probability,
        evidence_quality=0.8,
        explanation_quality=0.8,
    )


def permissive_policy(*, threshold: float = 0.8) -> SelectiveRiskPolicy:
    return SelectiveRiskPolicy(
        confidence_threshold=threshold,
        minimum_coverage=0.0,
        maximum_selective_risk=1.0,
        maximum_aurc=1.0,
    )


def test_policy_separates_coverage_risk_and_error_capture() -> None:
    rows = [
        prediction("certain-correct", label=1, probability=0.95),
        prediction("certain-wrong", label=0, probability=0.90),
        prediction("deferred-wrong", label=1, probability=0.30),
        prediction("deferred-correct", label=0, probability=0.40),
    ]

    report = audit_selective_risk(rows, permissive_policy())

    assert report.accepted
    assert report.coverage == 0.5
    assert report.selective_risk == 0.5
    assert report.deferred_error_capture == 0.5
    assert report.automated_cases == report.deferred_cases == 2


def test_release_fails_each_prespecified_guardrail() -> None:
    rows = [
        prediction("a", label=1, probability=0.95),
        prediction("b", label=0, probability=0.90),
        prediction("c", label=1, probability=0.51),
        prediction("d", label=0, probability=0.51),
    ]
    policy = SelectiveRiskPolicy(0.90, 0.75, 0.25, 0.10)

    report = audit_selective_risk(rows, policy)

    assert report.violations == (
        "coverage_below_minimum",
        "selective_risk_above_maximum",
        "aurc_above_maximum",
    )


def test_threshold_above_all_confidences_fails_closed() -> None:
    rows = [prediction("a", label=1, probability=0.8)]

    report = audit_selective_risk(rows, permissive_policy(threshold=0.9))

    assert report.selective_risk is None
    assert report.violations == ("no_automated_cases",)


def test_curve_keeps_equal_confidence_cases_in_one_group() -> None:
    rows = [
        prediction("a", label=1, probability=0.9),
        prediction("b", label=0, probability=0.1),
        prediction("c", label=0, probability=0.8),
    ]

    curve = risk_coverage_curve(rows)

    assert len(curve) == 2
    assert curve[0].automated_cases == 2
    assert curve[0].coverage == pytest.approx(2 / 3)
    assert curve[-1].coverage == 1


def test_aurc_rewards_ranking_correct_cases_above_errors() -> None:
    good_order = [
        prediction("a", label=1, probability=0.99),
        prediction("b", label=1, probability=0.90),
        prediction("c", label=1, probability=0.40),
    ]
    bad_order = [
        prediction("a", label=0, probability=0.99),
        prediction("b", label=1, probability=0.90),
        prediction("c", label=0, probability=0.40),
    ]

    assert area_under_risk_coverage(risk_coverage_curve(good_order)) < area_under_risk_coverage(
        risk_coverage_curve(bad_order)
    )


def test_report_is_input_order_independent() -> None:
    rows = [
        prediction("a", label=1, probability=0.9),
        prediction("b", label=1, probability=0.7),
        prediction("c", label=0, probability=0.2),
    ]

    assert audit_selective_risk(rows, permissive_policy()) == audit_selective_risk(
        list(reversed(rows)), permissive_policy()
    )


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([], "at least one"),
        (
            [
                prediction("duplicate", label=1, probability=0.9),
                prediction("duplicate", label=0, probability=0.1),
            ],
            "duplicate case_id",
        ),
        ([prediction("nan", label=1, probability=math.nan)], "probability must be finite"),
        ([prediction("range", label=1, probability=1.1)], "probability must be in"),
    ],
)
def test_malformed_evaluation_inputs_are_rejected(rows: list[Prediction], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        audit_selective_risk(rows, permissive_policy())


@pytest.mark.parametrize(
    "policy",
    [
        SelectiveRiskPolicy(math.nan, 0, 1, 1),
        SelectiveRiskPolicy(0.8, -0.1, 1, 1),
        SelectiveRiskPolicy(0.8, 0, 1.1, 1),
        SelectiveRiskPolicy(0.8, 0, 1, math.inf),
    ],
)
def test_invalid_policy_is_rejected(policy: SelectiveRiskPolicy) -> None:
    rows = [prediction("a", label=1, probability=0.9)]

    with pytest.raises(ValueError):
        audit_selective_risk(rows, policy)


def test_evidence_does_not_expose_case_ids() -> None:
    report = audit_selective_risk(
        [prediction("sensitive-case-id", label=1, probability=0.9)],
        permissive_policy(),
    )

    assert "sensitive-case-id" not in str(report.as_dict())
