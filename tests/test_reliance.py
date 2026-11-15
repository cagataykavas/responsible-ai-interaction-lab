from __future__ import annotations

from dataclasses import replace

import pytest

from experiment import HumanResponse, Variant
from reliance import SelectiveReliancePolicy, selective_reliance_audit


def _response(
    case_id: str,
    *,
    ai_correct: bool,
    agree: bool,
    deferred: bool = False,
    variant: Variant = Variant.EVIDENCE_LINKED,
) -> HumanResponse:
    true_label = int(ai_correct)
    ai_label = 1
    human_label = ai_label if agree else 1 - ai_label
    return HumanResponse(
        case_id=case_id,
        variant=variant,
        ai_label=ai_label,
        human_label=human_label,
        true_label=true_label,
        confidence_seen=None,
        deferred=deferred,
    )


def _selective_cohort(size: int = 50) -> list[HumanResponse]:
    return [
        *[_response(f"correct-{index:03d}", ai_correct=True, agree=True) for index in range(size)],
        *[
            _response(f"incorrect-{index:03d}", ai_correct=False, agree=False)
            for index in range(size)
        ],
    ]


def test_selective_reliance_cohort_passes_uncertainty_gate() -> None:
    report = selective_reliance_audit(_selective_cohort())

    assert report.accepted
    assert report.reasons == ()
    assert report.correct_ai_reliance.rate == 1.0
    assert report.harmful_ai_reliance.rate == 0.0
    assert report.beneficial_override.rate == 1.0
    assert report.reliance_gap_lower > 0.80


def test_persuasive_wrong_advice_fails_harm_and_discrimination_gates() -> None:
    responses = [
        *[_response(f"correct-{index}", ai_correct=True, agree=True) for index in range(50)],
        *[_response(f"incorrect-{index}", ai_correct=False, agree=True) for index in range(50)],
    ]

    report = selective_reliance_audit(responses)

    assert not report.accepted
    assert "harmful_reliance_above_policy" in report.reasons
    assert "reliance_discrimination_below_policy" in report.reasons


def test_small_outcome_strata_fail_closed_without_threshold_claims() -> None:
    report = selective_reliance_audit(
        [
            _response("correct", ai_correct=True, agree=True),
            _response("wrong", ai_correct=False, agree=False),
        ]
    )

    assert not report.accepted
    assert report.reasons == (
        "insufficient_ai_correct_cases",
        "insufficient_ai_incorrect_cases",
    )


def test_deferred_cases_are_reported_but_not_counted_as_reliance() -> None:
    responses = [
        *_selective_cohort(20),
        _response("deferred-correct", ai_correct=True, agree=False, deferred=True),
        _response("deferred-wrong", ai_correct=False, agree=True, deferred=True),
    ]

    report = selective_reliance_audit(responses)

    assert report.accepted
    assert report.cases_seen == 42
    assert report.cases_evaluated == 40
    assert report.deferred_cases == 2
    assert report.correct_ai_reliance.trials == 20
    assert report.harmful_ai_reliance.trials == 20


def test_report_and_digest_are_stable_under_input_reordering() -> None:
    responses = _selective_cohort(25)

    forward = selective_reliance_audit(responses)
    reverse = selective_reliance_audit(list(reversed(responses)))

    assert forward == reverse
    assert len(forward.evidence_sha256) == 64


def test_digest_changes_when_case_evidence_changes_but_aggregates_do_not() -> None:
    first = _selective_cohort(25)
    second = [replace(row, case_id=f"renamed-{row.case_id}") for row in first]

    first_report = selective_reliance_audit(first)
    second_report = selective_reliance_audit(second)

    assert first_report.correct_ai_reliance == second_report.correct_ai_reliance
    assert first_report.evidence_sha256 != second_report.evidence_sha256


def test_duplicate_cases_and_mixed_variants_are_rejected() -> None:
    row = _response("duplicate", ai_correct=True, agree=True)
    with pytest.raises(ValueError, match="unique"):
        selective_reliance_audit([row, row])

    with pytest.raises(ValueError, match="one known"):
        selective_reliance_audit(
            [
                row,
                _response(
                    "other",
                    ai_correct=False,
                    agree=False,
                    variant=Variant.WITH_CONFIDENCE,
                ),
            ]
        )


def test_policy_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="confidence_level"):
        replace(SelectiveReliancePolicy(), confidence_level=1.0)
    with pytest.raises(ValueError, match="confidence_level"):
        replace(SelectiveReliancePolicy(), confidence_level=True)
    with pytest.raises(ValueError, match="case budgets"):
        replace(SelectiveReliancePolicy(), min_cases_per_ai_outcome=0)
