from __future__ import annotations

import pytest

from evaluation import holm_bonferroni, paired_experiment_report, run_paired_responses
from experiment import Variant, generate_synthetic_predictions


def test_identical_variants_have_exactly_zero_paired_effect() -> None:
    predictions = generate_synthetic_predictions(count=200, seed=5)
    report = paired_experiment_report(
        predictions,
        baseline=Variant.EVIDENCE_LINKED,
        treatment=Variant.EVIDENCE_LINKED,
        seed=91,
        bootstrap_samples=200,
        randomization_samples=200,
    )

    for estimate in (
        report.human_accuracy,
        report.harmful_agreement,
        report.beneficial_override,
    ):
        assert estimate.absolute_effect == 0
        assert estimate.confidence_lower == 0
        assert estimate.confidence_upper == 0
        assert estimate.randomization_p_value == 1


def test_paired_response_draws_are_stable_when_input_order_changes() -> None:
    predictions = generate_synthetic_predictions(count=100, seed=8)
    forward = run_paired_responses(
        predictions,
        baseline=Variant.RECOMMENDATION_ONLY,
        treatment=Variant.WITH_CONFIDENCE,
        seed=44,
    )
    reverse = run_paired_responses(
        reversed(predictions),
        baseline=Variant.RECOMMENDATION_ONLY,
        treatment=Variant.WITH_CONFIDENCE,
        seed=44,
    )

    forward_by_case = {row.case_id: row for row in forward}
    reverse_by_case = {row.case_id: row for row in reverse}
    assert forward_by_case == reverse_by_case


def test_paired_report_is_reproducible_and_bounded() -> None:
    predictions = generate_synthetic_predictions(count=400, seed=12)
    kwargs = {
        "baseline": Variant.RECOMMENDATION_ONLY,
        "treatment": Variant.DEFER_LOW_CONFIDENCE,
        "seed": 88,
        "bootstrap_samples": 300,
        "randomization_samples": 400,
    }
    first = paired_experiment_report(predictions, **kwargs)
    second = paired_experiment_report(predictions, **kwargs)

    assert first == second
    assert first.pairs == 400
    assert -1 <= first.human_accuracy.absolute_effect <= 1
    assert 0 <= first.human_accuracy.randomization_p_value <= 1
    assert first.human_accuracy.confidence_lower <= first.human_accuracy.confidence_upper
    raw_by_metric = {
        estimate.metric: estimate.randomization_p_value
        for estimate in (
            first.human_accuracy,
            first.harmful_agreement,
            first.beneficial_override,
        )
    }
    assert first.familywise_error.method == "holm_bonferroni"
    assert {row.metric for row in first.familywise_error.hypotheses} == set(raw_by_metric)
    for row in first.familywise_error.hypotheses:
        assert row.raw_p_value == raw_by_metric[row.metric]
        assert row.adjusted_p_value >= row.raw_p_value
        assert row.rejected is (row.adjusted_p_value <= first.familywise_error.alpha)


def test_holm_bonferroni_controls_a_prespecified_family() -> None:
    report = holm_bonferroni(
        {"accuracy": 0.01, "harmful_agreement": 0.03, "override": 0.04},
        alpha=0.05,
    )

    adjusted = {row.metric: row for row in report.hypotheses}
    assert adjusted["accuracy"].adjusted_p_value == pytest.approx(0.03)
    assert adjusted["harmful_agreement"].adjusted_p_value == pytest.approx(0.06)
    assert adjusted["override"].adjusted_p_value == pytest.approx(0.06)
    assert {row.metric for row in report.hypotheses if row.rejected} == {"accuracy"}


@pytest.mark.parametrize("p_value", [-0.1, 1.1, float("nan"), float("inf")])
def test_holm_bonferroni_rejects_invalid_p_values(p_value: float) -> None:
    with pytest.raises(ValueError, match="p-values must be finite"):
        holm_bonferroni({"accuracy": p_value})


def test_duplicate_case_ids_are_rejected() -> None:
    prediction = generate_synthetic_predictions(count=1, seed=3)[0]
    with pytest.raises(ValueError, match="duplicate case_id"):
        run_paired_responses(
            [prediction, prediction],
            baseline=Variant.RECOMMENDATION_ONLY,
            treatment=Variant.EVIDENCE_LINKED,
            seed=1,
        )
