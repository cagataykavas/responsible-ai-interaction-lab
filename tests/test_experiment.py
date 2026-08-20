from experiment import (
    Variant,
    calibration_report,
    compare_variants,
    evaluate_interactions,
    generate_synthetic_predictions,
    run_variant,
)


def test_calibration_report_is_bounded():
    predictions = generate_synthetic_predictions(count=300, seed=1)
    report = calibration_report(predictions, n_bins=8)
    assert 0 <= report.brier_score <= 1
    assert 0 <= report.expected_calibration_error <= 1
    assert report.bins


def test_all_variants_produce_metrics():
    predictions = generate_synthetic_predictions(count=250, seed=2)
    for index, variant in enumerate(Variant):
        metrics = run_variant(predictions, variant=variant, seed=10 + index)
        assert metrics.cases == 250
        assert 0 <= metrics.human_accuracy <= 1
        assert 0 <= metrics.agreement_rate <= 1


def test_deferral_variant_actually_defers_some_cases():
    predictions = generate_synthetic_predictions(count=500, seed=3)
    metrics = run_variant(
        predictions,
        variant=Variant.DEFER_LOW_CONFIDENCE,
        seed=33,
    )
    assert metrics.deferral_rate > 0


def test_comparison_contains_deltas():
    predictions = generate_synthetic_predictions(count=300, seed=4)
    baseline = run_variant(
        predictions,
        variant=Variant.RECOMMENDATION_ONLY,
        seed=44,
    )
    treatment = run_variant(
        predictions,
        variant=Variant.EVIDENCE_LINKED,
        seed=45,
    )
    comparison = compare_variants(baseline, treatment)
    assert comparison.baseline.variant is Variant.RECOMMENDATION_ONLY
    assert comparison.treatment.variant is Variant.EVIDENCE_LINKED


def test_empty_interaction_evaluation_is_safe():
    metrics = evaluate_interactions([], variant=Variant.RECOMMENDATION_ONLY)
    assert metrics.cases == 0
