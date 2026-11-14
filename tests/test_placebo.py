from __future__ import annotations

from dataclasses import asdict

import pytest
from fastapi.testclient import TestClient

from app.api import app
from experiment import Prediction
from placebo import explanation_placebo_audit


def _aligned_predictions(count: int = 400) -> list[Prediction]:
    rows: list[Prediction] = []
    for index in range(count):
        true_label = index % 2
        ai_correct = index < count // 2
        predicted_label = true_label if ai_correct else 1 - true_label
        rows.append(
            Prediction(
                case_id=f"case-{index:04d}",
                true_label=true_label,
                probability=0.9 if predicted_label else 0.1,
                evidence_quality=1.0,
                explanation_quality=1.0 if ai_correct else 0.0,
            )
        )
    return rows


def test_case_linked_explanations_outperform_placebo_when_quality_is_aligned() -> None:
    report = explanation_placebo_audit(
        _aligned_predictions(),
        seed=19,
        bootstrap_samples=500,
        randomization_samples=800,
        minimum_accuracy_lift=0.0,
    )

    assert report.passed is True
    assert report.human_accuracy.confidence_lower > 0
    assert report.harmful_agreement.confidence_upper <= 0
    assert len(report.evidence_sha256) == 64


def test_uniform_explanation_quality_fails_specificity_policy() -> None:
    predictions = [
        Prediction(
            case_id=f"case-{index:03d}",
            true_label=index % 2,
            probability=0.8 if index % 2 else 0.2,
            evidence_quality=0.8,
            explanation_quality=0.5,
        )
        for index in range(100)
    ]

    report = explanation_placebo_audit(
        predictions,
        seed=3,
        bootstrap_samples=200,
        randomization_samples=200,
    )

    assert report.passed is False
    assert report.human_accuracy.absolute_effect == 0
    assert report.harmful_agreement.absolute_effect == 0
    assert "accuracy_lift_below_policy" in report.failure_reasons


def test_placebo_audit_is_stable_under_input_reordering() -> None:
    predictions = _aligned_predictions(100)
    kwargs = {
        "seed": 31,
        "bootstrap_samples": 200,
        "randomization_samples": 300,
        "minimum_accuracy_lift": -1.0,
        "maximum_harmful_agreement_increase": 1.0,
    }

    forward = explanation_placebo_audit(predictions, **kwargs)
    reverse = explanation_placebo_audit(reversed(predictions), **kwargs)

    assert forward == reverse


def test_duplicate_case_ids_and_unbounded_work_are_rejected() -> None:
    predictions = _aligned_predictions(20)
    with pytest.raises(ValueError, match="duplicate case_id"):
        explanation_placebo_audit(
            [predictions[0], *predictions],
            bootstrap_samples=100,
            randomization_samples=100,
        )

    with pytest.raises(ValueError, match="resampling work"):
        explanation_placebo_audit(
            _aligned_predictions(2_000),
            bootstrap_samples=3_000,
            randomization_samples=100,
        )


def test_placebo_api_returns_aggregate_evidence_only() -> None:
    client = TestClient(app)
    predictions = _aligned_predictions(40)
    response = client.post(
        "/explanations/placebo-audit",
        json={
            "predictions": [asdict(prediction) for prediction in predictions],
            "seed": 5,
            "bootstrap_samples": 100,
            "randomization_samples": 100,
            "minimum_accuracy_lift": -1.0,
            "maximum_harmful_agreement_increase": 1.0,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["cases"] == 40
    assert payload["passed"] is True
    assert len(payload["evidence_sha256"]) == 64
    assert "predictions" not in payload
