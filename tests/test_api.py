from fastapi.testclient import TestClient

from app.api import app


client = TestClient(app)


def predictions_payload() -> list[dict]:
    return [
        {
            "case_id": "a",
            "true_label": 1,
            "probability": 0.90,
            "evidence_quality": 0.95,
            "explanation_quality": 0.90,
        },
        {
            "case_id": "b",
            "true_label": 0,
            "probability": 0.25,
            "evidence_quality": 0.80,
            "explanation_quality": 0.75,
        },
        {
            "case_id": "c",
            "true_label": 1,
            "probability": 0.58,
            "evidence_quality": 0.70,
            "explanation_quality": 0.65,
        },
    ]


def test_calibration_endpoint() -> None:
    response = client.post(
        "/calibration",
        json={"predictions": predictions_payload(), "bins": 5},
    )
    assert response.status_code == 200
    report = response.json()
    assert 0 <= report["brier_score"] <= 1
    assert 0 <= report["expected_calibration_error"] <= 1
    assert report["bins"]


def test_deferral_variant_returns_metrics() -> None:
    response = client.post(
        "/interaction",
        json={
            "predictions": predictions_payload(),
            "variant": "defer_low_confidence",
            "seed": 7,
            "deferral_threshold": 0.75,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["metrics"]["variant"] == "defer_low_confidence"
    assert payload["metrics"]["cases"] == 3
    assert len(payload["responses"]) == 3


def test_synthetic_comparison_is_reproducible() -> None:
    request = {
        "count": 200,
        "seed": 123,
        "baseline": "recommendation_only",
        "treatment": "evidence_linked",
    }
    first = client.post("/synthetic/compare", json=request)
    second = client.post("/synthetic/compare", json=request)
    assert first.status_code == 200
    assert first.json() == second.json()


def test_paired_comparison_returns_uncertainty_and_randomization_test() -> None:
    response = client.post(
        "/synthetic/paired-compare",
        json={
            "count": 200,
            "seed": 41,
            "baseline": "recommendation_only",
            "treatment": "defer_low_confidence",
            "bootstrap_samples": 200,
            "randomization_samples": 300,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["pairs"] == 200
    assert payload["human_accuracy"]["metric"] == "human_accuracy"
    assert 0 <= payload["human_accuracy"]["randomization_p_value"] <= 1
    assert (
        payload["human_accuracy"]["confidence_lower"]
        <= payload["human_accuracy"]["confidence_upper"]
    )
