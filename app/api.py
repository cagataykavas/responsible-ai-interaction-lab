from __future__ import annotations

from dataclasses import asdict
from fastapi import FastAPI
from pydantic import BaseModel, Field

from experiment import (
    Prediction,
    Variant,
    calibration_report,
    compare_variants,
    evaluate_interactions,
    generate_synthetic_predictions,
    run_variant,
    simulate_human_response,
)
import random


class PredictionInput(BaseModel):
    case_id: str
    true_label: int = Field(ge=0, le=1)
    probability: float = Field(ge=0.0, le=1.0)
    evidence_quality: float = Field(ge=0.0, le=1.0)
    explanation_quality: float = Field(ge=0.0, le=1.0)


class CalibrationRequest(BaseModel):
    predictions: list[PredictionInput]
    bins: int = Field(default=10, ge=2, le=50)


class InteractionExperimentRequest(BaseModel):
    predictions: list[PredictionInput]
    variant: Variant
    seed: int = 42
    base_human_skill: float = Field(default=0.72, ge=0.0, le=1.0)
    deferral_threshold: float = Field(default=0.72, ge=0.5, le=1.0)


class SyntheticExperimentRequest(BaseModel):
    count: int = Field(default=2000, ge=50, le=100000)
    seed: int = 42
    baseline: Variant = Variant.RECOMMENDATION_ONLY
    treatment: Variant = Variant.DEFER_LOW_CONFIDENCE


def _prediction(row: PredictionInput) -> Prediction:
    return Prediction(**row.model_dump())


app = FastAPI(
    title="Responsible AI Interaction Experiments",
    version="0.2.0",
    description=(
        "Executable experiments for calibration, deferral, automation bias and "
        "human-AI interaction variants using synthetic data."
    ),
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/calibration")
def calibration(request: CalibrationRequest) -> dict:
    report = calibration_report(
        [_prediction(row) for row in request.predictions],
        n_bins=request.bins,
    )
    return asdict(report)


@app.post("/interaction")
def interaction(request: InteractionExperimentRequest) -> dict:
    rng = random.Random(request.seed)
    rows = [
        simulate_human_response(
            _prediction(row),
            variant=request.variant,
            rng=rng,
            base_human_skill=request.base_human_skill,
            deferral_threshold=request.deferral_threshold,
        )
        for row in request.predictions
    ]
    return {
        "metrics": asdict(evaluate_interactions(rows, variant=request.variant)),
        "responses": [asdict(row) for row in rows],
    }


@app.post("/synthetic/compare")
def synthetic_compare(request: SyntheticExperimentRequest) -> dict:
    predictions = generate_synthetic_predictions(count=request.count, seed=request.seed)
    baseline = run_variant(
        predictions,
        variant=request.baseline,
        seed=request.seed + 101,
    )
    treatment = run_variant(
        predictions,
        variant=request.treatment,
        seed=request.seed + 202,
    )
    return {
        "calibration": asdict(calibration_report(predictions)),
        "comparison": asdict(compare_variants(baseline, treatment)),
    }
