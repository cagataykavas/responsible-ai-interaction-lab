from __future__ import annotations

import random
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from evaluation import paired_experiment_report
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
from placebo import explanation_placebo_audit


class PredictionInput(BaseModel):
    case_id: str = Field(min_length=1, max_length=200)
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


class PairedExperimentRequest(SyntheticExperimentRequest):
    bootstrap_samples: int = Field(default=1000, ge=100, le=10000)
    randomization_samples: int = Field(default=2000, ge=100, le=20000)


class ExplanationPlaceboRequest(BaseModel):
    predictions: list[PredictionInput] = Field(min_length=20, max_length=5000)
    seed: int = Field(default=0, ge=0, lt=2**63)
    bootstrap_samples: int = Field(default=2000, ge=100, le=10000)
    randomization_samples: int = Field(default=4000, ge=100, le=20000)
    confidence_level: float = Field(default=0.95, ge=0.5, lt=1.0)
    minimum_accuracy_lift: float = Field(default=0.01, ge=-1.0, le=1.0)
    maximum_harmful_agreement_increase: float = Field(default=0.0, ge=-1.0, le=1.0)


def _prediction(row: PredictionInput) -> Prediction:
    return Prediction(**row.model_dump())


app = FastAPI(
    title="Responsible AI Interaction Experiments",
    version="0.4.0",
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


@app.post("/synthetic/paired-compare")
def synthetic_paired_compare(request: PairedExperimentRequest) -> dict:
    predictions = generate_synthetic_predictions(count=request.count, seed=request.seed)
    report = paired_experiment_report(
        predictions,
        baseline=request.baseline,
        treatment=request.treatment,
        seed=request.seed,
        bootstrap_samples=request.bootstrap_samples,
        randomization_samples=request.randomization_samples,
    )
    return asdict(report)


@app.post("/explanations/placebo-audit")
def explanation_placebo(request: ExplanationPlaceboRequest) -> dict:
    try:
        report = explanation_placebo_audit(
            [_prediction(row) for row in request.predictions],
            seed=request.seed,
            bootstrap_samples=request.bootstrap_samples,
            randomization_samples=request.randomization_samples,
            confidence_level=request.confidence_level,
            minimum_accuracy_lift=request.minimum_accuracy_lift,
            maximum_harmful_agreement_increase=request.maximum_harmful_agreement_increase,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return asdict(report)
