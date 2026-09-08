from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from math import sqrt
from statistics import mean


class Variant(str, Enum):
    RECOMMENDATION_ONLY = "recommendation_only"
    WITH_CONFIDENCE = "with_confidence"
    EVIDENCE_LINKED = "evidence_linked"
    DEFER_LOW_CONFIDENCE = "defer_low_confidence"


@dataclass(frozen=True)
class Prediction:
    case_id: str
    true_label: int
    probability: float
    evidence_quality: float
    explanation_quality: float

    @property
    def predicted_label(self) -> int:
        return int(self.probability >= 0.5)

    @property
    def confidence(self) -> float:
        return self.probability if self.predicted_label == 1 else 1.0 - self.probability

    @property
    def correct(self) -> bool:
        return self.predicted_label == self.true_label


@dataclass(frozen=True)
class HumanResponse:
    case_id: str
    variant: Variant
    ai_label: int
    human_label: int
    true_label: int
    confidence_seen: float | None
    deferred: bool

    @property
    def ai_correct(self) -> bool:
        return self.ai_label == self.true_label

    @property
    def human_correct(self) -> bool:
        return self.human_label == self.true_label

    @property
    def agreed(self) -> bool:
        return self.ai_label == self.human_label

    @property
    def overridden(self) -> bool:
        return not self.agreed


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_probability: float
    positive_rate: float
    absolute_gap: float


@dataclass(frozen=True)
class CalibrationReport:
    brier_score: float
    expected_calibration_error: float
    bins: tuple[CalibrationBin, ...]


@dataclass(frozen=True)
class InteractionMetrics:
    variant: Variant
    cases: int
    ai_accuracy: float
    human_accuracy: float
    agreement_rate: float
    override_rate: float
    harmful_agreement_rate: float
    beneficial_override_rate: float
    harmful_override_rate: float
    deferral_rate: float


@dataclass(frozen=True)
class ExperimentComparison:
    baseline: InteractionMetrics
    treatment: InteractionMetrics
    human_accuracy_delta: float
    agreement_delta: float
    harmful_agreement_delta: float
    override_delta: float


def brier_score(predictions: Iterable[Prediction]) -> float:
    rows = list(predictions)
    if not rows:
        return 0.0
    return mean((row.probability - row.true_label) ** 2 for row in rows)


def calibration_report(
    predictions: Iterable[Prediction],
    *,
    n_bins: int = 10,
) -> CalibrationReport:
    rows = list(predictions)
    if not rows:
        return CalibrationReport(0.0, 0.0, ())

    bins: list[CalibrationBin] = []
    weighted_gap = 0.0
    for index in range(n_bins):
        lower = index / n_bins
        upper = (index + 1) / n_bins
        if index == n_bins - 1:
            bucket = [row for row in rows if lower <= row.probability <= upper]
        else:
            bucket = [row for row in rows if lower <= row.probability < upper]
        if not bucket:
            continue

        mean_probability = mean(row.probability for row in bucket)
        positive_rate = mean(row.true_label for row in bucket)
        gap = abs(mean_probability - positive_rate)
        weighted_gap += len(bucket) / len(rows) * gap
        bins.append(
            CalibrationBin(
                lower=lower,
                upper=upper,
                count=len(bucket),
                mean_probability=mean_probability,
                positive_rate=positive_rate,
                absolute_gap=gap,
            )
        )

    return CalibrationReport(
        brier_score=brier_score(rows),
        expected_calibration_error=weighted_gap,
        bins=tuple(bins),
    )


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def simulate_human_response(
    prediction: Prediction,
    *,
    variant: Variant,
    rng: random.Random,
    base_human_skill: float = 0.72,
    deferral_threshold: float = 0.72,
) -> HumanResponse:
    """Synthetic reviewer simulation for interaction-design experiments.

    This is not a behavioral-science model. It exists to make metrics and
    experiment pipelines executable without claiming real human-study data.
    """
    deferred = (
        variant is Variant.DEFER_LOW_CONFIDENCE
        and prediction.confidence < deferral_threshold
    )

    # Independent human judgment starts from a configurable skill level and is
    # slightly improved when evidence quality is high.
    independent_skill = _clamp(
        base_human_skill + 0.10 * (prediction.evidence_quality - 0.5)
    )
    independently_correct = rng.random() < independent_skill
    independent_label = (
        prediction.true_label if independently_correct else 1 - prediction.true_label
    )

    if deferred:
        final_label = independent_label
        return HumanResponse(
            case_id=prediction.case_id,
            variant=variant,
            ai_label=prediction.predicted_label,
            human_label=final_label,
            true_label=prediction.true_label,
            confidence_seen=None,
            deferred=True,
        )

    # The interface changes the probability of following the AI recommendation.
    if variant is Variant.RECOMMENDATION_ONLY:
        follow_probability = 0.58
    elif variant is Variant.WITH_CONFIDENCE:
        # A visible confidence score creates stronger anchoring as confidence rises.
        follow_probability = 0.42 + 0.48 * prediction.confidence
    elif variant is Variant.EVIDENCE_LINKED:
        # Evidence-linked explanations increase agreement only when explanation
        # quality and evidence quality are both strong.
        explanation_support = sqrt(
            max(0.0, prediction.explanation_quality * prediction.evidence_quality)
        )
        follow_probability = 0.38 + 0.38 * explanation_support
    else:
        follow_probability = 0.55

    follow_probability = _clamp(follow_probability, 0.05, 0.97)
    if rng.random() < follow_probability:
        final_label = prediction.predicted_label
    else:
        final_label = independent_label

    return HumanResponse(
        case_id=prediction.case_id,
        variant=variant,
        ai_label=prediction.predicted_label,
        human_label=final_label,
        true_label=prediction.true_label,
        confidence_seen=(
            prediction.confidence
            if variant is Variant.WITH_CONFIDENCE
            else None
        ),
        deferred=False,
    )


def evaluate_interactions(
    responses: Iterable[HumanResponse],
    *,
    variant: Variant,
) -> InteractionMetrics:
    rows = list(responses)
    if not rows:
        return InteractionMetrics(variant, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    cases = len(rows)
    harmful_agreements = sum(
        row.agreed and not row.ai_correct for row in rows
    )
    beneficial_overrides = sum(
        row.overridden and not row.ai_correct and row.human_correct for row in rows
    )
    harmful_overrides = sum(
        row.overridden and row.ai_correct and not row.human_correct for row in rows
    )

    return InteractionMetrics(
        variant=variant,
        cases=cases,
        ai_accuracy=sum(row.ai_correct for row in rows) / cases,
        human_accuracy=sum(row.human_correct for row in rows) / cases,
        agreement_rate=sum(row.agreed for row in rows) / cases,
        override_rate=sum(row.overridden for row in rows) / cases,
        harmful_agreement_rate=harmful_agreements / cases,
        beneficial_override_rate=beneficial_overrides / cases,
        harmful_override_rate=harmful_overrides / cases,
        deferral_rate=sum(row.deferred for row in rows) / cases,
    )


def compare_variants(
    baseline: InteractionMetrics,
    treatment: InteractionMetrics,
) -> ExperimentComparison:
    return ExperimentComparison(
        baseline=baseline,
        treatment=treatment,
        human_accuracy_delta=treatment.human_accuracy - baseline.human_accuracy,
        agreement_delta=treatment.agreement_rate - baseline.agreement_rate,
        harmful_agreement_delta=(
            treatment.harmful_agreement_rate - baseline.harmful_agreement_rate
        ),
        override_delta=treatment.override_rate - baseline.override_rate,
    )


def generate_synthetic_predictions(
    *,
    count: int = 2000,
    seed: int = 42,
) -> list[Prediction]:
    rng = random.Random(seed)
    rows: list[Prediction] = []
    for index in range(count):
        latent = rng.random()
        true_label = int(latent > 0.48)

        # Build a probability that is useful but intentionally imperfectly calibrated.
        evidence_quality = rng.uniform(0.45, 1.0)
        explanation_quality = rng.uniform(0.35, 1.0)
        signal = (
            0.22
            + 0.56 * true_label
            + rng.gauss(0, 0.20)
            + 0.08 * (evidence_quality - 0.5)
        )
        probability = _clamp(signal, 0.01, 0.99)
        rows.append(
            Prediction(
                case_id=f"case-{index:05d}",
                true_label=true_label,
                probability=probability,
                evidence_quality=evidence_quality,
                explanation_quality=explanation_quality,
            )
        )
    return rows


def run_variant(
    predictions: list[Prediction],
    *,
    variant: Variant,
    seed: int,
) -> InteractionMetrics:
    rng = random.Random(seed)
    responses = [
        simulate_human_response(prediction, variant=variant, rng=rng)
        for prediction in predictions
    ]
    return evaluate_interactions(responses, variant=variant)


def main() -> None:
    predictions = generate_synthetic_predictions()
    calibration = calibration_report(predictions)

    print("Calibration")
    print(f"Brier score: {calibration.brier_score:.4f}")
    print(f"ECE:         {calibration.expected_calibration_error:.4f}")

    results = {
        variant: run_variant(predictions, variant=variant, seed=100 + index)
        for index, variant in enumerate(Variant)
    }

    print("\nInteraction experiments")
    for variant, metrics in results.items():
        print(
            f"{variant.value:24s} "
            f"human_acc={metrics.human_accuracy:.3f} "
            f"agree={metrics.agreement_rate:.3f} "
            f"harmful_agree={metrics.harmful_agreement_rate:.3f} "
            f"defer={metrics.deferral_rate:.3f}"
        )

    comparison = compare_variants(
        results[Variant.RECOMMENDATION_ONLY],
        results[Variant.EVIDENCE_LINKED],
    )
    print("\nEvidence-linked vs recommendation-only")
    print(comparison)


if __name__ == "__main__":
    main()
