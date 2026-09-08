from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from statistics import mean

from experiment import HumanResponse, Prediction, Variant, simulate_human_response


@dataclass(frozen=True, slots=True)
class PairedObservation:
    case_id: str
    baseline: HumanResponse
    treatment: HumanResponse


@dataclass(frozen=True, slots=True)
class EffectEstimate:
    metric: str
    baseline_mean: float
    treatment_mean: float
    absolute_effect: float
    confidence_lower: float
    confidence_upper: float
    randomization_p_value: float
    pairs: int


@dataclass(frozen=True, slots=True)
class PairedExperimentReport:
    baseline_variant: Variant
    treatment_variant: Variant
    seed: int
    pairs: int
    human_accuracy: EffectEstimate
    harmful_agreement: EffectEstimate
    beneficial_override: EffectEstimate


Metric = Callable[[HumanResponse], float]


def _case_seed(experiment_seed: int, case_id: str) -> int:
    """Derive a stable per-case seed without Python's randomized string hash."""
    digest = hashlib.blake2b(
        f"{experiment_seed}:{case_id}".encode(),
        digest_size=8,
        person=b"rai-pair",
    ).digest()
    return int.from_bytes(digest, "big")


def run_paired_responses(
    predictions: Iterable[Prediction],
    *,
    baseline: Variant,
    treatment: Variant,
    seed: int,
    base_human_skill: float = 0.72,
    deferral_threshold: float = 0.72,
) -> list[PairedObservation]:
    """Run both variants against identical per-case latent reviewer draws.

    Each arm receives a fresh RNG initialized to the same case seed. This common
    random-number design reduces Monte Carlo noise while keeping cases independent.
    It does not turn synthetic responses into evidence about real human behavior.
    """
    rows: list[PairedObservation] = []
    seen: set[str] = set()
    for prediction in predictions:
        if prediction.case_id in seen:
            raise ValueError(f"duplicate case_id in paired experiment: {prediction.case_id}")
        seen.add(prediction.case_id)
        case_seed = _case_seed(seed, prediction.case_id)
        baseline_response = simulate_human_response(
            prediction,
            variant=baseline,
            rng=random.Random(case_seed),
            base_human_skill=base_human_skill,
            deferral_threshold=deferral_threshold,
        )
        treatment_response = simulate_human_response(
            prediction,
            variant=treatment,
            rng=random.Random(case_seed),
            base_human_skill=base_human_skill,
            deferral_threshold=deferral_threshold,
        )
        rows.append(
            PairedObservation(
                case_id=prediction.case_id,
                baseline=baseline_response,
                treatment=treatment_response,
            )
        )
    return rows


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def estimate_paired_effect(
    observations: Iterable[PairedObservation],
    *,
    metric_name: str,
    metric: Metric,
    seed: int,
    bootstrap_samples: int = 2000,
    randomization_samples: int = 4000,
    confidence_level: float = 0.95,
) -> EffectEstimate:
    rows = list(observations)
    if not rows:
        return EffectEstimate(metric_name, 0, 0, 0, 0, 0, 1, 0)
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    if randomization_samples < 100:
        raise ValueError("randomization_samples must be at least 100")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")

    baseline_values = [metric(row.baseline) for row in rows]
    treatment_values = [metric(row.treatment) for row in rows]
    differences = [
        treatment - baseline
        for baseline, treatment in zip(baseline_values, treatment_values, strict=True)
    ]
    effect = mean(differences)

    rng = random.Random(seed)
    size = len(rows)
    bootstrap = sorted(
        mean(differences[rng.randrange(size)] for _ in range(size))
        for _ in range(bootstrap_samples)
    )
    alpha = 1 - confidence_level
    lower = _quantile(bootstrap, alpha / 2)
    upper = _quantile(bootstrap, 1 - alpha / 2)

    observed = abs(effect)
    extreme = 0
    for _ in range(randomization_samples):
        randomized = mean(
            difference if rng.random() < 0.5 else -difference
            for difference in differences
        )
        extreme += abs(randomized) >= observed - 1e-12
    p_value = (extreme + 1) / (randomization_samples + 1)

    return EffectEstimate(
        metric=metric_name,
        baseline_mean=mean(baseline_values),
        treatment_mean=mean(treatment_values),
        absolute_effect=effect,
        confidence_lower=lower,
        confidence_upper=upper,
        randomization_p_value=p_value,
        pairs=size,
    )


def paired_experiment_report(
    predictions: Iterable[Prediction],
    *,
    baseline: Variant,
    treatment: Variant,
    seed: int,
    bootstrap_samples: int = 2000,
    randomization_samples: int = 4000,
) -> PairedExperimentReport:
    observations = run_paired_responses(
        predictions,
        baseline=baseline,
        treatment=treatment,
        seed=seed,
    )

    def estimate(name: str, metric: Metric, salt: int) -> EffectEstimate:
        return estimate_paired_effect(
            observations,
            metric_name=name,
            metric=metric,
            seed=seed + salt,
            bootstrap_samples=bootstrap_samples,
            randomization_samples=randomization_samples,
        )

    return PairedExperimentReport(
        baseline_variant=baseline,
        treatment_variant=treatment,
        seed=seed,
        pairs=len(observations),
        human_accuracy=estimate("human_accuracy", lambda row: float(row.human_correct), 11),
        harmful_agreement=estimate(
            "harmful_agreement",
            lambda row: float(row.agreed and not row.ai_correct),
            23,
        ),
        beneficial_override=estimate(
            "beneficial_override",
            lambda row: float(row.overridden and not row.ai_correct and row.human_correct),
            37,
        ),
    )
