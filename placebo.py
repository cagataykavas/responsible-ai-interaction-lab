from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace

from evaluation import EffectEstimate, PairedObservation, estimate_paired_effect
from experiment import Prediction, Variant, simulate_human_response

MIN_CASES = 20
MAX_CASES = 5_000
MAX_RESAMPLE_WORK = 5_000_000


@dataclass(frozen=True, slots=True)
class ExplanationPlaceboReport:
    cases: int
    seed: int
    bootstrap_samples: int
    randomization_samples: int
    confidence_level: float
    minimum_accuracy_lift: float
    maximum_harmful_agreement_increase: float
    passed: bool
    failure_reasons: tuple[str, ...]
    evidence_sha256: str
    human_accuracy: EffectEstimate
    harmful_agreement: EffectEstimate
    beneficial_override: EffectEstimate


def _stable_seed(seed: int, case_id: str) -> int:
    digest = hashlib.blake2b(
        f"{seed}:{case_id}".encode(),
        digest_size=8,
        person=b"rai-placebo",
    ).digest()
    return int.from_bytes(digest, "big")


def _validate(
    predictions: list[Prediction],
    *,
    seed: int,
    bootstrap_samples: int,
    randomization_samples: int,
    confidence_level: float,
    minimum_accuracy_lift: float,
    maximum_harmful_agreement_increase: float,
) -> None:
    if not MIN_CASES <= len(predictions) <= MAX_CASES:
        raise ValueError(f"predictions must contain between {MIN_CASES} and {MAX_CASES} cases")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    for name, value, maximum in (
        ("bootstrap_samples", bootstrap_samples, 10_000),
        ("randomization_samples", randomization_samples, 20_000),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not 100 <= value <= maximum:
            raise ValueError(f"{name} must be an integer between 100 and {maximum}")
    if len(predictions) * max(bootstrap_samples, randomization_samples) > MAX_RESAMPLE_WORK:
        raise ValueError("case count times resampling work must not exceed 5000000")
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float))
        or not math.isfinite(confidence_level)
        or not 0.5 <= confidence_level < 1.0
    ):
        raise ValueError("confidence_level must be in [0.5, 1.0)")
    for name, value in (
        ("minimum_accuracy_lift", minimum_accuracy_lift),
        ("maximum_harmful_agreement_increase", maximum_harmful_agreement_increase),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not -1.0 <= value <= 1.0
        ):
            raise ValueError(f"{name} must be a finite number in [-1.0, 1.0]")

    seen: set[str] = set()
    for prediction in predictions:
        if (
            not isinstance(prediction.case_id, str)
            or not prediction.case_id.strip()
            or len(prediction.case_id) > 200
        ):
            raise ValueError("case IDs must contain between 1 and 200 characters")
        if prediction.case_id in seen:
            raise ValueError(f"duplicate case_id in placebo audit: {prediction.case_id}")
        seen.add(prediction.case_id)
        if isinstance(prediction.true_label, bool) or prediction.true_label not in (0, 1):
            raise ValueError(f"case {prediction.case_id!r} has an invalid true label")
        for name, value in (
            ("probability", prediction.probability),
            ("evidence_quality", prediction.evidence_quality),
            ("explanation_quality", prediction.explanation_quality),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"case {prediction.case_id!r} has an invalid {name}")


def _sattolo_donors(size: int, seed: int) -> list[int]:
    """Return a deterministic single-cycle permutation with no fixed points."""
    donors = list(range(size))
    rng = random.Random(seed)
    for index in range(size - 1, 0, -1):
        other = rng.randrange(index)
        donors[index], donors[other] = donors[other], donors[index]
    return donors


def _evidence_digest(
    predictions: list[Prediction],
    donors: list[int],
    *,
    seed: int,
    bootstrap_samples: int,
    randomization_samples: int,
    confidence_level: float,
    minimum_accuracy_lift: float,
    maximum_harmful_agreement_increase: float,
) -> str:
    payload = {
        "schema": "explanation-placebo-audit/v1",
        "seed": seed,
        "bootstrap_samples": bootstrap_samples,
        "randomization_samples": randomization_samples,
        "confidence_level": confidence_level,
        "minimum_accuracy_lift": minimum_accuracy_lift,
        "maximum_harmful_agreement_increase": maximum_harmful_agreement_increase,
        "assignments": [
            {
                "prediction": asdict(prediction),
                "placebo_explanation_case_id": predictions[donor].case_id,
            }
            for prediction, donor in zip(predictions, donors, strict=True)
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def explanation_placebo_audit(
    predictions: Iterable[Prediction],
    *,
    seed: int = 0,
    bootstrap_samples: int = 2_000,
    randomization_samples: int = 4_000,
    confidence_level: float = 0.95,
    minimum_accuracy_lift: float = 0.01,
    maximum_harmful_agreement_increase: float = 0.0,
) -> ExplanationPlaceboReport:
    """Test case-linked explanations against a distribution-preserving placebo assignment.

    The treatment keeps each case's explanation quality. The placebo reassigns those
    qualities with a deterministic derangement while preserving their distribution.
    Both arms use identical per-case random draws and the same model recommendation.
    """
    rows = sorted(predictions, key=lambda prediction: prediction.case_id)
    _validate(
        rows,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
        randomization_samples=randomization_samples,
        confidence_level=confidence_level,
        minimum_accuracy_lift=minimum_accuracy_lift,
        maximum_harmful_agreement_increase=maximum_harmful_agreement_increase,
    )
    donors = _sattolo_donors(len(rows), seed)
    observations: list[PairedObservation] = []
    for index, prediction in enumerate(rows):
        placebo_prediction = replace(
            prediction,
            explanation_quality=rows[donors[index]].explanation_quality,
        )
        case_seed = _stable_seed(seed, prediction.case_id)
        placebo = simulate_human_response(
            placebo_prediction,
            variant=Variant.EVIDENCE_LINKED,
            rng=random.Random(case_seed),
        )
        linked = simulate_human_response(
            prediction,
            variant=Variant.EVIDENCE_LINKED,
            rng=random.Random(case_seed),
        )
        observations.append(PairedObservation(prediction.case_id, placebo, linked))

    def estimate(name: str, metric, salt: int) -> EffectEstimate:
        return estimate_paired_effect(
            observations,
            metric_name=name,
            metric=metric,
            seed=seed + salt,
            bootstrap_samples=bootstrap_samples,
            randomization_samples=randomization_samples,
            confidence_level=confidence_level,
        )

    human_accuracy = estimate("human_accuracy", lambda row: float(row.human_correct), 11)
    harmful_agreement = estimate(
        "harmful_agreement",
        lambda row: float(row.agreed and not row.ai_correct),
        23,
    )
    beneficial_override = estimate(
        "beneficial_override",
        lambda row: float(row.overridden and not row.ai_correct and row.human_correct),
        37,
    )
    failure_reasons: list[str] = []
    if human_accuracy.confidence_lower < minimum_accuracy_lift:
        failure_reasons.append("accuracy_lift_below_policy")
    if harmful_agreement.confidence_upper > maximum_harmful_agreement_increase:
        failure_reasons.append("harmful_agreement_above_policy")

    return ExplanationPlaceboReport(
        cases=len(rows),
        seed=seed,
        bootstrap_samples=bootstrap_samples,
        randomization_samples=randomization_samples,
        confidence_level=float(confidence_level),
        minimum_accuracy_lift=float(minimum_accuracy_lift),
        maximum_harmful_agreement_increase=float(maximum_harmful_agreement_increase),
        passed=not failure_reasons,
        failure_reasons=tuple(failure_reasons),
        evidence_sha256=_evidence_digest(
            rows,
            donors,
            seed=seed,
            bootstrap_samples=bootstrap_samples,
            randomization_samples=randomization_samples,
            confidence_level=float(confidence_level),
            minimum_accuracy_lift=float(minimum_accuracy_lift),
            maximum_harmful_agreement_increase=float(maximum_harmful_agreement_increase),
        ),
        human_accuracy=human_accuracy,
        harmful_agreement=harmful_agreement,
        beneficial_override=beneficial_override,
    )
