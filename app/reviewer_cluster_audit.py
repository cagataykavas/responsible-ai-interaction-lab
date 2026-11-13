from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

PASS_EXIT_CODE = 0
POLICY_FAILURE_EXIT_CODE = 2
INVALID_EVIDENCE_EXIT_CODE = 3
MAX_INPUT_BYTES = 4_000_000
MAX_OBSERVATIONS_LIMIT = 50_000
MAX_REVIEWERS_LIMIT = 10_000
MAX_BOOTSTRAP_SAMPLES = 5_000
MAX_BOOTSTRAP_WORK = 5_000_000
MAX_REPORTED_INFLUENCES = 10


class EvidenceError(ValueError):
    """Evaluation evidence is malformed or exceeds a bounded workload."""


def _integer(name: str, value: int, *, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvidenceError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise EvidenceError(f"{name} must be between {minimum} and {maximum}")


def _rate(name: str, value: float, *, inclusive_zero: bool = True) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{name} must be numeric")
    lower_valid = value >= 0 if inclusive_zero else value > 0
    if not math.isfinite(value) or not lower_valid or value > 1:
        qualifier = "between 0 and 1" if inclusive_zero else "greater than 0 and at most 1"
        raise EvidenceError(f"{name} must be finite and {qualifier}")


def _identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceError(f"{name} must be a non-empty string")
    if value != value.strip() or len(value) > 128:
        raise EvidenceError(f"{name} must be trimmed and at most 128 characters")
    if any(ord(character) < 33 or ord(character) > 126 for character in value):
        raise EvidenceError(f"{name} must contain printable ASCII without spaces")


@dataclass(frozen=True, slots=True)
class ReviewerDecision:
    decision_id: str
    reviewer_id: str
    baseline_success: bool
    treatment_success: bool

    def __post_init__(self) -> None:
        _identifier("decision_id", self.decision_id)
        _identifier("reviewer_id", self.reviewer_id)
        if not isinstance(self.baseline_success, bool):
            raise EvidenceError("baseline_success must be boolean")
        if not isinstance(self.treatment_success, bool):
            raise EvidenceError("treatment_success must be boolean")

    @property
    def difference(self) -> int:
        return int(self.treatment_success) - int(self.baseline_success)


@dataclass(frozen=True, slots=True)
class ReviewerClusterPolicy:
    min_observations: int = 40
    min_reviewers: int = 8
    max_reviewer_share: float = 0.25
    confidence: float = 0.95
    bootstrap_samples: int = 1_000
    seed: int = 17
    max_confidence_interval_width: float = 0.30
    max_leave_one_reviewer_shift: float = 0.15
    min_direction_stability: float = 0.80
    practical_effect_floor: float = 0.02
    max_observations: int = 5_000
    max_reviewers: int = 1_000

    def __post_init__(self) -> None:
        _integer(
            "min_observations", self.min_observations, minimum=2, maximum=MAX_OBSERVATIONS_LIMIT
        )
        _integer("min_reviewers", self.min_reviewers, minimum=2, maximum=MAX_REVIEWERS_LIMIT)
        _integer(
            "bootstrap_samples",
            self.bootstrap_samples,
            minimum=100,
            maximum=MAX_BOOTSTRAP_SAMPLES,
        )
        _integer("seed", self.seed, minimum=0, maximum=2**63 - 1)
        _integer(
            "max_observations",
            self.max_observations,
            minimum=self.min_observations,
            maximum=MAX_OBSERVATIONS_LIMIT,
        )
        _integer(
            "max_reviewers",
            self.max_reviewers,
            minimum=self.min_reviewers,
            maximum=MAX_REVIEWERS_LIMIT,
        )
        _rate("max_reviewer_share", self.max_reviewer_share, inclusive_zero=False)
        _rate("confidence", self.confidence, inclusive_zero=False)
        if self.confidence >= 1:
            raise EvidenceError("confidence must be less than 1")
        _rate("max_confidence_interval_width", self.max_confidence_interval_width)
        _rate("max_leave_one_reviewer_shift", self.max_leave_one_reviewer_shift)
        _rate("min_direction_stability", self.min_direction_stability)
        _rate("practical_effect_floor", self.practical_effect_floor)


@dataclass(frozen=True, slots=True)
class ReviewerInfluence:
    reviewer_ref: str
    observations: int
    observation_share: float
    leave_one_out_effect: float
    absolute_shift: float


@dataclass(frozen=True, slots=True)
class ClusterViolation:
    code: str


@dataclass(frozen=True, slots=True)
class ReviewerClusterReport:
    passed: bool
    observations: int
    reviewers: int
    baseline_success_rate: float
    treatment_success_rate: float
    paired_effect: float
    confidence_lower: float
    confidence_upper: float
    confidence_interval_width: float
    max_reviewer_share: float
    max_leave_one_reviewer_shift: float | None
    direction_stability: float | None
    top_influences: tuple[ReviewerInfluence, ...]
    violations: tuple[ClusterViolation, ...]
    evidence_digest: str
    policy: ReviewerClusterPolicy

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = "reviewer-cluster-stability-v1"
        return payload


def _quantile(sorted_values: list[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _reviewer_ref(reviewer_id: str) -> str:
    return hashlib.sha256(f"reviewer\0{reviewer_id}".encode()).hexdigest()[:16]


def _digest(observations: Sequence[ReviewerDecision], policy: ReviewerClusterPolicy) -> str:
    payload = {
        "observations": [
            asdict(row) for row in sorted(observations, key=lambda item: item.decision_id)
        ],
        "policy": asdict(policy),
        "schema_version": "reviewer-cluster-evidence-v1",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def audit_reviewer_clusters(
    observations: Sequence[ReviewerDecision],
    *,
    policy: ReviewerClusterPolicy | None = None,
) -> ReviewerClusterReport:
    active_policy = policy or ReviewerClusterPolicy()
    rows = tuple(observations)
    if not 2 <= len(rows) <= active_policy.max_observations:
        raise EvidenceError(
            f"observation count must be between 2 and {active_policy.max_observations}"
        )
    if not all(isinstance(row, ReviewerDecision) for row in rows):
        raise EvidenceError("observations must contain ReviewerDecision values")
    decision_ids = [row.decision_id for row in rows]
    if len(decision_ids) != len(set(decision_ids)):
        raise EvidenceError("decision_id values must be unique")

    clusters: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        clusters[row.reviewer_id].append(row.difference)
    if len(clusters) > active_policy.max_reviewers:
        raise EvidenceError("reviewer count exceeds policy maximum")
    differences = [row.difference for row in rows]
    total_difference = sum(differences)
    effect = mean(differences)
    baseline_rate = mean(row.baseline_success for row in rows)
    treatment_rate = mean(row.treatment_success for row in rows)
    reviewer_ids = sorted(clusters)
    if len(reviewer_ids) * active_policy.bootstrap_samples > MAX_BOOTSTRAP_WORK:
        raise EvidenceError(f"cluster bootstrap exceeds {MAX_BOOTSTRAP_WORK} reviewer draws")
    maximum_share = max(len(values) for values in clusters.values()) / len(rows)
    cluster_totals = {
        reviewer_id: (sum(values), len(values)) for reviewer_id, values in clusters.items()
    }

    rng = random.Random(active_policy.seed)
    bootstrap: list[float] = []
    for _ in range(active_policy.bootstrap_samples):
        sampled_total = 0
        sampled_count = 0
        for _ in reviewer_ids:
            cluster_total, cluster_count = cluster_totals[rng.choice(reviewer_ids)]
            sampled_total += cluster_total
            sampled_count += cluster_count
        bootstrap.append(sampled_total / sampled_count)
    bootstrap.sort()
    alpha = 1 - active_policy.confidence
    confidence_lower = _quantile(bootstrap, alpha / 2)
    confidence_upper = _quantile(bootstrap, 1 - alpha / 2)
    interval_width = confidence_upper - confidence_lower

    influences: list[ReviewerInfluence] = []
    direction_matches = 0
    for reviewer_id in reviewer_ids:
        reviewer_values = clusters[reviewer_id]
        remaining_count = len(rows) - len(reviewer_values)
        if remaining_count == 0:
            continue
        leave_one_out = (total_difference - sum(reviewer_values)) / remaining_count
        shift = abs(leave_one_out - effect)
        if (effect > 0 and leave_one_out > 0) or (effect < 0 and leave_one_out < 0):
            direction_matches += 1
        influences.append(
            ReviewerInfluence(
                reviewer_ref=_reviewer_ref(reviewer_id),
                observations=len(reviewer_values),
                observation_share=len(reviewer_values) / len(rows),
                leave_one_out_effect=leave_one_out,
                absolute_shift=shift,
            )
        )

    max_shift = max((item.absolute_shift for item in influences), default=None)
    direction_stability = None
    if abs(effect) >= active_policy.practical_effect_floor and influences:
        direction_stability = direction_matches / len(influences)

    violations: list[ClusterViolation] = []
    if len(rows) < active_policy.min_observations:
        violations.append(ClusterViolation("insufficient_observations"))
    if len(clusters) < active_policy.min_reviewers:
        violations.append(ClusterViolation("insufficient_reviewers"))
    if maximum_share > active_policy.max_reviewer_share:
        violations.append(ClusterViolation("reviewer_concentration_exceeded"))
    if interval_width > active_policy.max_confidence_interval_width:
        violations.append(ClusterViolation("confidence_interval_too_wide"))
    if max_shift is not None and max_shift > active_policy.max_leave_one_reviewer_shift:
        violations.append(ClusterViolation("leave_one_reviewer_shift_exceeded"))
    if (
        direction_stability is not None
        and direction_stability < active_policy.min_direction_stability
    ):
        violations.append(ClusterViolation("effect_direction_unstable"))

    return ReviewerClusterReport(
        passed=not violations,
        observations=len(rows),
        reviewers=len(clusters),
        baseline_success_rate=baseline_rate,
        treatment_success_rate=treatment_rate,
        paired_effect=effect,
        confidence_lower=confidence_lower,
        confidence_upper=confidence_upper,
        confidence_interval_width=interval_width,
        max_reviewer_share=maximum_share,
        max_leave_one_reviewer_shift=max_shift,
        direction_stability=direction_stability,
        top_influences=tuple(
            sorted(
                influences,
                key=lambda item: (-item.absolute_shift, item.reviewer_ref),
            )[:MAX_REPORTED_INFLUENCES]
        ),
        violations=tuple(violations),
        evidence_digest=_digest(rows, active_policy),
        policy=active_policy,
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load(path: Path) -> list[ReviewerDecision]:
    if path.is_symlink():
        raise EvidenceError("input must not be a symlink")
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise EvidenceError(f"input exceeds {MAX_INPUT_BYTES} bytes")
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=lambda value: (_ for _ in ()).throw(
            EvidenceError(f"non-finite JSON value: {value}")
        ),
    )
    if not isinstance(payload, list):
        raise EvidenceError("input must be a JSON array")
    expected = {"baseline_success", "decision_id", "reviewer_id", "treatment_success"}
    rows: list[ReviewerDecision] = []
    for value in payload:
        if not isinstance(value, dict) or set(value) != expected:
            raise EvidenceError(
                "each observation must contain exactly baseline_success, decision_id, "
                "reviewer_id, and treatment_success"
            )
        rows.append(ReviewerDecision(**value))
    return rows


def _write_atomic(path: Path, content: str) -> None:
    if path.is_symlink():
        raise EvidenceError("output must not be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit paired agent-evaluation outcomes for reviewer-cluster dependence."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-observations", type=int, default=40)
    parser.add_argument("--min-reviewers", type=int, default=8)
    parser.add_argument("--max-reviewer-share", type=float, default=0.25)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-confidence-interval-width", type=float, default=0.30)
    parser.add_argument("--max-leave-one-reviewer-shift", type=float, default=0.15)
    parser.add_argument("--min-direction-stability", type=float, default=0.80)
    parser.add_argument("--practical-effect-floor", type=float, default=0.02)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        policy = ReviewerClusterPolicy(
            min_observations=arguments.min_observations,
            min_reviewers=arguments.min_reviewers,
            max_reviewer_share=arguments.max_reviewer_share,
            bootstrap_samples=arguments.bootstrap_samples,
            seed=arguments.seed,
            max_confidence_interval_width=arguments.max_confidence_interval_width,
            max_leave_one_reviewer_shift=arguments.max_leave_one_reviewer_shift,
            min_direction_stability=arguments.min_direction_stability,
            practical_effect_floor=arguments.practical_effect_floor,
        )
        report = audit_reviewer_clusters(_load(arguments.input), policy=policy)
        rendered = json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n"
        if arguments.output:
            _write_atomic(arguments.output, rendered)
        else:
            print(rendered, end="")
    except (EvidenceError, json.JSONDecodeError, OSError, TypeError) as exc:
        print(f"reviewer cluster audit failed: {exc}", file=sys.stderr)
        return INVALID_EVIDENCE_EXIT_CODE
    return PASS_EXIT_CODE if report.passed else POLICY_FAILURE_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
