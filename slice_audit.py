"""Subgroup audits for human-AI interaction harms."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from math import isfinite

from experiment import HumanResponse


@dataclass(frozen=True, slots=True)
class SliceMetrics:
    group: str
    cases: int
    human_accuracy: float
    harmful_agreement_rate: float
    beneficial_override_rate: float
    deferral_rate: float

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SliceAudit:
    eligible_cases: int
    excluded_cases: int
    minimum_group_size: int
    slices: tuple[SliceMetrics, ...]
    worst_harm_group: str
    harmful_agreement_gap: float
    human_accuracy_gap: float

    def to_dict(self) -> dict[str, object]:
        return {
            "eligible_cases": self.eligible_cases,
            "excluded_cases": self.excluded_cases,
            "minimum_group_size": self.minimum_group_size,
            "slices": [item.to_dict() for item in self.slices],
            "worst_harm_group": self.worst_harm_group,
            "harmful_agreement_gap": self.harmful_agreement_gap,
            "human_accuracy_gap": self.human_accuracy_gap,
        }


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator


def audit_interaction_slices(
    responses: Iterable[HumanResponse],
    group_by_case: Mapping[str, str],
    *,
    minimum_group_size: int = 30,
) -> SliceAudit:
    """Expose interaction harms hidden by aggregate experiment metrics.

    Groups smaller than minimum_group_size are excluded from disparity
    comparisons and counted explicitly. This is a reporting guardrail, not a
    claim that the threshold provides adequate statistical power.
    """
    if not isinstance(minimum_group_size, int) or isinstance(minimum_group_size, bool):
        raise ValueError("minimum_group_size must be an integer")
    if minimum_group_size < 2:
        raise ValueError("minimum_group_size must be at least 2")

    rows = list(responses)
    case_ids = [row.case_id for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("response case_id values must be unique")
    missing = sorted(set(case_ids) - set(group_by_case))
    if missing:
        preview = ", ".join(missing[:3])
        raise ValueError(f"missing group labels for cases: {preview}")

    grouped: dict[str, list[HumanResponse]] = defaultdict(list)
    for row in rows:
        group = group_by_case[row.case_id]
        if not isinstance(group, str) or not group.strip():
            raise ValueError(f"group label for {row.case_id} must be a non-empty string")
        grouped[group].append(row)

    metrics: list[SliceMetrics] = []
    excluded = 0
    for group, group_rows in sorted(grouped.items()):
        count = len(group_rows)
        if count < minimum_group_size:
            excluded += count
            continue
        metric = SliceMetrics(
            group=group,
            cases=count,
            human_accuracy=_rate(sum(row.human_correct for row in group_rows), count),
            harmful_agreement_rate=_rate(
                sum(row.agreed and not row.ai_correct for row in group_rows), count
            ),
            beneficial_override_rate=_rate(
                sum(
                    row.overridden and not row.ai_correct and row.human_correct
                    for row in group_rows
                ),
                count,
            ),
            deferral_rate=_rate(sum(row.deferred for row in group_rows), count),
        )
        if not all(
            isfinite(value)
            for value in (
                metric.human_accuracy,
                metric.harmful_agreement_rate,
                metric.beneficial_override_rate,
                metric.deferral_rate,
            )
        ):
            raise ValueError("slice metrics must be finite")
        metrics.append(metric)

    if not metrics:
        raise ValueError("no group meets minimum_group_size")

    harmful_values = [item.harmful_agreement_rate for item in metrics]
    accuracy_values = [item.human_accuracy for item in metrics]
    worst = max(metrics, key=lambda item: (item.harmful_agreement_rate, item.group))
    return SliceAudit(
        eligible_cases=sum(item.cases for item in metrics),
        excluded_cases=excluded,
        minimum_group_size=minimum_group_size,
        slices=tuple(metrics),
        worst_harm_group=worst.group,
        harmful_agreement_gap=max(harmful_values) - min(harmful_values),
        human_accuracy_gap=max(accuracy_values) - min(accuracy_values),
    )
