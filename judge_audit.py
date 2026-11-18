from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from math import exp, lgamma, log, sqrt


class CandidateOrder(str, Enum):
    AB = "ab"
    BA = "ba"


class Winner(str, Enum):
    FIRST = "first"
    SECOND = "second"
    TIE = "tie"


@dataclass(frozen=True, slots=True)
class PairwiseJudgment:
    case_id: str
    order: CandidateOrder
    winner: Winner


@dataclass(frozen=True, slots=True)
class OrderBiasReport:
    pairs: int
    judgments: int
    comparable_pairs: int
    candidate_consistent_pairs: int
    position_driven_pairs: int
    tie_consistent_pairs: int
    mixed_tie_pairs: int
    first_position_repeats: int
    second_position_repeats: int
    order_invariance_rate: float
    position_driven_rate: float | None
    position_driven_ci_lower: float | None
    position_driven_ci_upper: float | None
    directional_p_value: float | None


def _selected_candidate(judgment: PairwiseJudgment) -> str | None:
    if judgment.winner is Winner.TIE:
        return None
    if judgment.order is CandidateOrder.AB:
        return "a" if judgment.winner is Winner.FIRST else "b"
    return "b" if judgment.winner is Winner.FIRST else "a"


def _wilson_interval(successes: int, total: int, z: float) -> tuple[float, float]:
    if total == 0:
        raise ValueError("Wilson interval requires at least one observation")
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    margin = z * sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _two_sided_fair_binomial_p_value(successes: int, total: int) -> float:
    """Exact two-sided p-value for a fair first/second position split."""
    if total == 0:
        raise ValueError("binomial test requires at least one observation")
    tail = min(successes, total - successes)
    log_probabilities = [
        lgamma(total + 1) - lgamma(index + 1) - lgamma(total - index + 1) - total * log(2)
        for index in range(tail + 1)
    ]
    scale = max(log_probabilities)
    probability = 2 * exp(scale) * sum(exp(value - scale) for value in log_probabilities)
    return min(1.0, probability)


def audit_pairwise_order_bias(
    judgments: Iterable[PairwiseJudgment],
    *,
    confidence_z: float = 1.959963984540054,
) -> OrderBiasReport:
    """Audit pairwise judge results using one AB and one BA run per case.

    Candidate consistency means the same candidate wins after its position is
    swapped. A position-driven pair instead repeats the same winning position,
    which necessarily changes the selected candidate. Ties are retained rather
    than silently coerced into a preference.
    """
    if confidence_z <= 0:
        raise ValueError("confidence_z must be positive")

    rows = list(judgments)
    grouped: dict[str, dict[CandidateOrder, PairwiseJudgment]] = defaultdict(dict)
    for row in rows:
        if not row.case_id.strip():
            raise ValueError("case_id must not be blank")
        if row.order in grouped[row.case_id]:
            raise ValueError(f"duplicate {row.order.value} judgment for case_id: {row.case_id}")
        grouped[row.case_id][row.order] = row

    for case_id, pair in grouped.items():
        missing = set(CandidateOrder) - pair.keys()
        if missing:
            orders = ", ".join(sorted(order.value for order in missing))
            raise ValueError(f"case_id {case_id} is missing order: {orders}")

    candidate_consistent = 0
    position_driven = 0
    tie_consistent = 0
    mixed_tie = 0
    first_repeats = 0
    second_repeats = 0

    for pair in grouped.values():
        ab = pair[CandidateOrder.AB]
        ba = pair[CandidateOrder.BA]
        if ab.winner is Winner.TIE and ba.winner is Winner.TIE:
            tie_consistent += 1
            continue
        if ab.winner is Winner.TIE or ba.winner is Winner.TIE:
            mixed_tie += 1
            continue

        if _selected_candidate(ab) == _selected_candidate(ba):
            candidate_consistent += 1
            continue

        position_driven += 1
        if ab.winner is Winner.FIRST and ba.winner is Winner.FIRST:
            first_repeats += 1
        elif ab.winner is Winner.SECOND and ba.winner is Winner.SECOND:
            second_repeats += 1
        else:  # pragma: no cover - protected by candidate mapping above
            raise AssertionError("candidate switch must repeat a winning position")

    pairs = len(grouped)
    comparable = candidate_consistent + position_driven
    invariant = candidate_consistent + tie_consistent
    if comparable:
        rate = position_driven / comparable
        lower, upper = _wilson_interval(position_driven, comparable, confidence_z)
    else:
        rate = lower = upper = None

    directional_p_value = (
        _two_sided_fair_binomial_p_value(first_repeats, position_driven)
        if position_driven
        else None
    )
    return OrderBiasReport(
        pairs=pairs,
        judgments=len(rows),
        comparable_pairs=comparable,
        candidate_consistent_pairs=candidate_consistent,
        position_driven_pairs=position_driven,
        tie_consistent_pairs=tie_consistent,
        mixed_tie_pairs=mixed_tie,
        first_position_repeats=first_repeats,
        second_position_repeats=second_repeats,
        order_invariance_rate=invariant / pairs if pairs else 0.0,
        position_driven_rate=rate,
        position_driven_ci_lower=lower,
        position_driven_ci_upper=upper,
        directional_p_value=directional_p_value,
    )
