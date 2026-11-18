from __future__ import annotations

import pytest

from judge_audit import (
    CandidateOrder,
    PairwiseJudgment,
    Winner,
    audit_pairwise_order_bias,
)


def judgment(case_id: str, order: CandidateOrder, winner: Winner) -> PairwiseJudgment:
    return PairwiseJudgment(case_id=case_id, order=order, winner=winner)


def test_audit_distinguishes_candidate_consistency_from_position_repetition() -> None:
    report = audit_pairwise_order_bias(
        [
            judgment("stable-a", CandidateOrder.AB, Winner.FIRST),
            judgment("stable-a", CandidateOrder.BA, Winner.SECOND),
            judgment("stable-b", CandidateOrder.AB, Winner.SECOND),
            judgment("stable-b", CandidateOrder.BA, Winner.FIRST),
            judgment("first-bias", CandidateOrder.AB, Winner.FIRST),
            judgment("first-bias", CandidateOrder.BA, Winner.FIRST),
            judgment("second-bias", CandidateOrder.AB, Winner.SECOND),
            judgment("second-bias", CandidateOrder.BA, Winner.SECOND),
        ]
    )

    assert report.pairs == 4
    assert report.judgments == 8
    assert report.candidate_consistent_pairs == 2
    assert report.position_driven_pairs == 2
    assert report.first_position_repeats == 1
    assert report.second_position_repeats == 1
    assert report.order_invariance_rate == 0.5
    assert report.position_driven_rate == 0.5
    assert report.directional_p_value == 1.0
    assert report.position_driven_ci_lower < 0.5 < report.position_driven_ci_upper


def test_ties_are_retained_without_inflating_comparable_pairs() -> None:
    report = audit_pairwise_order_bias(
        [
            judgment("tie", CandidateOrder.AB, Winner.TIE),
            judgment("tie", CandidateOrder.BA, Winner.TIE),
            judgment("mixed", CandidateOrder.AB, Winner.TIE),
            judgment("mixed", CandidateOrder.BA, Winner.FIRST),
        ]
    )

    assert report.tie_consistent_pairs == 1
    assert report.mixed_tie_pairs == 1
    assert report.comparable_pairs == 0
    assert report.order_invariance_rate == 0.5
    assert report.position_driven_rate is None
    assert report.directional_p_value is None


def test_exact_directional_test_detects_one_sided_position_repetition() -> None:
    rows = [
        judgment(f"case-{index}", order, Winner.FIRST)
        for index in range(6)
        for order in CandidateOrder
    ]
    report = audit_pairwise_order_bias(rows)

    assert report.first_position_repeats == 6
    assert report.second_position_repeats == 0
    assert report.directional_p_value == pytest.approx(0.03125)


@pytest.mark.parametrize(
    "rows, message",
    [
        (
            [judgment("incomplete", CandidateOrder.AB, Winner.FIRST)],
            "missing order: ba",
        ),
        (
            [
                judgment("duplicate", CandidateOrder.AB, Winner.FIRST),
                judgment("duplicate", CandidateOrder.AB, Winner.SECOND),
            ],
            "duplicate ab judgment",
        ),
        (
            [
                judgment(" ", CandidateOrder.AB, Winner.FIRST),
                judgment(" ", CandidateOrder.BA, Winner.SECOND),
            ],
            "case_id must not be blank",
        ),
    ],
)
def test_invalid_pairing_fails_closed(rows: list[PairwiseJudgment], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        audit_pairwise_order_bias(rows)


def test_empty_audit_is_explicitly_zero_without_inference() -> None:
    report = audit_pairwise_order_bias([])
    assert report.pairs == 0
    assert report.order_invariance_rate == 0
    assert report.position_driven_rate is None
