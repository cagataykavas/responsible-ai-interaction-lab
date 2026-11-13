from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from app.reviewer_cluster_audit import (
    INVALID_EVIDENCE_EXIT_CODE,
    PASS_EXIT_CODE,
    POLICY_FAILURE_EXIT_CODE,
    EvidenceError,
    ReviewerClusterPolicy,
    ReviewerDecision,
    audit_reviewer_clusters,
    main,
)


def balanced_observations() -> list[ReviewerDecision]:
    rows: list[ReviewerDecision] = []
    for reviewer in range(10):
        for decision in range(10):
            rows.append(
                ReviewerDecision(
                    decision_id=f"decision-{reviewer}-{decision}",
                    reviewer_id=f"reviewer-{reviewer}",
                    baseline_success=decision < 4,
                    treatment_success=decision < 6,
                )
            )
    return rows


def test_balanced_reviewer_clusters_produce_stable_effect() -> None:
    report = audit_reviewer_clusters(balanced_observations())

    assert report.passed
    assert report.observations == 100
    assert report.reviewers == 10
    assert report.paired_effect == pytest.approx(0.2)
    assert report.confidence_lower == pytest.approx(0.2)
    assert report.confidence_upper == pytest.approx(0.2)
    assert report.direction_stability == 1.0
    assert report.max_reviewer_share == 0.1
    rendered = json.dumps(report.as_dict())
    assert "reviewer-0" not in rendered


def test_reviewer_concentration_fails_even_with_many_decisions() -> None:
    rows: list[ReviewerDecision] = []
    counts = [40, 2, 2, 2, 2, 2, 2, 2]
    for reviewer, count in enumerate(counts):
        for decision in range(count):
            rows.append(
                ReviewerDecision(
                    f"d-{reviewer}-{decision}",
                    f"r-{reviewer}",
                    False,
                    True,
                )
            )

    report = audit_reviewer_clusters(rows)

    assert {violation.code for violation in report.violations} == {
        "reviewer_concentration_exceeded"
    }


def test_insufficient_reviewers_is_policy_failure_not_invalid_evidence() -> None:
    rows = [
        ReviewerDecision(f"d-{index}", f"r-{index % 4}", False, index % 2 == 0)
        for index in range(40)
    ]

    report = audit_reviewer_clusters(rows)

    assert not report.passed
    assert "insufficient_reviewers" in {item.code for item in report.violations}


def test_influential_reviewer_triggers_leave_one_out_and_direction_gates() -> None:
    rows: list[ReviewerDecision] = []
    for reviewer in range(8):
        for decision in range(5):
            treatment = reviewer == 0
            rows.append(
                ReviewerDecision(
                    f"d-{reviewer}-{decision}",
                    f"r-{reviewer}",
                    False,
                    treatment,
                )
            )
    policy = ReviewerClusterPolicy(
        max_leave_one_reviewer_shift=0.10,
        min_direction_stability=0.90,
    )

    report = audit_reviewer_clusters(rows, policy=policy)

    assert not report.passed
    assert {item.code for item in report.violations} >= {
        "leave_one_reviewer_shift_exceeded",
        "effect_direction_unstable",
    }


def test_duplicate_decisions_are_invalid_evidence() -> None:
    row = ReviewerDecision("same", "reviewer", False, True)
    with pytest.raises(EvidenceError, match="decision_id values must be unique"):
        audit_reviewer_clusters([row, row])


def test_input_order_does_not_change_report_or_digest() -> None:
    rows = balanced_observations()

    first = audit_reviewer_clusters(rows)
    second = audit_reviewer_clusters(list(reversed(rows)))

    assert first == second


def test_cli_distinguishes_pass_policy_failure_and_invalid_evidence(tmp_path: Path) -> None:
    rows = balanced_observations()
    passing = tmp_path / "passing.json"
    failing = tmp_path / "failing.json"
    invalid = tmp_path / "invalid.json"
    output = tmp_path / "report.json"
    payload = json.dumps([asdict(row) for row in rows])
    passing.write_text(payload, encoding="utf-8")
    failing.write_text(payload, encoding="utf-8")
    invalid.write_text('[{"decision_id": "a", "decision_id": "b"}]', encoding="utf-8")

    assert main([str(passing), "--output", str(output)]) == PASS_EXIT_CODE
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True
    assert main([str(failing), "--min-reviewers", "20"]) == POLICY_FAILURE_EXIT_CODE
    assert main([str(invalid)]) == INVALID_EVIDENCE_EXIT_CODE
