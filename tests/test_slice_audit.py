import pytest

from experiment import HumanResponse, Variant
from slice_audit import audit_interaction_slices


def response(case_id, *, ai_label=1, human_label=1, true_label=1, deferred=False):
    return HumanResponse(
        case_id=case_id,
        variant=Variant.EVIDENCE_LINKED,
        ai_label=ai_label,
        human_label=human_label,
        true_label=true_label,
        confidence_seen=None,
        deferred=deferred,
    )


def test_surfaces_worst_harm_group_and_disparity():
    rows = [
        response("a1", ai_label=1, human_label=1, true_label=0),
        response("a2", ai_label=1, human_label=1, true_label=0),
        response("a3"),
        response("b1", ai_label=1, human_label=0, true_label=0),
        response("b2"),
        response("b3"),
    ]
    groups = {row.case_id: ("alpha" if row.case_id.startswith("a") else "beta") for row in rows}

    report = audit_interaction_slices(rows, groups, minimum_group_size=3)

    assert report.worst_harm_group == "alpha"
    assert report.harmful_agreement_gap == pytest.approx(2 / 3)
    assert report.human_accuracy_gap == pytest.approx(1 / 3)
    assert report.eligible_cases == 6
    assert report.excluded_cases == 0


def test_excludes_small_slices_and_reports_excluded_cases():
    rows = [response("large-1"), response("large-2"), response("small-1")]
    groups = {"large-1": "large", "large-2": "large", "small-1": "small"}

    report = audit_interaction_slices(rows, groups, minimum_group_size=2)

    assert [item.group for item in report.slices] == ["large"]
    assert report.excluded_cases == 1
    assert report.to_dict()["minimum_group_size"] == 2


def test_slice_order_is_deterministic():
    rows = [response("z1"), response("z2"), response("a1"), response("a2")]
    groups = {"z1": "zeta", "z2": "zeta", "a1": "alpha", "a2": "alpha"}
    report = audit_interaction_slices(reversed(rows), groups, minimum_group_size=2)
    assert [item.group for item in report.slices] == ["alpha", "zeta"]


@pytest.mark.parametrize(
    "rows, groups, message",
    [
        ([response("a")], {}, "missing group"),
        ([response("a"), response("a")], {"a": "group"}, "unique"),
        ([response("a"), response("b")], {"a": "", "b": "group"}, "non-empty"),
        ([response("a")], {"a": "small"}, "no group"),
    ],
)
def test_fails_closed_on_invalid_slice_contract(rows, groups, message):
    with pytest.raises(ValueError, match=message):
        audit_interaction_slices(rows, groups, minimum_group_size=2)


@pytest.mark.parametrize("minimum", [True, 1, 1.5])
def test_rejects_invalid_minimum_group_size(minimum):
    with pytest.raises((TypeError, ValueError), match="minimum_group_size"):
        audit_interaction_slices([], {}, minimum_group_size=minimum)
