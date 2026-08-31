import pytest

from kbqa_r1.action_constraints import HyPERActionConstraintSpec
from kbqa_r1.hyper_r1 import GraphActionAffordances


def _spec(**kwargs):
    values = dict(
        state_key="state-1",
        turn=3,
        affordances=GraphActionAffordances(
            select=("H0", "H2"),
            park=("H0", "H2"),
            commit=("H0",),
            combine=("H0|H2",),
            recall=("H4",),
            find_relation=("m.topic", "expression2"),
        ),
        inspect=("P1", "P7"),
        widen=("m.topic",),
        selected_expression="expression2",
    )
    values.update(kwargs)
    return HyPERActionConstraintSpec.build(**values)


@pytest.mark.parametrize(
    "action",
    (
        "Select [ H2 ]",
        "Inspect [ P7 ]",
        "Commit [ H0 ]",
        "Combine [ H0 | H2 ]",
        "Combine [ H2 | H0 ]",
        "Find_relation [ m.topic ]",
        "Widen [ m.topic ]",
        "Count [ expression2 ]",
        "Merge [ expression2 | people.person ]",
        "Order [ ARGMIN | expression2 | people.person.date_of_birth ]",
        "Order [ ARGMAX | expression2 | (JOIN (R people.person.place_of_birth) location.location.date_founded) ]",
        "Compare [ ge | people.person.height_meters | 1.8 ]",
        "Time_constraint [ people.person.date_of_birth | 1980-01-01 ]",
        "Compare [ le | time.event.start_date | 2010-07-03T16:00:00-08:00^^http://www.w3.org/2001/XMLSchema#dateTime ]",
    ),
)
def test_constraint_accepts_every_structurally_legal_action(action):
    assert _spec().accepts_response(
        f"<think>This may still be semantically wrong.</think>\n<action>{action}</action>"
    )


@pytest.mark.parametrize(
    "response",
    (
        "<action>Select [ H99 ]</action>",
        "<action>Inspect [ P0 ]</action>",
        "<action>Commit [ H2 ]</action>",
        "<action>Commit [ ]</action>",
        "<action>Recall [ H0 ]</action>",
        "<action>Combine [ H2 | H4 ]</action>",
        "<action>Select [ H0 ] trailing</action>",
        "<action>Select [ H0 ]</action><action>Commit [ H0 ]</action>",
        "<action>Count [ expression1 ]</action>",
        "<action>Merge [ expression1 | people.person ]</action>",
        "<action>Order [ ARGMIN | people.person | people.person.date_of_birth ]</action>",
        "<action>Compare [ eq | people.person.height_meters | 1.8 ]</action>",
        "<action>Time_constraint [ people.person.date_of_birth | yesterday ]</action>",
        "<action>Abstain</action>",
    ),
)
def test_constraint_rejects_impossible_or_malformed_actions(response):
    assert not _spec().accepts_response(response)


def test_thinking_cannot_swallow_an_earlier_action_boundary():
    response = (
        "<think>reasoning</think><action>Commit [ H99 ]</action>"
        "</think><action>Select [ H0 ]</action>"
    )
    assert not _spec().accepts_response(response)


def test_thinking_rejects_literal_tag_openers():
    assert not _spec().accepts_response(
        "<think>use H0 < H2</think><action>Select [ H0 ]</action>"
    )


def test_constraint_round_trip_is_digest_checked():
    spec = _spec()
    restored = HyPERActionConstraintSpec.from_dict(spec.to_dict())
    assert restored == spec
    assert restored.response_pattern == spec.response_pattern

    corrupted = spec.to_dict()
    corrupted["turn"] = 4
    with pytest.raises(ValueError, match="digest mismatch"):
        HyPERActionConstraintSpec.from_dict(corrupted)

    stale = spec.to_dict()
    stale["version"] = "hyper-action-v1"
    with pytest.raises(ValueError, match="unsupported HyPER constraint version"):
        HyPERActionConstraintSpec.from_dict(stale)


@pytest.mark.parametrize(
    "action",
    (
        "Order [ ARGMIN | people.person | (JOIN people.person.place_of_birth location.location.date_founded) ]",
        "Order [ ARGMAX | user.example.default_domain.person | (JOIN people.person.place_of_birth (JOIN location.location.people_born_here people.person.date_of_birth)) ]",
        "Compare [ ge | people.person.height_meters | 1.8 ]",
    ),
)
def test_only_independent_operators_are_available_without_a_selection(action):
    spec = _spec(selected_expression="")
    assert spec.accepts_response(f"<action>{action}</action>")


@pytest.mark.parametrize(
    "action",
    (
        "Count [ expression2 ]",
        "Merge [ expression2 | people.person ]",
        "Time_constraint [ people.person.date_of_birth | 1980-01-01 ]",
        "Order [ ARGMIN | expression2 | people.person.date_of_birth ]",
    ),
)
def test_continuation_operators_require_a_selected_hypothesis(action):
    spec = _spec(selected_expression="")
    assert not spec.accepts_response(f"<action>{action}</action>")
