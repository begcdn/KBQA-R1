from types import SimpleNamespace

import pytest

from kbqa_r1.action_constraints import HyPERActionConstraintSpec
from kbqa_r1.hyper_gold_oracle import (
    GoldContinuationOracle,
    GoldContinuationUnavailable,
)
from kbqa_r1.hyper_r1 import HypothesisNode, HypothesisStatus


def _node(
    node_id,
    functions,
    target="expression1",
    *,
    status=HypothesisStatus.ACTIVE,
    depth=0,
):
    return HypothesisNode(
        node_id=node_id,
        sample_id=0,
        function_state=tuple(functions),
        target_expression=target,
        sexpr=node_id,
        denotation=(node_id,),
        parent_id=None,
        operation="expand",
        depth=depth,
        status=status,
    )


def _constraint(*actions, selected_expression=""):
    return HyPERActionConstraintSpec(
        state_key="state",
        turn=1,
        exact_actions=tuple(actions),
        selected_expression=selected_expression,
    )


def _linear_oracle():
    return GoldContinuationOracle(
        (
            "expression1 = START('m.root')",
            "expression1 = JOIN('r.first', expression1)",
            "expression1 = JOIN('r.second', expression1)",
        ),
        "expression1",
    )


def test_selects_retained_gold_prefix_before_continuing():
    oracle = _linear_oracle()
    node = _node(
        "H0",
        ("expression1 = START('m.root')", "expression1 = JOIN('r.first', expression1)"),
    )

    choice = oracle.choose(
        nodes=(node,),
        selected_id=None,
        frontiers=(),
        constraint=_constraint("Select [ H0 ]", "Commit [ H0 ]"),
    )

    assert choice.action == "Select [ H0 ]"


def test_recalls_parked_gold_prefix():
    oracle = _linear_oracle()
    node = _node(
        "H0",
        ("expression1 = START('m.root')", "expression1 = JOIN('r.first', expression1)"),
        status=HypothesisStatus.PARKED,
    )

    choice = oracle.choose(
        nodes=(node,),
        selected_id=None,
        frontiers=(),
        constraint=_constraint("Recall [ H0 ]"),
    )

    assert choice.action == "Recall [ H0 ]"


def test_opens_then_inspects_live_gold_relation_without_inventing_it():
    oracle = _linear_oracle()
    node = _node(
        "H0",
        ("expression1 = START('m.root')", "expression1 = JOIN('r.first', expression1)"),
    )
    opened = oracle.choose(
        nodes=(node,),
        selected_id="H0",
        frontiers=(),
        constraint=_constraint(
            "Find_relation [ expression1 ]", selected_expression="expression1"
        ),
    )
    assert opened.action == "Find_relation [ expression1 ]"

    frontier = {
        "source": "expression1",
        "decision": SimpleNamespace(
            ranked_relations=(
                SimpleNamespace(relation="r.wrong"),
                SimpleNamespace(relation="r.second"),
            )
        ),
        "proposals": {
            "P4": {
                "candidate": SimpleNamespace(relation="r.wrong"),
                "status": "visible",
            },
            "P5": {
                "candidate": SimpleNamespace(relation="r.second"),
                "status": "visible",
            },
        },
        "next_offset": 2,
        "closed": False,
    }
    inspected = oracle.choose(
        nodes=(node,),
        selected_id=None,
        frontiers=(frontier,),
        constraint=_constraint("Inspect [ P4 ]", "Inspect [ P5 ]"),
    )
    assert inspected.action == "Inspect [ P5 ]"


def test_widens_only_when_gold_relation_exists_later_in_ranked_catalog():
    oracle = _linear_oracle()
    node = _node(
        "H0",
        ("expression1 = START('m.root')", "expression1 = JOIN('r.first', expression1)"),
    )
    frontier = {
        "source": "expression1",
        "decision": SimpleNamespace(
            ranked_relations=(
                SimpleNamespace(relation="r.wrong"),
                SimpleNamespace(relation="r.second"),
            )
        ),
        "proposals": {
            "P0": {
                "candidate": SimpleNamespace(relation="r.wrong"),
                "status": "visible",
            }
        },
        "next_offset": 1,
        "closed": False,
    }

    choice = oracle.choose(
        nodes=(node,),
        selected_id=None,
        frontiers=(frontier,),
        constraint=_constraint("Inspect [ P0 ]", "Widen [ expression1 ]"),
    )

    assert choice.action == "Widen [ expression1 ]"


def test_commits_complete_gold_hypothesis_despite_extra_wrong_node():
    oracle = _linear_oracle()
    gold = _node(
        "H7",
        (
            "expression1 = START('m.root')",
            "expression1 = JOIN('r.first', expression1)",
            "expression1 = JOIN('r.second', expression1)",
        ),
        depth=2,
    )
    wrong = _node(
        "H0",
        ("expression1 = START('m.root')", "expression1 = JOIN('r.wrong', expression1)"),
    )

    choice = oracle.choose(
        nodes=(wrong, gold),
        selected_id="H0",
        frontiers=(),
        constraint=_constraint("Commit [ H0 ]", "Commit [ H7 ]"),
    )

    assert choice.action == "Commit [ H7 ]"


def test_applies_gold_ontology_intersection_through_merge():
    oracle = GoldContinuationOracle(
        (
            "expression1 = START('m.root')",
            "expression1 = JOIN('r.first', expression1)",
            "expression2 = START('base.exoplanetology.exoplanet')",
            "expression1 = AND(expression2, expression1)",
        ),
        "expression1",
    )
    node = _node(
        "H0",
        ("expression1 = START('m.root')", "expression1 = JOIN('r.first', expression1)"),
    )

    choice = oracle.choose(
        nodes=(node,),
        selected_id="H0",
        frontiers=(),
        constraint=_constraint(selected_expression="expression1"),
    )

    assert choice.action == (
        "Merge [ expression1 | base.exoplanetology.exoplanet ]"
    )


def test_combines_two_retained_gold_branches():
    oracle = GoldContinuationOracle(
        (
            "expression1 = START('m.left')",
            "expression1 = JOIN('r.left', expression1)",
            "expression2 = START('m.right')",
            "expression2 = JOIN('r.right', expression2)",
            "expression3 = AND(expression1, expression2)",
        ),
        "expression3",
    )
    left = _node(
        "H2",
        ("expression1 = START('m.left')", "expression1 = JOIN('r.left', expression1)"),
    )
    right = _node(
        "H5",
        ("expression2 = START('m.right')", "expression2 = JOIN('r.right', expression2)"),
        target="expression2",
    )

    choice = oracle.choose(
        nodes=(left, right),
        selected_id=None,
        frontiers=(),
        constraint=_constraint("Combine [ H2 | H5 ]", "Combine [ H5 | H2 ]"),
    )

    assert choice.action == "Combine [ H2 | H5 ]"


def test_applies_nested_order_path_after_ontology_intersection():
    functions = (
        "expression1 = START('m.02bh_v')",
        "expression1 = JOIN('soccer.football_match.teams', expression1)",
        "expression2 = START('soccer.football_match')",
        "expression1 = AND(expression2, expression1)",
        "expression1 = ARG('ARGMIN', expression1, '(JOIN soccer.football_match.substitution soccer.football_player_substitution.minute)')",
    )
    oracle = GoldContinuationOracle(functions, "expression1")
    merged = _node("H4", functions[:-1], target="expression1", depth=2)
    action = (
        "Order [ ARGMIN | expression1 | "
        "(JOIN soccer.football_match.substitution "
        "soccer.football_player_substitution.minute) ]"
    )

    choice = oracle.choose(
        nodes=(merged,),
        selected_id="H4",
        frontiers=(),
        constraint=_constraint(action, selected_expression="expression1"),
    )

    assert choice.action == action

    completed = _node("H5", functions, target="expression1", depth=3)
    committed = oracle.choose(
        nodes=(completed,),
        selected_id="H5",
        frontiers=(),
        constraint=_constraint("Commit [ H5 ]", selected_expression="expression1"),
    )

    assert committed.action == "Commit [ H5 ]"


def test_fails_closed_when_ranker_does_not_contain_gold_relation():
    oracle = _linear_oracle()
    node = _node(
        "H0",
        ("expression1 = START('m.root')", "expression1 = JOIN('r.first', expression1)"),
    )
    frontier = {
        "source": "expression1",
        "decision": SimpleNamespace(
            ranked_relations=(SimpleNamespace(relation="r.wrong"),)
        ),
        "proposals": {},
        "next_offset": 1,
        "closed": False,
    }

    with pytest.raises(GoldContinuationUnavailable):
        oracle.choose(
            nodes=(node,),
            selected_id=None,
            frontiers=(frontier,),
            constraint=_constraint("Inspect [ P0 ]"),
        )
