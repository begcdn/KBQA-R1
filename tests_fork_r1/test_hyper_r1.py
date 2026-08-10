import pytest
import torch

from kbqa_r1.hyper_r1 import (
    HypothesisEdgeKind,
    HypothesisGraph,
    HypothesisStatus,
    combine_function_states,
    concentrate_graph_credit,
    charge_execution_budget,
    graph_action_token_mask,
)


def add(graph, relation, answers, parent=None, score=0.5, group=None):
    return graph.add_executed(
        sample_id=0,
        function_state=(f"expression1 = JOIN('{relation}', expression1)",),
        target_expression="expression1",
        sexpr=f"(JOIN {relation} m.topic)",
        denotation=answers,
        parent_id=parent,
        operation="expand",
        relation_id=relation,
        relation_prompt="relation clue",
        resolver_score=score,
        contrast_group=group,
    )


def test_contrast_siblings_remain_active():
    graph = HypothesisGraph(max_active=4)
    left = add(graph, "people.person.parents", ["m.mother"], group="turn0:action0")
    right = add(graph, "people.person.children", ["m.child"], group="turn0:action0")

    assert [node.node_id for node in graph.active_nodes(0)] == [left.node_id, right.node_id]
    assert any(edge.kind == HypothesisEdgeKind.CONTRAST for edge in graph.state(0).edges)


def test_same_denotation_merges_without_erasing_provenance():
    graph = HypothesisGraph(max_active=4)
    canonical = add(graph, "r.alias_a", ["m.answer"], group="choice")
    duplicate = add(graph, "r.alias_b", ["m.answer"], group="choice")

    assert duplicate.status == HypothesisStatus.MERGED
    assert duplicate.equivalent_to == canonical.node_id
    assert len(graph.active_nodes(0)) == 1
    assert any(edge.kind == HypothesisEdgeKind.EQUIVALENCE for edge in graph.state(0).edges)


def test_empty_hypotheses_do_not_merge():
    graph = HypothesisGraph(max_active=4)
    first = add(graph, "r.empty_a", [])
    second = add(graph, "r.empty_b", [])

    assert first.is_active and second.is_active


def test_environment_enforces_hard_active_budget():
    graph = HypothesisGraph(max_active=2)
    low = add(graph, "r.low", ["m.low"], score=0.1)
    middle = add(graph, "r.middle", ["m.middle"], score=0.5)
    high = add(graph, "r.high", ["m.high"], score=0.9)

    assert low.status == HypothesisStatus.PRUNED
    assert {node.node_id for node in graph.active_nodes(0)} == {middle.node_id, high.node_id}


def test_commit_requires_nonempty_active_hypothesis():
    graph = HypothesisGraph(max_active=3)
    empty = add(graph, "r.empty", [])
    answer = add(graph, "r.answer", ["m.answer"])

    with pytest.raises(ValueError):
        graph.commit(0, empty.node_id)

    committed = graph.commit(0, answer.node_id)
    assert committed.status == HypothesisStatus.COMMITTED
    assert graph.state(0).committed_id == answer.node_id
    assert empty.status == HypothesisStatus.PRUNED


def test_serialization_exposes_compact_executable_state():
    graph = HypothesisGraph(max_active=3)
    node = add(graph, "people.person.place_of_birth", ["m.city"])

    text = graph.serialize(0)
    assert node.node_id in text
    assert "people.person.place_of_birth" in text
    assert "m.city" in text
    assert "Actions: Explore/Expand, Combine, Prune, or Commit" in text


def test_graph_action_mask_marks_all_occurrences():
    assert graph_action_token_mask([1, 2, 3, 2, 3], [2, 3]) == [0, 1, 1, 1, 1]


def test_combine_function_states_preserves_both_fork_branches():
    prefix = ["expression1 = START('m.topic')"]
    left = prefix + ["expression1 = JOIN('r.left', expression1)"]
    right = prefix + ["expression1 = JOIN('r.right', expression1)"]

    state, target = combine_function_states(left, "expression1", right, "expression1")

    assert state == [
        "expression1 = START('m.topic')",
        "expression1 = JOIN('r.left', expression1)",
        "expression2 = JOIN('r.right', expression1)",
        "expression3 = AND(expression1, expression2)",
    ]
    assert target == "expression3"


def test_graph_credit_is_concentrated_on_committed_actions():
    advantages = torch.tensor([[2.0, 2.0, 2.0]])
    mask = torch.tensor([[0.0, 1.0, 0.0]])
    result = concentrate_graph_credit(advantages, mask, weight=0.5)
    assert result.tolist() == [[2.0, 3.0, 2.0]]


def test_execution_budget_is_charged_once_on_last_response_token():
    rewards = torch.zeros((2, 4))
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    counts = torch.tensor([6.0, 3.0])
    result = charge_execution_budget(rewards, mask, counts, max_nodes=6, cost=0.12)
    assert result[0, 2].item() == pytest.approx(-0.12)
    assert result[1, 1].item() == pytest.approx(-0.06)
    assert torch.count_nonzero(result).item() == 2
