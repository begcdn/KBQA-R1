import pytest
import torch
from types import SimpleNamespace

from kbqa_r1.hyper_r1 import (
    HypothesisEdgeKind,
    HypothesisGraph,
    HypothesisStatus,
    apply_grouped_decision_credit,
    combine_function_states,
    charge_execution_budget,
    enforce_commit_reward,
    dependency_function_state,
    graph_action_token_mask,
    required_hyper_relation_model,
)


def test_hyper_runtime_requires_explicit_relation_ranker():
    with pytest.raises(ValueError, match="requires hyper_r1_relation_model"):
        required_hyper_relation_model(
            SimpleNamespace(hyper_r1_enable=True, hyper_r1_relation_model=None)
        )
    assert required_hyper_relation_model(
        SimpleNamespace(hyper_r1_enable=True, hyper_r1_relation_model="/models/simcse")
    ) == "/models/simcse"
    assert required_hyper_relation_model(
        SimpleNamespace(hyper_r1_enable=False, hyper_r1_relation_model=None)
    ) is None


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


def test_same_denotation_with_different_meaning_stays_distinct():
    graph = HypothesisGraph(max_active=4)
    canonical = add(graph, "r.alias_a", ["m.answer"], group="choice")
    duplicate = add(graph, "r.alias_b", ["m.answer"], group="choice")

    assert canonical.is_active and duplicate.is_active
    assert duplicate.equivalent_to is None
    assert len(graph.active_nodes(0)) == 2
    assert not any(
        edge.kind == HypothesisEdgeKind.EQUIVALENCE
        for edge in graph.state(0).edges
    )


def test_identical_program_is_deduplicated():
    graph = HypothesisGraph(max_active=4)
    canonical = add(graph, "r.same", ["m.answer"], group="choice")
    duplicate = add(graph, "r.same", ["m.answer"], group="choice")

    assert duplicate.status == HypothesisStatus.MERGED
    assert duplicate.equivalent_to == canonical.node_id


def test_empty_hypotheses_do_not_merge():
    graph = HypothesisGraph(max_active=4)
    first = add(graph, "r.empty_a", [])
    second = add(graph, "r.empty_b", [])

    assert first.is_active and second.is_active


def test_environment_enforces_hard_active_budget():
    graph = HypothesisGraph(max_active=2)
    low = add(graph, "r.low", ["m.low"], score=0.1)
    middle = add(graph, "r.middle", ["m.middle"], score=0.5)
    with pytest.raises(RuntimeError, match="prune before exploring"):
        add(graph, "r.high", ["m.high"], score=0.9)

    assert low.status == HypothesisStatus.ACTIVE
    assert {node.node_id for node in graph.active_nodes(0)} == {low.node_id, middle.node_id}


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
    assert "committed" in graph.execution_error(
        0, opens_frontier=True, frontier_width=3
    ).lower()


def test_executable_continuation_requires_select_and_full_frontier_capacity():
    graph = HypothesisGraph(max_active=4, max_nodes=8)
    first = add(graph, "r.first", ["m.first"])
    add(graph, "r.second", ["m.second"])

    assert "Select" in graph.execution_error(
        0, opens_frontier=False, frontier_width=2
    )
    graph.select(0, first.node_id)
    assert graph.execution_error(0, opens_frontier=False, frontier_width=2) is None
    assert "full relation frontier" in graph.execution_error(
        0, opens_frontier=True, frontier_width=4
    )


def test_relation_source_is_bound_to_candidates_and_selected_state():
    graph = HypothesisGraph(max_active=4, max_nodes=8)
    assert graph.relation_source_error(0, "m.topic", ["m.topic"]) is None
    assert "supplied candidate" in graph.relation_source_error(
        0, "m.invented", ["m.topic"]
    )

    selected = add(graph, "r.first", ["m.first"])
    graph.select(0, selected.node_id)
    assert graph.relation_source_error(
        0, selected.target_expression, ["m.topic"]
    ) is None
    assert "selected hypothesis target" in graph.relation_source_error(
        0, "expression999", ["m.topic"]
    )


def test_new_candidate_root_can_open_while_other_hypotheses_stay_active():
    graph = HypothesisGraph(max_active=6, max_nodes=12)
    add(graph, "r.first", ["m.first"])

    assert graph.execution_error(
        0,
        opens_frontier=True,
        frontier_width=3,
        opens_new_root=True,
    ) is None
    assert graph.relation_source_error(
        0, "m.second_topic", ["m.first_topic", "m.second_topic"]
    ) is None
    assert "supplied candidate" in graph.relation_source_error(
        0, "m.invented", ["m.first_topic", "m.second_topic"]
    )


def test_non_frontier_execution_cannot_escape_exhausted_node_budget():
    graph = HypothesisGraph(max_active=2, max_nodes=2)
    first = add(graph, "r.first", ["m.first"])
    add(graph, "r.second", ["m.second"])
    graph.select(0, first.node_id)

    assert "node budget" in graph.execution_error(
        0, opens_frontier=False, frontier_width=1
    ).lower()


def test_combine_node_retains_both_parent_edges():
    graph = HypothesisGraph(max_active=4)
    left = add(graph, "r.left", ["m.shared"])
    right = add(graph, "r.right", ["m.shared"])
    graph.mark_expanded(0, left.node_id)
    graph.mark_expanded(0, right.node_id)
    combined = graph.add_executed(
        sample_id=0,
        function_state=("expression3 = AND(expression1, expression2)",),
        target_expression="expression3",
        sexpr="(AND left right)",
        denotation=["m.shared"],
        parent_id=left.node_id,
        parent_ids=(left.node_id, right.node_id),
        operation="combine",
    )

    incoming = {
        edge.source
        for edge in graph.state(0).edges
        if edge.target == combined.node_id
    }
    assert incoming == {left.node_id, right.node_id}
    assert f"parents={left.node_id}+{right.node_id} operation=combine" in graph.serialize(0)


def test_combine_validation_is_atomic():
    graph = HypothesisGraph(max_active=3)
    left = add(graph, "r.left", ["m.left"])

    with pytest.raises(ValueError, match="distinct"):
        graph.combination_parents(0, left.node_id, left.node_id)

    assert left.is_active
    assert graph.state(0).execution_calls == 1


def test_serialization_exposes_compact_executable_state():
    graph = HypothesisGraph(max_active=3)
    node = add(graph, "people.person.place_of_birth", ["m.city"])

    text = graph.serialize(0)
    assert node.node_id in text
    assert "people.person.place_of_birth" in text
    assert "m.city" in text
    assert "Actions: Select, Find_relation [ source ], Combine, Prune, or Commit" in text


def test_expanding_parent_frees_one_frontier_slot_but_keeps_history():
    graph = HypothesisGraph(max_active=3)
    parent = add(graph, "r.parent", ["m.parent"])
    sibling = add(graph, "r.sibling", ["m.sibling"])

    graph.mark_expanded(0, parent.node_id)
    child = add(graph, "r.child", ["m.child"], parent=parent.node_id)

    assert parent.status == HypothesisStatus.EXPANDED
    assert {node.node_id for node in graph.active_nodes(0)} == {sibling.node_id, child.node_id}


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


def test_dependency_state_drops_unrelated_retained_root():
    state = [
        "expression1 = START('m.left')",
        "expression1 = JOIN('r.left', expression1)",
        "expression2 = START('m.right')",
        "expression2 = JOIN('r.right', expression2)",
    ]

    assert dependency_function_state(state, "expression2") == (
        "expression2 = START('m.right')",
        "expression2 = JOIN('r.right', expression2)",
    )


def test_graph_credit_compares_different_actions_from_same_state():
    advantages = torch.zeros((3, 4))
    action_ids = torch.tensor(
        [[0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0]]
    )
    rewards = torch.tensor([1.0, 0.0, 0.7])
    records = [
        [{"state_key": "S", "action_key": "commit-H1"}],
        [{"state_key": "S", "action_key": "commit-H2"}],
        [{"state_key": "unique", "action_key": "commit-H3"}],
    ]

    result, compared = apply_grouped_decision_credit(
        advantages,
        action_ids,
        rewards,
        ["question", "question", "question"],
        records,
        weight=0.5,
    )

    assert result[0, 1:3].tolist() == pytest.approx([0.5, 0.5])
    assert result[1, 1:3].tolist() == pytest.approx([-0.5, -0.5])
    assert result[2].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert compared[2].sum().item() == 0


def test_decision_state_distinguishes_remaining_turn_budget():
    graph = HypothesisGraph(max_active=3)
    add(graph, "r.answer", ["m.answer"])

    assert graph.decision_state_key(0, turn=1) != graph.decision_state_key(0, turn=4)


def test_invalid_commit_cannot_keep_answer_reward():
    rewards = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 1]])
    result = enforce_commit_reward(
        rewards, mask, torch.tensor([True, False]), invalid_penalty=0.25
    )

    assert result[0].tolist() == [0.0, 0.0, 1.0]
    assert result[1].tolist() == [0.0, 0.0, -0.25]


def test_final_answer_must_equal_committed_denotation():
    graph = HypothesisGraph(max_active=3)
    answer = add(graph, "r.answer", ["m.one", "m.two"])
    graph.commit(0, answer.node_id)

    assert graph.answer_matches_commit(0, "<answer>m.two m.one</answer>")
    assert not graph.answer_matches_commit(0, "<answer>m.one</answer>")


def test_execution_budget_only_charges_excess_over_sibling_rollout():
    rewards = torch.zeros((2, 4))
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    counts = torch.tensor([6.0, 3.0])
    result = charge_execution_budget(
        rewards,
        mask,
        counts,
        max_nodes=6,
        cost=0.12,
        group_ids=["question", "question"],
    )
    assert result[0, 2].item() == pytest.approx(-0.06)
    assert result[1, 1].item() == 0.0
    assert torch.count_nonzero(result).item() == 1
