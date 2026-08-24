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
    penalize_invalid_actions,
    proposal_action_targets,
    public_frontier_signature,
    public_question_contract,
    render_hyper_observation_suffix,
    result_display_labels,
    result_denotation_values,
    required_hyper_relation_model,
)


class _TinyChatTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        return rendered + ("<assistant>" if add_generation_prompt else "")


def test_runtime_observation_suffix_matches_decision_sft_context():
    tokenizer = _TinyChatTokenizer()
    initial = tokenizer.apply_chat_template(
        [{"role": "user", "content": "question"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    runtime = initial + render_hyper_observation_suffix(tokenizer, "state")
    sft = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "state"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    assert runtime == sft


def test_execution_results_keep_identity_separate_from_display_label():
    rows = [{"x": "m.answer", "name": "Readable Answer"}]

    assert result_denotation_values(rows, ["x:m.answer (Readable Answer)"]) == (
        "m.answer",
    )


def test_runtime_display_labels_match_builder_fallback_for_schema_values():
    assert result_display_labels([{"x": "people.person.author"}]) == {
        "people.person.author": "author"
    }


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


def test_frontier_does_not_expose_teacher_only_provenance_labels():
    graph = HypothesisGraph(max_active=4)
    policy = add(graph, "r.policy", ["m.policy"])
    alternative = add(graph, "r.alternative", ["m.alternative"])
    policy.provenance.append("policy_choice")
    alternative.provenance.append("ranked_alternative")

    rendered = graph.serialize(0)

    assert "source=policy" not in rendered
    assert "source=alternative" not in rendered
    assert "source=derived" not in rendered


def test_recall_restores_runtime_creation_order_in_every_affordance():
    graph = HypothesisGraph(max_active=4)
    first = add(graph, "r.first", ["m.first"])
    second = add(graph, "r.second", ["m.second"])
    third = add(graph, "r.third", ["m.third"])
    graph.park(0, first.node_id)
    graph.recall(0, first.node_id)

    rendered = graph.serialize(0)

    assert "Select=[H0,H1,H2]" in rendered
    assert "Park=[H0,H1,H2]" in rendered
    assert "Commit(nonempty active)=[H0,H1,H2]" in rendered
    assert "Combine=[H0|H1,H0|H2,H1|H2]" in rendered


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


def test_public_question_contract_recognizes_count_without_phone_number_false_positive():
    assert public_question_contract(
        "How many people satisfy the condition?"
    ).count_required is True
    assert public_question_contract(
        "What is the number of people satisfying the condition?"
    ).count_required is True
    assert public_question_contract(
        "What is the phone number of the hotel?"
    ).count_required is False


def test_public_question_contract_recognizes_grailqa_count_paraphrases_and_typos():
    questions = (
        "What number of planets are orbiting in heliocentric orbits?",
        "What is the amount of camera uncompressed formats for Fujifilm?",
        "What is the quantity of film characters that are foxes?",
        "What is the count of building complexes with this function?",
        "How may composers were there in Steal Away?",
        "Find many Olympic disciplines are there in the games.",
        "What numer of bicycle models are the same type as this bike?",
    )

    assert all(public_question_contract(question).count_required for question in questions)


def test_empty_count_branch_is_preserved_and_can_reach_zero_commit():
    graph = HypothesisGraph(max_active=3, max_nodes=6)
    graph.register_public_question(0, "How many people satisfy condition X?")
    empty = add(graph, "r.people", [])

    with pytest.raises(ValueError, match="Count can produce"):
        graph.prune(0, empty.node_id)

    graph.select(0, empty.node_id)
    assert graph.execution_error(
        0,
        opens_frontier=False,
        frontier_width=6,
        operation="Count",
    ) is None
    assert "only be continued" in graph.execution_error(
        0,
        opens_frontier=True,
        frontier_width=6,
        operation="Find_relation",
    )

    graph.mark_expanded(0, empty.node_id)
    counted = graph.add_executed(
        sample_id=0,
        function_state=(
            "expression1 = JOIN('r.people', expression1)",
            "expression1 = COUNT(expression1)",
        ),
        target_expression="expression1",
        sexpr="(COUNT (JOIN r.people m.topic))",
        denotation=("0",),
        parent_id=empty.node_id,
        operation="count",
    )

    assert graph.commit(0, counted.node_id).denotation == ("0",)


def test_noncount_empty_prune_requires_and_records_public_certificate():
    graph = HypothesisGraph(max_active=3)
    graph.register_public_question(0, "Which people satisfy condition X?")
    empty = add(graph, "r.people", [])

    certificate = graph.prune(0, empty.node_id)

    assert certificate.kind == "empty_monotone"
    assert certificate.empty_preserving_completion is True
    assert graph.state(0).prune_certificates[empty.node_id] == certificate
    assert empty.semantic_status.value == "proved_false"


def test_missing_public_question_fails_closed_for_empty_prune():
    graph = HypothesisGraph(max_active=3)
    empty = add(graph, "r.people", [])

    with pytest.raises(ValueError, match="Count can produce"):
        graph.prune(0, empty.node_id)


def test_count_is_rejected_when_public_question_does_not_require_it():
    graph = HypothesisGraph(max_active=3)
    graph.register_public_question(0, "Which people satisfy condition X?")
    answer = add(graph, "r.people", ["m.person"])
    graph.select(0, answer.node_id)

    assert "not licensed" in graph.execution_error(
        0,
        opens_frontier=False,
        frontier_width=6,
        operation="Count",
    )


def test_environment_enforces_hard_active_budget():
    graph = HypothesisGraph(max_active=2)
    low = add(graph, "r.low", ["m.low"], score=0.1)
    middle = add(graph, "r.middle", ["m.middle"], score=0.5)
    with pytest.raises(RuntimeError, match="park a viable hypothesis"):
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
    assert empty.status == HypothesisStatus.CLOSED
    assert "committed" in graph.execution_error(
        0, opens_frontier=True, frontier_width=3
    ).lower()


def test_symbolic_relation_page_does_not_consume_hypothesis_capacity():
    graph = HypothesisGraph(max_active=4, max_nodes=8)
    first = add(graph, "r.first", ["m.first"])
    add(graph, "r.second", ["m.second"])

    assert "Select" in graph.execution_error(
        0, opens_frontier=False, frontier_width=2
    )
    graph.select(0, first.node_id)
    assert graph.execution_error(0, opens_frontier=False, frontier_width=2) is None
    assert graph.execution_error(0, opens_frontier=True, frontier_width=40) is None
    assert len(graph.state(0).nodes) == 2


def test_park_and_recall_change_storage_visibility_not_semantic_truth():
    graph = HypothesisGraph(max_active=2)
    first = add(graph, "r.first", ["m.first"])
    second = add(graph, "r.second", ["m.second"])

    graph.park(0, first.node_id)
    assert first.status == HypothesisStatus.PARKED
    assert first.semantic_status.value == "unresolved"
    assert graph.available_active_slots(0) == 1
    graph.recall(0, first.node_id)
    assert first.status == HypothesisStatus.ACTIVE
    assert {node.node_id for node in graph.active_nodes(0)} == {
        first.node_id,
        second.node_id,
    }


def test_execution_attempt_budget_is_separate_from_persistent_store():
    graph = HypothesisGraph(
        max_active=2, max_nodes=10, max_execution_attempts=2
    )
    assert graph.record_execution_attempt(0) == 1
    assert graph.record_execution_attempt(0) == 2
    with pytest.raises(RuntimeError, match="execution budget exhausted"):
        graph.record_execution_attempt(0)
    assert graph.has_capacity(0, 5)


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


def test_public_literal_can_open_an_independent_relation_root():
    graph = HypothesisGraph(max_active=4, max_nodes=8)
    add(graph, "r.first", ["m.first"])
    literal = "1.5^^http://www.w3.org/2001/XMLSchema#float"

    assert graph.relation_source_error(0, literal, ["m.topic", literal]) is None
    assert graph.execution_error(
        0,
        opens_frontier=True,
        frontier_width=2,
        opens_new_root=True,
    ) is None


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


def test_independent_logical_constraint_can_open_as_a_new_root():
    graph = HypothesisGraph(max_active=6, max_nodes=12)

    assert graph.execution_error(
        0,
        opens_frontier=False,
        frontier_width=3,
        opens_new_root=True,
    ) is None
    add(graph, "r.first", ["m.first"])
    assert graph.execution_error(
        0,
        opens_frontier=False,
        frontier_width=3,
        opens_new_root=True,
    ) is None


def test_non_frontier_execution_cannot_escape_exhausted_node_budget():
    graph = HypothesisGraph(max_active=2, max_nodes=2)
    first = add(graph, "r.first", ["m.first"])
    add(graph, "r.second", ["m.second"])
    graph.select(0, first.node_id)

    assert "persistent hypothesis store" in graph.execution_error(
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


def test_combine_depth_follows_the_deepest_parent():
    graph = HypothesisGraph(max_active=6)
    shallow = add(graph, "r.shallow", ["m.shared"])
    deep_root = add(graph, "r.deep_root", ["m.middle"])
    deep = graph.add_executed(
        sample_id=0,
        function_state=("expression2 = JOIN('r.deep', expression2)",),
        target_expression="expression2",
        sexpr="(JOIN r.deep m.root)",
        denotation=["m.shared"],
        parent_id=deep_root.node_id,
        operation="expand",
    )
    combined = graph.add_executed(
        sample_id=0,
        function_state=("expression3 = AND(expression1, expression2)",),
        target_expression="expression3",
        sexpr="(AND shallow deep)",
        denotation=["m.shared"],
        parent_id=shallow.node_id,
        parent_ids=(shallow.node_id, deep.node_id),
        operation="combine",
    )

    assert shallow.depth == 0
    assert deep.depth == 1
    assert combined.depth == 2


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
    assert f"Select=[{node.node_id}]" in text
    assert f"Park=[{node.node_id}]" in text
    assert f"Commit(nonempty active)=[{node.node_id}]" in text
    assert "Actions: Select, Find_relation [ source ], Widen [ source ]" in text


def test_selected_expression_is_the_only_advertised_continuation_source():
    graph = HypothesisGraph(max_active=3)
    node = add(graph, "r.answer", ["m.answer"])
    graph.select(0, node.node_id)

    assert "Find_relation sources=[expression1]" in graph.serialize(0)


def test_unselected_graph_exposes_only_supplied_entity_roots():
    graph = HypothesisGraph(max_active=3)
    node = add(graph, "r.answer", ["m.answer"])

    observation = graph.serialize(
        0,
        candidate_sources=(
            "m.topic",
            "m.other_topic",
            "1.5^^http://www.w3.org/2001/XMLSchema#float",
        ),
    )
    assert (
        "Find_relation sources=[m.topic,m.other_topic,"
        "1.5^^http://www.w3.org/2001/XMLSchema#float]"
    ) in observation

    graph.select(0, node.node_id)
    selected = graph.serialize(
        0, candidate_sources=("m.topic", "m.other_topic")
    )
    assert "Find_relation sources=[expression1]" in selected
    assert "Find_relation sources=[m.topic" not in selected


def test_action_affordances_match_select_prune_and_recall_legality():
    graph = HypothesisGraph(max_active=2, max_nodes=4)
    graph.register_public_question(0, "Which people satisfy the condition?")
    empty = add(graph, "r.empty", [])
    parked = add(graph, "r.parked", ["m.parked"])
    graph.park(0, parked.node_id)

    line = next(
        value
        for value in graph.serialize(0).splitlines()
        if value.startswith("Available targets:")
    )
    assert "Select=[none]" in line
    assert f"Park=[{empty.node_id}]" in line
    assert f"Prune candidates=[{empty.node_id}]" in line
    assert f"Recall=[{parked.node_id}]" in line

    active = add(graph, "r.active", ["m.active"])
    full_line = next(
        value
        for value in graph.serialize(0).splitlines()
        if value.startswith("Available targets:")
    )
    assert "Recall=[none]" in full_line
    with pytest.raises(RuntimeError, match="workspace is full"):
        graph.recall(0, parked.node_id)
    assert active.node_id in full_line


def test_proposal_affordances_respect_selection_and_resource_budgets():
    inspect, widen = proposal_action_targets(
        ("P0", "P1"),
        "expression1",
        exposed=2,
        total=4,
        selected_id=None,
        active_count=1,
        max_active=2,
        node_count=2,
        max_nodes=4,
        execution_attempts=1,
        execution_budget=3,
    )
    assert inspect == ("P0", "P1")
    assert widen == "expression1"

    inspect, widen = proposal_action_targets(
        ("P0",),
        "expression1",
        exposed=1,
        total=4,
        selected_id="H0",
        active_count=2,
        max_active=2,
        node_count=4,
        max_nodes=4,
        execution_attempts=3,
        execution_budget=3,
    )
    assert inspect == ()
    assert widen is None


def test_serialization_pairs_readable_labels_with_stable_entity_ids():
    graph = HypothesisGraph(max_active=3)
    node = graph.add_executed(
        sample_id=0,
        function_state=("expression1 = JOIN('r.answer', expression1)",),
        target_expression="expression1",
        sexpr="(JOIN r.answer m.topic)",
        denotation=["m.answer"],
        denotation_labels={"m.answer": "Readable Answer"},
        parent_id=None,
        operation="expand",
        relation_id="r.answer",
    )

    text = graph.serialize(0)
    assert node.denotation == ("m.answer",)
    assert "Readable Answer [m.answer]" in text


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


def test_forced_terminal_rollouts_do_not_supply_or_receive_local_credit():
    advantages = torch.zeros((3, 3))
    action_ids = torch.tensor([[1, 1, 0], [1, 1, 0], [1, 1, 0]])
    rewards = torch.tensor([1.0, 0.9, 0.0])
    records = [
        [{"state_key": "S", "action_key": "commit-H0"}],
        [{"state_key": "S", "action_key": "wait"}],
        [{"state_key": "S", "action_key": "commit-H1"}],
    ]

    result, compared = apply_grouped_decision_credit(
        advantages,
        action_ids,
        rewards,
        ["question"] * 3,
        records,
        eligible_rollouts=torch.tensor([True, False, True]),
    )

    assert result[0, :2].tolist() == pytest.approx([1.0, 1.0])
    assert result[1].tolist() == [0.0, 0.0, 0.0]
    assert result[2, :2].tolist() == pytest.approx([-1.0, -1.0])
    assert compared[1].sum().item() == 0


def test_decision_state_distinguishes_remaining_turn_budget():
    graph = HypothesisGraph(max_active=3)
    add(graph, "r.answer", ["m.answer"])

    assert graph.decision_state_key(0, turn=1) != graph.decision_state_key(0, turn=4)


def test_decision_state_distinguishes_hidden_widen_choices():
    graph = HypothesisGraph(max_active=3)
    add(graph, "r.answer", ["m.answer"])

    assert graph.decision_state_key(
        0, turn=1, legal_context=(("m.topic", 2, 4, ("r.a", "r.b")),)
    ) != graph.decision_state_key(
        0, turn=1, legal_context=(("m.topic", 2, 4, ("r.a", "r.c")),)
    )


def test_public_frontier_signature_keeps_fully_exposed_catalogs_distinct():
    candidate = SimpleNamespace(relation="r.visible", score=0.9)
    frontier = {
        "closed": False,
        "source": "m.first",
        "next_offset": 1,
        "decision": SimpleNamespace(ranked_relations=[candidate]),
        "proposals": {
            "P0": {
                "candidate": candidate,
                "rank": 1,
                "status": "visible",
            }
        },
    }
    first = public_frontier_signature([frontier], 6)
    frontier["source"] = "m.second"
    second = public_frontier_signature([frontier], 6)
    frontier["source"] = "m.first"
    frontier["proposals"]["P0"]["status"] = "failed"
    failed = public_frontier_signature([frontier], 6)

    assert first
    assert first != second
    assert first != failed


def test_combine_action_key_is_parent_order_invariant():
    graph = HypothesisGraph(max_active=3)
    left = add(graph, "r.left", ["m.left"])
    right = add(graph, "r.right", ["m.right"])

    assert graph.action_key(0, "Combine", [left.node_id, right.node_id]) == graph.action_key(
        0, "Combine", [right.node_id, left.node_id]
    )


def test_execution_attempts_include_failures_without_inventing_nodes():
    graph = HypothesisGraph(max_active=3)

    graph.record_execution_attempt(0)

    assert graph.state(0).execution_attempts == 1
    assert graph.state(0).execution_calls == 0


def test_hypothesis_graph_supports_three_hop_continuation():
    graph = HypothesisGraph(max_active=3)
    first = add(graph, "r.one", ["m.one"])
    graph.mark_expanded(0, first.node_id)
    second = add(graph, "r.two", ["m.two"], parent=first.node_id)
    graph.mark_expanded(0, second.node_id)
    third = add(graph, "r.three", ["m.answer"], parent=second.node_id)

    # Root executions have graph depth zero, so a three-execution lineage ends
    # at depth two.
    assert third.depth == 2
    assert graph.lineage(0, third.node_id) == [
        third.node_id,
        second.node_id,
        first.node_id,
    ]


def test_illegal_commit_cannot_keep_answer_reward():
    rewards = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 1]])
    result = enforce_commit_reward(
        rewards,
        mask,
        torch.tensor([True, False]),
        torch.tensor([1.0, 1.0]),
        invalid_penalty=0.25,
    )

    assert result[0].tolist() == [0.0, 0.0, 1.0]
    assert result[1].tolist() == [0.0, 0.0, -0.25]


def test_answer_correct_alternative_is_not_vetoed_by_intent_certificate():
    rewards = torch.full((2, 3), 0.9)
    mask = torch.ones_like(rewards)

    result = enforce_commit_reward(
        rewards,
        mask,
        torch.tensor([True, True]),
        torch.tensor([1.0, 0.5]),
        commit_intent_equivalent=torch.tensor([False, True]),
        semantic_bonus=0.1,
    )

    assert result[0].tolist() == [0.0, 0.0, 1.0]
    assert result[1].tolist() == pytest.approx([0.0, 0.0, 0.5])


def test_semantic_certificate_is_a_metric_not_an_f1_bonus():
    rewards = torch.zeros((2, 2))
    mask = torch.ones_like(rewards)

    result = enforce_commit_reward(
        rewards,
        mask,
        torch.tensor([True, True]),
        torch.tensor([0.0, 0.0]),
        commit_intent_equivalent=torch.tensor([False, True]),
        semantic_bonus=0.1,
    )

    assert result[0].tolist() == [0.0, 0.0]
    assert result[1].tolist() == [0.0, 0.0]


def test_abstention_is_penalized_under_f1_objective():
    rewards = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    mask = torch.ones_like(rewards)
    result = enforce_commit_reward(
        rewards,
        mask,
        torch.tensor([False, False]),
        torch.tensor([0.0, 0.0]),
        abstained=torch.tensor([True, False]),
        invalid_penalty=0.25,
    )

    assert result[0].tolist() == [0.0, 0.0, -0.25]
    assert result[1].tolist() == [0.0, 0.0, -0.25]


def test_runtime_rejects_abstention_under_f1_objective():
    graph = HypothesisGraph(max_active=3)
    add(graph, "r.candidate", ["m.answer"])

    with pytest.raises(ValueError, match="disabled for F1 evaluation"):
        graph.abstain(0)


def test_visible_clock_is_part_of_the_shared_graph_state():
    graph = HypothesisGraph(max_active=3)
    graph.set_clock(0, turns_used=7, max_turns=32)

    rendered = graph.serialize(0)

    assert "turns_used=7/32" in rendered
    assert "turns_remaining=25" in rendered


def test_forced_terminal_prefers_the_policy_selected_nonempty_candidate():
    graph = HypothesisGraph(max_active=3)
    preferred = add(graph, "r.preferred", ["m.partial"])
    add(graph, "r.newer", ["m.other"])
    graph.select(0, preferred.node_id)
    graph.mark_expanded(0, preferred.node_id)

    chosen = graph.force_terminal(0)

    assert chosen is preferred
    assert graph.state(0).committed_id == preferred.node_id
    assert graph.state(0).terminal_kind == "forced_candidate"
    assert graph.is_terminal(0)


def test_forced_terminal_without_nonempty_candidate_returns_empty_prediction():
    graph = HypothesisGraph(max_active=3)
    add(graph, "r.empty", [])

    assert graph.force_terminal(0) is None
    assert graph.state(0).committed_id is None
    assert graph.state(0).terminal_kind == "forced_empty"
    assert graph.is_terminal(0)


def test_forced_terminal_does_not_choose_an_unselected_historical_node():
    graph = HypothesisGraph(max_active=3)
    historical = add(graph, "r.old", ["m.old"])
    graph.mark_expanded(0, historical.node_id)

    assert graph.force_terminal(0) is None
    assert graph.state(0).terminal_kind == "forced_empty"


def test_repeated_select_is_rejected_as_no_progress():
    graph = HypothesisGraph(max_active=3)
    node = add(graph, "r.choice", ["m.answer"])
    graph.select(0, node.node_id)

    with pytest.raises(ValueError, match="already selected"):
        graph.select(0, node.node_id)


def test_invalid_action_tokens_cannot_keep_positive_advantage():
    advantages = torch.tensor([[0.8, -0.7, 0.1]])
    invalid = torch.tensor([[True, True, False]])

    result = penalize_invalid_actions(advantages, invalid, penalty=0.25)

    assert torch.allclose(result, torch.tensor([[-0.25, -0.7, 0.1]]))


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
        max_execution_attempts=6,
        cost=0.12,
        group_ids=["question", "question"],
    )
    assert result[0, 2].item() == pytest.approx(-0.06)
    assert result[1, 1].item() == 0.0
    assert torch.count_nonzero(result).item() == 1


def test_execution_cost_cannot_reverse_answer_f1_ordering():
    rewards = torch.tensor([[0.0, 0.70], [0.0, 0.69]])
    mask = torch.ones_like(rewards)

    result = charge_execution_budget(
        rewards,
        mask,
        torch.tensor([24.0, 1.0]),
        max_execution_attempts=24,
        cost=1.0,
        group_ids=["question", "question"],
    )

    totals = result.sum(dim=-1)
    assert totals[0] > totals[1]
