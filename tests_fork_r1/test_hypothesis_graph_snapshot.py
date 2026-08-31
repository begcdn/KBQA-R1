from copy import deepcopy
import json

import pytest

from kbqa_r1.hyper_r1 import (
    HYPOTHESIS_GRAPH_SNAPSHOT_VERSION,
    HypothesisGraph,
    HypothesisSemanticStatus,
    HypothesisStatus,
)


def _rich_graph(sample_id: int = 3) -> HypothesisGraph:
    graph = HypothesisGraph(
        max_active=6,
        max_nodes=12,
        max_execution_attempts=9,
    )
    graph.register_public_question(
        sample_id, "Which locations are associated with this entity?"
    )
    graph.set_clock(sample_id, turns_used=7, max_turns=32)

    empty = graph.add_executed(
        sample_id=sample_id,
        function_state=("expression1 = START('m.empty')",),
        target_expression="expression1",
        sexpr="m.empty",
        denotation=(),
        parent_id=None,
        operation="root",
        provenance=("empty_probe",),
    )
    graph.prune(sample_id, empty.node_id, reason="public_empty")

    root = graph.add_executed(
        sample_id=sample_id,
        function_state=("expression1 = START('m.root')",),
        target_expression="expression1",
        sexpr="m.root",
        denotation=("m.answer",),
        denotation_labels={"m.answer": "Answer"},
        parent_id=None,
        operation="root",
        relation_prompt="initial entity",
        resolver_score=0.75,
        provenance=("root",),
    )
    child = graph.add_executed(
        sample_id=sample_id,
        function_state=(
            "expression1 = START('m.root')",
            "expression1 = JOIN('location.location.contains', expression1)",
        ),
        target_expression="expression1",
        sexpr="(JOIN location.location.contains m.root)",
        denotation=("m.child",),
        denotation_labels={"m.child": "Child"},
        parent_id=root.node_id,
        operation="expand",
        relation_id="location.location.contains",
        relation_prompt="location relation",
        resolver_score=0.625,
        contrast_group="catalog-0",
        provenance=("inspected_proposal:P0",),
    )
    root.semantic_status = HypothesisSemanticStatus.VIABLE
    child.semantic_status = HypothesisSemanticStatus.PROVED_COMPLETE
    graph.record_execution_attempt(sample_id)
    graph.record_execution_attempt(sample_id)
    graph.select(sample_id, child.node_id)
    return graph


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def test_snapshot_round_trip_is_json_safe_complete_and_deterministic():
    sample_id = 3
    graph = _rich_graph(sample_id)

    snapshot = graph.snapshot_sample_state(sample_id)
    encoded = _canonical(snapshot)
    decoded = json.loads(encoded)

    assert snapshot["schema_version"] == HYPOTHESIS_GRAPH_SNAPSHOT_VERSION
    assert snapshot["capacity"] == {
        "max_active": 6,
        "max_nodes": 12,
        "max_execution_attempts": 9,
    }
    assert snapshot["state"]["selected_id"] == "H2"
    assert snapshot["state"]["execution_attempts"] == 2
    assert snapshot["state"]["execution_calls"] == 3
    assert snapshot["state"]["next_node_index"] == 3
    assert snapshot["state"]["turns_used"] == 7
    assert snapshot["state"]["max_turns"] == 32
    assert snapshot["state"]["last_preferred_nonempty_id"] == "H2"
    assert snapshot["state"]["question_contract"]["count_required"] is False
    assert snapshot["state"]["prune_certificates"]["H0"]["node_id"] == "H0"
    assert {edge["kind"] for edge in snapshot["state"]["edges"]} == {
        "expansion"
    }

    graph.clear(sample_id)
    graph.restore_sample_state(sample_id, decoded)

    assert _canonical(graph.snapshot_sample_state(sample_id)) == encoded
    restored = graph.state(sample_id)
    assert restored.nodes["H0"].status == HypothesisStatus.PRUNED
    assert restored.nodes["H0"].semantic_status == HypothesisSemanticStatus.PROVED_FALSE
    assert restored.nodes["H2"].semantic_status == HypothesisSemanticStatus.PROVED_COMPLETE
    assert restored.question_contract.question.startswith("Which locations")


def test_restore_can_retarget_an_independent_counterfactual_sample():
    source_id = 3
    target_id = 11
    graph = _rich_graph(source_id)
    source_snapshot = graph.snapshot_sample_state(source_id)

    graph.restore_sample_state(target_id, source_snapshot)

    target = graph.state(target_id)
    assert target.sample_id == target_id
    assert all(node.sample_id == target_id for node in target.nodes.values())
    assert target.selected_id == graph.state(source_id).selected_id == "H2"

    graph.park(target_id, "H2")
    assert graph.state(target_id).nodes["H2"].status == HypothesisStatus.PARKED
    assert graph.state(source_id).nodes["H2"].status == HypothesisStatus.ACTIVE


def test_snapshot_preserves_commit_terminal_and_abstain_fields():
    graph = HypothesisGraph(max_active=3, max_nodes=5, max_execution_attempts=4)
    node = graph.add_executed(
        sample_id=0,
        function_state=("expression1 = START('m.root')",),
        target_expression="expression1",
        sexpr="m.root",
        denotation=("m.answer",),
        parent_id=None,
        operation="root",
    )
    graph.select(0, node.node_id)
    graph.commit(0, node.node_id)
    snapshot = graph.snapshot_sample_state(0)
    graph.clear(0)
    graph.restore_sample_state(0, snapshot)

    restored = graph.state(0)
    assert restored.committed_id == "H0"
    assert restored.nodes["H0"].status == HypothesisStatus.COMMITTED
    assert restored.terminal_kind == "explicit_commit"
    assert restored.terminal_reason == "explicit_commit"

    graph.state(1).abstained = True
    graph.state(1).terminal_kind = "legacy_abstain"
    graph.state(1).terminal_reason = "policy_abstain"
    abstain_snapshot = graph.snapshot_sample_state(1)
    graph.clear(1)
    graph.restore_sample_state(1, abstain_snapshot)
    assert graph.state(1).abstained is True
    assert graph.state(1).terminal_kind == "legacy_abstain"
    assert graph.state(1).terminal_reason == "policy_abstain"


def test_restore_rejects_capacity_mismatch_without_mutating_graph():
    source = _rich_graph()
    snapshot = source.snapshot_sample_state(3)
    target = HypothesisGraph(max_active=4, max_nodes=12, max_execution_attempts=9)
    before = target.snapshot_sample_state(0)

    with pytest.raises(ValueError, match="capacity settings do not match"):
        target.restore_sample_state(0, snapshot)

    assert target.snapshot_sample_state(0) == before


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda snapshot: snapshot["state"]["nodes"][0].update(
                status="invented"
            ),
            "enum",
        ),
        (
            lambda snapshot: snapshot["state"]["edges"][0].update(
                target="H999"
            ),
            "unknown node",
        ),
        (
            lambda snapshot: snapshot["state"]["nodes"][0].pop("operation"),
            "malformed node",
        ),
        (
            lambda snapshot: snapshot["state"]["prune_certificates"]["H0"].update(
                node_id="H1"
            ),
            "node mismatch",
        ),
        (
            lambda snapshot: snapshot["state"].update(abstained=1),
            "abstained",
        ),
    ],
)
def test_restore_fails_closed_and_is_atomic(mutation, message):
    graph = _rich_graph()
    target_id = 8
    graph.state(target_id).turns_used = 2
    before = graph.snapshot_sample_state(target_id)
    malformed = deepcopy(graph.snapshot_sample_state(3))
    mutation(malformed)

    with pytest.raises(ValueError, match=message):
        graph.restore_sample_state(target_id, malformed)

    assert graph.snapshot_sample_state(target_id) == before
