import json
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from kbqa_r1.action_constraints import HyPERActionConstraintSpec
from kbqa_r1.fork_r1 import ForkDecision, RelationCandidate
from kbqa_r1.hyper_r1 import HypothesisGraph
from kbqa_r1.llm_agent.sexpr_generation import SExprLLMGenerationManager
from kbqa_r1.llm_agent.sexpr_state_manager import SExprStateManager


def _manager(masked: bool):
    manager = SExprLLMGenerationManager.__new__(SExprLLMGenerationManager)
    manager.hyper_structural_constraints = masked
    manager.hyper_graph = HypothesisGraph()
    manager.hyper_frontier_width = 6
    manager.tokenizer = SimpleNamespace(pad_token_id=0)
    manager._hyper_frontiers = {}
    manager._hyper_next_proposal_index = {}
    manager._hyper_trace_roots = {
        0: {
            "prompt": {
                "role": "user",
                "content": "Question\n\nHyPER-R1 executable hypothesis graph:\n- contract",
                "loss_mask": 0,
            }
        }
    }
    manager._hyper_latest_observations = {}
    manager._hyper_pending_decisions = {}
    manager._hyper_decision_traces = {0: []}
    manager._hyper_action_records = {}
    manager._hyper_gold_contracts = {}
    manager.state_manager = SExprStateManager()

    def constraint(self, sample_id, turn):
        return HyPERActionConstraintSpec(
            state_key=f"state-{turn}",
            turn=turn,
            exact_actions=("Find_relation [ m.1 ]",),
            allow_open_operators=False,
        )

    manager._hyper_action_constraint = MethodType(constraint, manager)
    return manager


def test_masked_and_unmasked_capture_the_same_decision_contract():
    records = []
    for masked in (False, True):
        manager = _manager(masked)
        non_tensors, payloads = manager._prepare_hyper_turn_contracts([0], 0)
        manager._hyper_action_records[0] = [{"turn": 0}]
        manager._finish_hyper_turn_trace(
            0,
            0,
            "<think>x</think>\n<action>Find_relation [ m.1 ]</action>",
            "<information>accepted</information>",
        )
        record = manager._hyper_decision_traces[0][0]
        assert record["constraint_spec"] == payloads[0]
        assert record["constraint_spec"]["digest"]
        assert record["state_before_hash"] == manager._trace_hash(
            record["messages"][:-1]
        )
        assert ("hyper_action_constraint" in non_tensors) is masked
        records.append(record)

    assert records[0] == records[1]


def test_progress_hash_ignores_turn_clock():
    manager = _manager(False)
    manager.hyper_graph.set_clock(0, turns_used=1, max_turns=10)
    first = manager._hyper_progress_hash(0)
    manager.hyper_graph.set_clock(0, turns_used=7, max_turns=10)
    second = manager._hyper_progress_hash(0)

    assert first == second


def test_execution_state_round_trip_preserves_graph_and_relation_catalog():
    manager = _manager(False)
    manager.state_manager.update_sample_function_state(
        0, "expression1 = START('m.1')"
    )
    manager.state_manager.set_current_expression_id(0, 1)
    manager.state_manager.set_sample_entities(0, [("m.1", "Topic")])
    manager.state_manager.set_sample_prompt(0, "Question")
    manager.hyper_graph.register_public_question(0, "Who is the topic?")
    manager.hyper_graph.add_executed(
        sample_id=0,
        function_state=("expression1 = START('m.1')",),
        target_expression="expression1",
        sexpr="m.1",
        denotation=("m.1",),
        parent_id=None,
        operation="start",
    )
    decision = ForkDecision(
        sample_id=0,
        turn=0,
        action_index=0,
        step_number=0,
        entity_argument="m.1",
        relation_prompt="topic relation",
        chosen_relation="people.person.parents",
        ranked_relations=(
            RelationCandidate("people.person.parents", 0.9),
            RelationCandidate("people.person.children", 0.8),
        ),
        resolver_margin=0.1,
        state_before=("expression1 = START('m.1')",),
        expression_counter=1,
        entities=(("m.1", "Topic"),),
        prompt="Question",
        raw_action="Find_relation [ m.1 ]",
    )
    manager._hyper_frontiers[0] = [
        {
            "source": "m.1",
            "decision": decision,
            "parent_id": None,
            "contrast_group": "turn0:catalog0:parentroot",
            "proposals": {
                "P0": {
                    "candidate": decision.ranked_relations[0],
                    "rank": 1,
                    "status": "visible",
                }
            },
            "next_offset": 1,
            "closed": False,
        }
    ]
    manager._hyper_next_proposal_index[0] = 1
    manager._hyper_action_records[0] = [{"action_index": 1, "turn": 0}]
    manager._hyper_latest_observations[0] = "<information>state</information>"
    manager._hyper_gold_contracts[0] = {"function_list": ["JOIN"]}
    expected_graph = manager.hyper_graph.to_dict(0)
    expected_frontier = manager._public_frontier_signature(0)

    snapshot = manager._capture_hyper_execution_state(0)
    json.dumps(snapshot)
    manager.state_manager.clear_sample_state(0)
    manager.hyper_graph.clear(0)
    manager._hyper_frontiers[0] = []
    manager._hyper_action_records[0] = []
    manager._hyper_latest_observations.pop(0)
    manager._hyper_gold_contracts.pop(0)

    manager._restore_hyper_execution_state(0, snapshot)

    assert manager.hyper_graph.to_dict(0) == expected_graph
    assert manager._trace_hash(
        manager._capture_hyper_execution_state(0)
    ) == manager._trace_hash(snapshot)
    assert manager._public_frontier_signature(0) == expected_frontier
    assert manager._hyper_next_proposal_index[0] == 1
    assert manager._hyper_action_records[0] == [{"action_index": 1, "turn": 0}]
    assert manager._hyper_latest_observations[0] == "<information>state</information>"
    assert manager._hyper_gold_contracts[0] == {"function_list": ["JOIN"]}

    manager._restore_hyper_execution_state(1, snapshot)
    assert manager.hyper_graph.state(1).sample_id == 1
    assert manager._hyper_frontiers[1][0]["decision"].sample_id == 1


def test_rejected_action_keeps_exact_failure_observation_and_no_progress():
    manager = _manager(False)
    manager._prepare_hyper_turn_contracts([0], 0)

    manager._finish_hyper_turn_trace(
        0,
        0,
        "<think>x</think>\n<action>Select [ H99 ]</action>",
        (
            "template-prefix<information>Graph action failed: "
            "unknown hypothesis H99.</information>template-suffix"
        ),
    )

    record = manager._hyper_decision_traces[0][0]
    assert record["accepted"] is False
    assert record["failure_kind"] == "stale_id"
    assert record["no_progress"] is True
    assert record["failure_observation"] == (
        "<information>Graph action failed: unknown hypothesis H99.</information>"
    )
    assert record["acceptance_observation"] == ""


def test_trace_roots_make_repeated_rollouts_unique_and_preserve_split():
    manager = _manager(False)
    manager._call_counter = 12
    manager.is_validation = True
    manager._hyper_trace_roots = {}
    manager._hyper_decision_traces = {}
    batch = SimpleNamespace(
        non_tensor_batch={
            "raw_prompt": np.asarray(
                [
                    [
                        {
                            "role": "user",
                            "content": (
                                "Question: q\n\n"
                                "HyPER-R1 executable hypothesis graph:\n- contract"
                            ),
                        }
                    ],
                    [
                        {
                            "role": "user",
                            "content": (
                                "Question: q\n\n"
                                "HyPER-R1 executable hypothesis graph:\n- contract"
                            ),
                        }
                    ],
                ],
                dtype=object,
            ),
            "extra_info": np.asarray(
                [
                    {"question_id": "q", "source_split": "train"},
                    {"question_id": "q", "source_split": "train"},
                ],
                dtype=object,
            ),
            "uid": np.asarray(["shared", "shared"], dtype=object),
            "raw_prompt_ids": np.asarray([[1, 2], [1, 2]], dtype=object),
        }
    )

    manager._initialize_hyper_decision_traces(
        batch, 2, torch.tensor([[0, 1, 2], [0, 1, 2]])
    )

    roots = manager._hyper_trace_roots
    assert roots[0]["rollout_id"] != roots[1]["rollout_id"]
    assert roots[0]["source_split"] == roots[1]["source_split"] == "train"


def test_trace_initialization_rejects_prompt_token_mismatch():
    manager = _manager(False)
    manager._call_counter = 1
    manager.is_validation = False
    batch = SimpleNamespace(
        non_tensor_batch={
            "raw_prompt": np.asarray(
                [[{"role": "user", "content": (
                    "Question: q\n\nHyPER-R1 executable hypothesis graph:\n- contract"
                )}]],
                dtype=object,
            ),
            "raw_prompt_ids": np.asarray([[1, 2, 3]], dtype=object),
            "extra_info": np.asarray([{"question_id": "q"}], dtype=object),
            "uid": np.asarray(["rollout"], dtype=object),
        }
    )

    with pytest.raises(RuntimeError, match="model-visible token IDs"):
        manager._initialize_hyper_decision_traces(
            batch, 1, torch.tensor([[0, 1, 9]])
        )
