from types import MethodType, SimpleNamespace

import numpy as np

from kbqa_r1.action_constraints import HyPERActionConstraintSpec
from kbqa_r1.hyper_r1 import HypothesisGraph
from kbqa_r1.llm_agent.sexpr_generation import SExprLLMGenerationManager


def _manager(masked: bool):
    manager = SExprLLMGenerationManager.__new__(SExprLLMGenerationManager)
    manager.hyper_structural_constraints = masked
    manager.hyper_graph = HypothesisGraph()
    manager._hyper_frontiers = {}
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
        }
    )

    manager._initialize_hyper_decision_traces(batch, 2)

    roots = manager._hyper_trace_roots
    assert roots[0]["rollout_id"] != roots[1]["rollout_id"]
    assert roots[0]["source_split"] == roots[1]["source_split"] == "train"
