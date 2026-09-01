import importlib.util
from pathlib import Path

import pytest

from kbqa_r1.action_constraints import HyPERActionConstraintSpec
from kbqa_r1.hyper_r1 import GraphActionAffordances
from scripts.data_process import assemble_hyper_corrective_sft as ASSEMBLER


MODULE_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "data_process"
    / "certify_hyper_gold_recoveries.py"
)
SPEC = importlib.util.spec_from_file_location("certify_hyper_gold_recoveries", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _decision(turn, action):
    messages = [
        {
            "role": "user",
            "content": f"HyPER-R1 executable hypothesis graph:\nstate-{turn}",
            "loss_mask": 0,
        },
        {
            "role": "assistant",
            "content": f"<think>observed</think>\n<action>{action}</action>",
            "loss_mask": 1,
        },
    ]
    constraint = HyPERActionConstraintSpec.build(
        state_key=f"state-{turn}",
        turn=turn,
        affordances=GraphActionAffordances(select=("H0",), recall=("H1",)),
        allow_open_operators=False,
    )
    return {
        "turn": turn,
        "accepted": True,
        "raw_action": action,
        "raw_response": messages[-1]["content"],
        "messages": messages,
        "state_before_hash": MODULE._state_hash(messages),
        "constraint_spec": constraint.to_dict(),
        "private_execution_state": {"snapshot": True},
    }


def _row():
    return {
        "rollout_id": "rollout-1",
        "question_id": "q1",
        "source_split": "train",
        "family": "recovery",
        "masked": False,
        "decisions": [
            _decision(2, "Select [ H0 ]"),
            _decision(3, "Select [ H0 ]"),
        ],
    }


def _outcome(decision, action, q):
    spec = HyPERActionConstraintSpec.from_dict(decision["constraint_spec"])
    first = "Recall [ H1 ]" if action is None else action
    exact = q == 1.0
    return {
        "success": exact,
        "status": "exact_intent_commit" if exact else "terminal_not_exact_intent",
        "actions": [first, "Commit [ H1 ]"],
        "semantic_q": q,
        "first_action_accepted": True,
        "explicit_commit": True,
        "answer_exact": exact,
        "intent_equivalent": True,
        "constraint_digest": spec.digest,
        "start_turn": decision["turn"],
        "max_turns": 24,
        "continuation_policy": "bounded_gold_oracle_after_first_action",
    }


def test_certifies_earliest_pairwise_regret_and_validates_in_assembler():
    def evaluate(decision, action):
        if decision["turn"] == 2:
            return _outcome(decision, action, 1.0)
        return _outcome(decision, action, 1.0 if action is None else 0.25)

    certified, status, diagnostics = MODULE.certify_pairwise_rollout(
        _row(),
        evaluate,
        ranker_sha256="ranker",
        freebase_sha256="freebase",
        executor_hash="executor",
    )

    assert status == "certified"
    assert certified is not None
    assert certified["turn"] == 3
    assert diagnostics[0]["status"] == "no_positive_regret"
    certificate = certified["certification"]
    assert certificate["target_q"] == 1.0
    assert certificate["failed_q"] == 0.25
    assert certificate["earlier_states_examined"] == 1
    assert ASSEMBLER._validate_semantic_certificate(
        certified,
        action="Recall [ H1 ]",
        state_hash=MODULE._state_hash(certified["messages"]),
        provenance={"ranker_sha256": "ranker", "freebase_sha256": "freebase"},
    ) == MODULE._digest(certificate)


def test_rejects_when_observed_branch_has_no_regret():
    def evaluate(decision, action):
        return _outcome(decision, action, 1.0)

    certified, status, diagnostics = MODULE.certify_pairwise_rollout(
        _row(),
        evaluate,
        ranker_sha256="ranker",
        freebase_sha256="freebase",
        executor_hash="executor",
    )

    assert certified is None
    assert status == "no_certified_pairwise_regret"
    assert {value["status"] for value in diagnostics} == {"no_positive_regret"}


def test_rejects_mismatched_branch_budget_contract():
    def evaluate(decision, action):
        result = _outcome(decision, action, 1.0 if action is None else 0.0)
        if action is not None:
            result["max_turns"] = 23
        return result

    certified, status, diagnostics = MODULE.certify_pairwise_rollout(
        _row(),
        evaluate,
        ranker_sha256="ranker",
        freebase_sha256="freebase",
        executor_hash="executor",
    )

    assert certified is None
    assert status == "branch_replay_mismatch"
    assert {value["status"] for value in diagnostics} == {"branch_contract_mismatch"}


def test_earlier_protocol_failure_blocks_later_semantic_certificate():
    row = _row()
    row["decisions"][0]["accepted"] = False
    row["decisions"][0]["failure_kind"] = "protocol"

    def evaluate(decision, action):
        return _outcome(decision, action, 1.0 if action is None else 0.0)

    certified, status, diagnostics = MODULE.certify_pairwise_rollout(
        row,
        evaluate,
        ranker_sha256="ranker",
        freebase_sha256="freebase",
        executor_hash="executor",
    )

    assert certified is None
    assert status == "protocol_failure_precedes_semantic_regret"
    assert diagnostics == [
        {
            "turn": 2,
            "model_action": "Select [ H0 ]",
            "status": "protocol_failure_precedes_semantic_regret",
        }
    ]


def test_assembler_rejects_pairwise_claim_without_global_upper_bound():
    def evaluate(decision, action):
        return _outcome(decision, action, 1.0 if action is None else 0.0)

    certified, _, _ = MODULE.certify_pairwise_rollout(
        _row(),
        evaluate,
        ranker_sha256="ranker",
        freebase_sha256="freebase",
        executor_hash="executor",
    )
    assert certified is not None
    certified["certification"]["global_upper_bound_achieved"] = False
    with pytest.raises(ValueError, match="pairwise protocol metadata"):
        ASSEMBLER._validate_semantic_certificate(
            certified,
            action="Recall [ H1 ]",
            state_hash=MODULE._state_hash(certified["messages"]),
            provenance={"ranker_sha256": "ranker", "freebase_sha256": "freebase"},
        )


def test_assembler_rejects_pairwise_branch_action_mismatch():
    def evaluate(decision, action):
        return _outcome(decision, action, 1.0 if action is None else 0.0)

    certified, _, _ = MODULE.certify_pairwise_rollout(
        _row(),
        evaluate,
        ranker_sha256="ranker",
        freebase_sha256="freebase",
        executor_hash="executor",
    )
    assert certified is not None
    certified["certification"]["failed_outcome"]["actions"][0] = "Recall [ H1 ]"
    with pytest.raises(ValueError, match="branch first actions"):
        ASSEMBLER._validate_semantic_certificate(
            certified,
            action="Recall [ H1 ]",
            state_hash=MODULE._state_hash(certified["messages"]),
            provenance={"ranker_sha256": "ranker", "freebase_sha256": "freebase"},
        )
