import importlib.util
from pathlib import Path
import sys

from kbqa_r1.action_constraints import HyPERActionConstraintSpec


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "data_process"
    / "certify_hyper_trace_corrections.py"
)
SPEC = importlib.util.spec_from_file_location("certify_hyper_trace_corrections", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _messages(response: str):
    return [
        {"role": "user", "content": "prompt", "loss_mask": 0},
        {"role": "assistant", "content": response, "loss_mask": 1},
    ]


def _decision(turn: int, response: str, *, accepted: bool, failure: str = ""):
    spec = HyPERActionConstraintSpec(
        state_key="state",
        turn=turn,
        exact_actions=("Select [ H1 ]", "Commit [ H1 ]"),
    )
    messages = _messages(response)
    return {
        "turn": turn,
        "messages": messages,
        "raw_response": response,
        "raw_action": response.split("<action>", 1)[-1].split("</action>", 1)[0],
        "accepted": accepted,
        "failure_kind": failure,
        "no_progress": not accepted,
        "state_before_hash": MODULE._state_hash(messages),
        "progress_before_hash": "same-progress",
        "progress_after_hash": "same-progress" if not accepted else "advanced",
        "constraint_spec": spec.to_dict(),
    }


def _rollout(decisions, *, exact=True):
    return {
        "rollout_id": "rollout",
        "question_id": "question",
        "trajectory_success": exact,
        "explicit_model_commit": exact,
        "forced_terminal": False,
        "commit_answer_f1": 1.0 if exact else 0.0,
        "decisions": decisions,
    }


def test_certifies_later_legal_action_from_unchanged_state():
    failed = _decision(
        2,
        "<think>bad</think>\n<action>Select [ H9 ]</action>",
        accepted=False,
        failure="stale_id",
    )
    corrected = _decision(
        3,
        "<think>recover</think>\n<action>Select [ H1 ]</action>",
        accepted=True,
    )

    row, status = MODULE.certify_rollout(_rollout([failed, corrected]))

    assert status == "certified_recovery"
    correction = row["decisions"][0]["correction"]
    assert correction["legal_target_certified"] is True
    assert correction["messages"][-1]["content"] == corrected["raw_response"]
    assert correction["state_hash"] == failed["state_before_hash"]
    assert correction["constraint_digest"] == HyPERActionConstraintSpec.from_dict(
        failed["constraint_spec"]
    ).digest


def test_rejects_candidate_outside_original_state_contract():
    failed = _decision(
        2,
        "<think>bad</think>\n<action>Select [ H9 ]</action>",
        accepted=False,
        failure="stale_id",
    )
    candidate = _decision(
        3,
        "<think>different</think>\n<action>Park [ H1 ]</action>",
        accepted=True,
    )

    row, status = MODULE.certify_rollout(_rollout([failed, candidate]))

    assert row is None
    assert status == "first_failure_uncertified"


def test_rejects_legal_action_that_failed_inside_the_executor():
    failed = _decision(
        2,
        "<think>legal but bad</think>\n<action>Select [ H1 ]</action>",
        accepted=False,
        failure="protocol",
    )
    corrected = _decision(
        3,
        "<think>recover</think>\n<action>Commit [ H1 ]</action>",
        accepted=True,
    )

    row, status = MODULE.certify_rollout(_rollout([failed, corrected]))

    assert row is None
    assert status == "first_failure_uncertified"


def test_exact_state_replay_can_certify_legal_executor_failure():
    failed = _decision(
        2,
        "<think>legal but rejected</think>\n<action>Select [ H1 ]</action>",
        accepted=False,
        failure="protocol",
    )
    snapshot = {
        "schema_version": "hyper-execution-state-v1",
        "latest_observation": "before",
        "graph": {"node": "H1"},
    }
    failed["private_execution_state"] = snapshot
    failed["execution_snapshot_hash"] = MODULE._digest(snapshot)
    failed["execution_state_hash"] = MODULE._execution_state_hash(snapshot)
    corrected = _decision(
        3,
        "<think>recover</think>\n<action>Commit [ H1 ]</action>",
        accepted=True,
    )
    corrected["execution_state_hash"] = "later-turn-clock"
    calls = []

    def replay(row, failed_decision, candidate, suffix):
        calls.append((row, failed_decision, candidate, suffix))
        return {
            "certified": True,
            "failed_reproduced": True,
            "target_accepted": True,
            "target_made_progress": True,
            "explicit_exact_completion": True,
            "intent_equivalent": True,
            "answer_f1": 1.0,
        }

    row, status = MODULE.certify_rollout(
        _rollout([failed, corrected]),
        replay_verifier=replay,
    )

    assert status == "certified_recovery"
    correction = row["decisions"][0]["correction"]
    assert correction["evidence"]["exact_state_replayed"] is True
    assert correction["evidence"]["failed_response_outside_contract"] is False
    assert correction["evidence"]["replay"]["answer_f1"] == 1.0
    assert calls[0][2]["raw_action"] == "Commit [ H1 ]"


def test_exact_state_replay_fails_closed_when_suffix_is_not_verified():
    failed = _decision(
        2,
        "<think>legal but rejected</think>\n<action>Select [ H1 ]</action>",
        accepted=False,
        failure="protocol",
    )
    corrected = _decision(
        3,
        "<think>recover</think>\n<action>Commit [ H1 ]</action>",
        accepted=True,
    )

    row, status = MODULE.certify_rollout(
        _rollout([failed, corrected]),
        replay_verifier=lambda *_: {
            "certified": True,
            "failed_reproduced": True,
            "target_accepted": True,
            "target_made_progress": True,
            "explicit_exact_completion": True,
            "intent_equivalent": False,
        },
    )

    assert row is None
    assert status == "first_failure_uncertified"


def test_rejects_recovery_without_exact_explicit_completion():
    failed = _decision(
        2,
        "<think>bad</think>\n<action>Select [ H9 ]</action>",
        accepted=False,
        failure="stale_id",
    )
    corrected = _decision(
        3,
        "<think>recover</think>\n<action>Select [ H1 ]</action>",
        accepted=True,
    )

    row, status = MODULE.certify_rollout(_rollout([failed, corrected], exact=False))

    assert row is None
    assert status == "failed_trajectory_not_exact"


def test_clean_rollout_is_preserved_without_fabricated_correction():
    clean = _decision(
        0,
        "<think>good</think>\n<action>Select [ H1 ]</action>",
        accepted=True,
    )

    row, status = MODULE.certify_rollout(_rollout([clean]))

    assert status == "clean"
    assert "correction" not in row["decisions"][0]


def test_rejects_later_action_from_a_different_executable_state():
    failed = _decision(
        2,
        "<think>bad</think>\n<action>Select [ H9 ]</action>",
        accepted=False,
        failure="stale_id",
    )
    snapshot = {
        "schema_version": "hyper-execution-state-v1",
        "latest_observation": "before",
        "graph": {"node": "H1"},
    }
    failed["private_execution_state"] = snapshot
    failed["execution_snapshot_hash"] = MODULE._digest(snapshot)
    failed["execution_state_hash"] = MODULE._execution_state_hash(snapshot)
    corrected = _decision(
        3,
        "<think>recover</think>\n<action>Select [ H1 ]</action>",
        accepted=True,
    )
    corrected["execution_state_hash"] = "different-state"

    row, status = MODULE.certify_rollout(_rollout([failed, corrected]))

    assert row is None
    assert status == "first_failure_uncertified"
