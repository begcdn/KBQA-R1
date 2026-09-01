import importlib.util
from pathlib import Path

from kbqa_r1.action_constraints import HyPERActionConstraintSpec
from kbqa_r1.hyper_r1 import GraphActionAffordances
from scripts.data_process import assemble_hyper_corrective_sft as ASSEMBLER


MODULE_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "data_process"
    / "certify_hyper_semantic_commits.py"
)
SPEC = importlib.util.spec_from_file_location("certify_hyper_semantic_commits", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _row():
    messages = [
        {
            "role": "user",
            "content": "HyPER-R1 executable hypothesis graph:\nquestion",
            "loss_mask": 0,
        },
        {
            "role": "assistant",
            "content": "<think>done</think>\n<action>Commit [ H0 ]</action>",
            "loss_mask": 1,
        },
    ]
    constraint = HyPERActionConstraintSpec.build(
        state_key="state",
        turn=5,
        affordances=GraphActionAffordances(commit=("H0", "H1")),
        allow_open_operators=False,
    )
    return {
        "rollout_id": "rollout-1",
        "question_id": "q1",
        "source_split": "train",
        "family": "linear",
        "masked": False,
        "explicit_model_commit": True,
        "commit_answer_f1": 0.0,
        "decisions": [
            {
                "turn": 5,
                "accepted": True,
                "raw_action": "Commit [ H0 ]",
                "messages": messages,
                "state_before_hash": MODULE._state_hash(messages),
                "constraint_spec": constraint.to_dict(),
                "private_execution_state": {"snapshot": True},
            }
        ],
    }


def _evaluate(_decision, action):
    exact = action == "Commit [ H1 ]"
    return {
        "accepted": True,
        "explicit_commit": True,
        "answer_f1": 1.0 if exact else 0.0,
        "answer_exact": exact,
        "intent_equivalent": exact,
        "constraint_digest": HyPERActionConstraintSpec.from_dict(
            _decision["constraint_spec"]
        ).digest,
    }


def test_certifies_exhaustive_terminal_commit_regret():
    certified, status, diagnostic = MODULE.certify_commit_rollout(
        _row(),
        _evaluate,
        ranker_sha256="ranker",
        freebase_sha256="freebase",
        executor_hash="executor",
    )

    assert status == "certified"
    assert diagnostic["best_semantic_q"] == 1.0
    assert certified is not None
    assert certified["messages"][-1]["content"].endswith(
        "<action>Commit [ H1 ]</action>"
    )
    certificate = certified["certification"]
    assert certificate["decision_scope"] == (
        "all_terminal_commit_actions_in_live_contract"
    )
    assert certificate["q_values"] == {
        "Commit [ H0 ]": 0.0,
        "Commit [ H1 ]": 1.0,
    }
    assert certificate["optimal_actions"] == ["Commit [ H1 ]"]
    assert ASSEMBLER._validate_semantic_certificate(
        certified,
        action="Commit [ H1 ]",
        state_hash=MODULE._state_hash(certified["messages"]),
        provenance={"ranker_sha256": "ranker", "freebase_sha256": "freebase"},
    ) == MODULE._digest(certificate)


def test_rejects_when_wrong_commit_is_not_strictly_suboptimal():
    def tied(decision, action):
        result = dict(_evaluate(decision, action))
        result["answer_f1"] = 1.0
        result["answer_exact"] = True
        result["intent_equivalent"] = True
        return result

    certified, status, _ = MODULE.certify_commit_rollout(
        _row(),
        tied,
        ranker_sha256="ranker",
        freebase_sha256="freebase",
        executor_hash="executor",
    )

    assert certified is None
    assert status == "no_positive_regret"


def test_rejects_answer_match_without_formal_intent_equivalence():
    def answer_only(decision, action):
        result = dict(_evaluate(decision, action))
        if action == "Commit [ H1 ]":
            result["intent_equivalent"] = False
        return result

    certified, status, diagnostic = MODULE.certify_commit_rollout(
        _row(),
        answer_only,
        ranker_sha256="ranker",
        freebase_sha256="freebase",
        executor_hash="executor",
    )

    assert certified is None
    assert status == "no_exact_intent_equivalent_alternative"
    assert diagnostic["answer_exact_candidates"] == 1
    assert diagnostic["intent_equivalent_candidates"] == 0


def test_rejects_single_commit_candidate():
    row = _row()
    spec = HyPERActionConstraintSpec.build(
        state_key="state",
        turn=5,
        affordances=GraphActionAffordances(commit=("H0",)),
        allow_open_operators=False,
    )
    row["decisions"][0]["constraint_spec"] = spec.to_dict()

    certified, status, diagnostic = MODULE.certify_commit_rollout(
        row,
        _evaluate,
        ranker_sha256="ranker",
        freebase_sha256="freebase",
        executor_hash="executor",
    )

    assert certified is None
    assert status == "insufficient_commit_candidates"
    assert diagnostic["candidate_count"] == 1
