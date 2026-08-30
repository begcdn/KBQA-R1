import importlib.util
import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "data_process"
    / "assemble_hyper_corrective_sft.py"
)
SPEC = importlib.util.spec_from_file_location("assemble_hyper_corrective_sft", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _prompt(question_id: str) -> str:
    return (
        "You are an expert assistant for querying Freebase with executable actions.\n"
        f"Question: question {question_id}\n\n"
        "HyPER-R1 executable hypothesis graph:\n- canonical"
    )


def _messages(question_id: str, state: str, action: str):
    messages = [
        {"role": "user", "content": _prompt(question_id), "loss_mask": 0},
    ]
    if state != "initial":
        messages.extend(
            [
                {"role": "assistant", "content": "", "loss_mask": 0},
                {
                    "role": "user",
                    "content": (
                        f"<information>state {state}</information>\n"
                        "<hypothesis_graph>active=1</hypothesis_graph>"
                    ),
                    "loss_mask": 0,
                },
            ]
        )
    messages.append(
        {
            "role": "assistant",
            "content": f"<think>canonical</think>\n<action>{action}</action>",
            "loss_mask": 1,
        }
    )
    return messages


def _partitioned_qids(seed: int, dev_fraction: float, count: int):
    result = {"train": [], "dev": []}
    index = 0
    while min(map(len, result.values())) < count:
        qid = f"q{index}"
        partition = MODULE._question_partition(qid, seed, dev_fraction)
        if len(result[partition]) < count:
            result[partition].append(qid)
        index += 1
    return result


def _write_jsonl(path: Path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _provenance():
    return {
        "repository_commit": "commit",
        "checkpoint_sha256": "checkpoint",
        "tokenizer_sha256": "tokenizer",
        "prompt_schema_version": "v23",
        "ranker_sha256": "ranker",
        "freebase_sha256": "freebase",
        "horizon": 32,
        "execution_budget": 24,
        "frontier_size": 24,
        "page_size": 6,
    }


def _ordinary_row(qid: str):
    return {
        "messages": _messages(qid, "ordinary", "Find_relation [ m.ordinary ]"),
        "data_source": "hyper_r1_verified_decision",
        "extra_info": {
            "question_id": qid,
            "family": "ordinary",
            "decision_index": 0,
        },
    }


def _successful_rollout(qid: str, index: int, masked: bool):
    return {
        "rollout_id": f"success-{qid}-{masked}",
        "question_id": qid,
        "source_split": "train",
        "masked": masked,
        "trajectory_success": True,
        "explicit_model_commit": True,
        "forced_terminal": False,
        "commit_answer_f1": 1.0,
        "family": "success",
        "decisions": [
            {
                "turn": index % 4,
                "accepted": True,
                "messages": _messages(
                    qid,
                    f"successful-{masked}-{index}",
                    "Inspect [ P0 ]",
                ),
            }
        ],
    }


def _protocol_rollout(qid: str, index: int, masked: bool):
    bad_messages = _messages(qid, f"protocol-{masked}-{index}", "Inspect [ P999 ]")
    correction = _messages(qid, f"protocol-{masked}-{index}", "Widen [ m.root ]")
    state_hash = f"protocol-state-{qid}-{masked}"
    return {
        "rollout_id": f"protocol-{qid}-{masked}",
        "question_id": qid,
        "source_split": "train",
        "masked": masked,
        "trajectory_success": False,
        "explicit_model_commit": False,
        "forced_terminal": False,
        "commit_answer_f1": 0.0,
        "family": "protocol",
        "decisions": [
            {
                "turn": 2,
                "accepted": False,
                "failure_kind": "protocol",
                "state_before_hash": state_hash,
                "messages": bad_messages,
                "correction": {
                    "messages": correction,
                    "legal_target_certified": True,
                    "state_hash": state_hash,
                    "certifier_hash": "legal-set",
                },
            }
        ],
    }


def _semantic_row(qid: str, index: int):
    messages = _messages(qid, f"semantic-{index}", "Recall [ H1 ]")
    state_hash = MODULE._state_hash(messages)
    return {
        "rollout_id": f"semantic-{qid}",
        "question_id": qid,
        "source_split": "train",
        "rollout_mode": "masked",
        "turn": 8,
        "family": "semantic",
        "messages": messages,
        "certification": {
            "schema_version": MODULE.SEMANTIC_CERTIFICATE_VERSION,
            "certified": True,
            "legal_action": True,
            "execution_verified": True,
            "budget_respected": True,
            "intent_equivalent": True,
            "first_meaningful_failure": True,
            "state_hash": state_hash,
            "target_action": "Recall [ H1 ]",
            "ranker_sha256": "ranker",
            "freebase_sha256": "freebase",
            "executor_hash": "executor",
            "certifier_hash": "certifier",
            "epsilon": 0.01,
            "q_values": {"Recall [ H1 ]": 1.0, "Commit [ H0 ]": 0.4},
            "optimal_actions": ["Recall [ H1 ]"],
        },
    }


def _fixture(tmp_path: Path, *, seed=17, dev_fraction=0.5):
    qids = _partitioned_qids(seed, dev_fraction, 20)
    all_qids = qids["train"] + qids["dev"]
    ordinary = tmp_path / "ordinary.parquet"
    pq.write_table(pa.Table.from_pylist([_ordinary_row(qid) for qid in all_qids]), ordinary)

    masked = []
    unmasked = []
    semantic = []
    for index, qid in enumerate(all_qids):
        masked.append(_successful_rollout(qid, index, True))
        masked.append(_protocol_rollout(qid, index, True))
        unmasked.append(_successful_rollout(qid, index, False))
        unmasked.append(_protocol_rollout(qid, index, False))
        semantic.append(_semantic_row(qid, index))
    masked_path = tmp_path / "masked.jsonl"
    unmasked_path = tmp_path / "unmasked.jsonl"
    semantic_path = tmp_path / "semantic.jsonl"
    _write_jsonl(masked_path, masked)
    _write_jsonl(unmasked_path, unmasked)
    _write_jsonl(semantic_path, semantic)
    test_ids = tmp_path / "test_ids.txt"
    test_ids.write_text("held-out-test\n", encoding="utf-8")
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps(_provenance()), encoding="utf-8")
    return {
        "ordinary_v23": ordinary,
        "masked_rollouts": masked_path,
        "unmasked_rollouts": unmasked_path,
        "semantic_recoveries": semantic_path,
        "test_question_ids": test_ids,
        "provenance_config": provenance,
        "seed": seed,
        "dev_fraction": dev_fraction,
    }


def test_assembles_exact_deterministic_question_disjoint_mixture(tmp_path):
    inputs = _fixture(tmp_path)
    manifests = []
    for name in ("one", "two"):
        manifest = MODULE.assemble_corrective_dataset(
            **inputs,
            output=tmp_path / name,
            train_size=20,
            dev_size=20,
        )
        manifests.append(manifest)

    expected = {
        "ordinary_v23": 10,
        "autonomous_success": 5,
        "protocol_recovery": 3,
        "semantic_recovery": 2,
    }
    assert manifests[0]["actual_rows"] == {"train": expected, "dev": expected}
    assert manifests[0]["output_sha256"] == manifests[1]["output_sha256"]
    assert manifests[0]["student_visible_recovery_marker"] is False

    train = pq.read_table(tmp_path / "one" / "train_decision.parquet").to_pylist()
    dev = pq.read_table(tmp_path / "one" / "dev_decision.parquet").to_pylist()
    train_qids = {row["extra_info"]["question_id"] for row in train}
    dev_qids = {row["extra_info"]["question_id"] for row in dev}
    assert train_qids.isdisjoint(dev_qids)
    assert "held-out-test" not in train_qids | dev_qids
    assert all(row["extra_info"]["state_weight"] == 1.0 for row in train + dev)
    assert all("recovery mode" not in str(row["messages"]).lower() for row in train + dev)
    recovery_rollouts = [
        row["extra_info"]["source_rollout_id"]
        for row in train + dev
        if "recovery" in row["extra_info"]["correction_source"]
    ]
    assert len(recovery_rollouts) == len(set(recovery_rollouts))


def test_selects_second_identical_no_progress_before_later_protocol_failure(tmp_path):
    qid = "train-question"
    first = _messages(qid, "loop", "Find_relation [ m.root ]")
    second = _messages(qid, "loop", "Find_relation [ m.root ]")
    correction = _messages(qid, "loop", "Widen [ m.root ]")
    state_hash = "same-state"
    rollout = {
        "rollout_id": "loop-rollout",
        "question_id": qid,
        "source_split": "train",
        "masked": False,
        "trajectory_success": False,
        "explicit_model_commit": False,
        "forced_terminal": False,
        "commit_answer_f1": 0.0,
        "decisions": [
            {
                "turn": 0,
                "action": "Find_relation [ m.root ]",
                "no_progress": True,
                "state_before_hash": state_hash,
                "state_after_hash": state_hash,
                "messages": first,
            },
            {
                "turn": 1,
                "action": "Find_relation [ m.root ]",
                "no_progress": True,
                "state_before_hash": state_hash,
                "state_after_hash": state_hash,
                "messages": second,
                "correction": {
                    "messages": correction,
                    "legal_target_certified": True,
                    "state_hash": state_hash,
                    "certifier_hash": "legal-set",
                },
            },
            {
                "turn": 2,
                "failure_kind": "protocol",
                "state_before_hash": "later",
                "messages": _messages(qid, "later", "Inspect [ P999 ]"),
                "correction": {
                    "messages": _messages(qid, "later", "Inspect [ P0 ]"),
                    "legal_target_certified": True,
                    "state_hash": "later",
                    "certifier_hash": "legal-set",
                },
            },
        ],
    }
    path = tmp_path / "unmasked.jsonl"
    _write_jsonl(path, [rollout])
    _, recoveries = MODULE._load_rollout_candidates(
        path,
        expected_masked=False,
        seed=1,
        dev_fraction=0.5,
        test_ids={"test"},
        train_ids={qid},
    )
    assert len(recoveries) == 1
    assert recoveries[0].turn == 1
    assert recoveries[0].failure_kind == "no_progress_loop"
    assert "Widen [ m.root ]" in recoveries[0].record["messages"][-1]["content"]


def test_uncertified_semantic_recovery_fails_closed(tmp_path):
    inputs = _fixture(tmp_path)
    rows = list(MODULE._read_jsonl(inputs["semantic_recoveries"]))
    for row in rows:
        row.pop("__line_number__", None)
    rows[-1]["certification"]["execution_verified"] = False
    _write_jsonl(inputs["semantic_recoveries"], rows)
    with pytest.raises(ValueError, match="uncertified semantic recovery"):
        MODULE.assemble_corrective_dataset(
            **inputs,
            output=tmp_path / "output",
            train_size=20,
            dev_size=20,
        )


def test_test_question_in_ordinary_input_is_rejected(tmp_path):
    inputs = _fixture(tmp_path)
    ordinary = pq.read_table(inputs["ordinary_v23"]).to_pylist()
    leaked = ordinary[0]["extra_info"]["question_id"]
    inputs["test_question_ids"].write_text(leaked + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contains test question"):
        MODULE.assemble_corrective_dataset(
            **inputs,
            output=tmp_path / "output",
            train_size=20,
            dev_size=20,
        )


def test_semantic_optimal_set_must_match_q_values(tmp_path):
    inputs = _fixture(tmp_path)
    rows = list(MODULE._read_jsonl(inputs["semantic_recoveries"]))
    for row in rows:
        row.pop("__line_number__", None)
    rows[0]["certification"]["optimal_actions"] = ["Commit [ H0 ]"]
    _write_jsonl(inputs["semantic_recoveries"], rows)
    with pytest.raises(ValueError, match="optimal action set"):
        MODULE.assemble_corrective_dataset(
            **inputs,
            output=tmp_path / "output",
            train_size=20,
            dev_size=20,
        )
