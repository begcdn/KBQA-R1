import gzip
import importlib.util
import json
from pathlib import Path

import pandas as pd

from kbqa_r1.hyper_data import (
    DemonstrationStep,
    ExecutedHypothesis,
    HyperDemonstration,
)


SCRIPT = Path(__file__).parents[1] / "scripts" / "data_process" / "repair_hyper_v5.py"
SPEC = importlib.util.spec_from_file_location("repair_hyper_v5", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _demo(family, rationale, visible_before=("H0", "H1"), max_active=2):
    nodes = {
        "H0": ExecutedHypothesis(
            "H0", ("expression1 = START('m.topic')",), "expression1", ("m.answer",)
        ),
        "H1": ExecutedHypothesis(
            "H1", ("expression1 = START('m.topic')", "expression1 = JOIN('r.alt', expression1)"),
            "expression1", ("m.other",), relation="r.alt", role="alternative"
        ),
    }
    steps = [
        DemonstrationStep("Find_relation", ("m.topic",), (), ("H0", "H1")),
        DemonstrationStep("Prune", ("H1",), visible_before, (), (rationale,)),
        DemonstrationStep("Commit", ("H0",), ("H0",), (), ("complete", "executable")),
    ]
    return HyperDemonstration(
        "demo-" + family,
        "question-" + family,
        "Which answer?",
        family,
        nodes,
        steps,
        ("m.answer",),
        {"max_active": max_active},
    )


def test_repair_converts_adaptive_nonempty_mismatch_by_pre_repair_capacity():
    full, counts = MODULE.repair_demonstration(
        _demo("adaptive_frontier_widen", "question_path_mismatch:r.alt")
    )
    assert full.steps[1].rationale_facts == ("frontier_capacity_eviction",)
    assert counts["converted_to_eviction"] == 1

    reserved, counts = MODULE.repair_demonstration(
        _demo(
            "adaptive_frontier_widen",
            "question_path_mismatch:r.alt",
            max_active=3,
        )
    )
    assert reserved.steps[1].rationale_facts == ("frontier_capacity_reservation",)
    assert counts["converted_to_reservation"] == 1


def test_repair_leaves_empty_and_semantic_mismatch_prunes_unchanged():
    empty = _demo("adaptive_frontier_widen", "question_path_mismatch:r.alt")
    empty.hypotheses["H1"] = ExecutedHypothesis(
        "H1", empty.hypotheses["H1"].function_state, "expression1", ()
    )
    repaired, counts = MODULE.repair_demonstration(empty)
    assert repaired.steps[1].rationale_facts == ("question_path_mismatch:r.alt",)
    assert counts["empty_adaptive_or_other_mismatch_unchanged"] == 1

    semantic, counts = MODULE.repair_demonstration(
        _demo("semantic_frontier_recovery", "question_path_mismatch:r.alt")
    )
    assert semantic.steps[1].rationale_facts == ("question_path_mismatch:r.alt",)
    assert counts["semantic_recovery_mismatch_unchanged"] == 1


def test_export_accepts_jsonl_gz_writes_sft_parquets_and_split_report(tmp_path):
    source = tmp_path / "demonstrations.jsonl.gz"
    with gzip.open(source, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(_demo("adaptive_frontier_widen", "question_path_mismatch:r.alt").to_dict()))
        handle.write("\n")
    output = tmp_path / "repaired"

    report = MODULE.export_repaired_corpus(source, output)

    assert report["converted_total"] == 1
    assert report["graph_execution_performed"] is False
    assert report["assertions"] == {
        "only_rationale_facts_changed": True,
        "actions_unchanged": True,
        "question_disjoint_train_validation": True,
    }
    for filename in (
        "demonstrations.jsonl",
        "train.parquet",
        "validation.parquet",
        "train_decision.parquet",
        "validation_decision.parquet",
        "repair_report.json",
    ):
        assert (output / filename).exists()

    messages = pd.read_parquet(
        output / "train_decision.parquet", columns=["messages"]
    )["messages"]
    assert {
        type(message["loss_mask"])
        for conversation in messages
        for message in conversation
    } == {int}


def test_export_carries_forward_audited_report_with_repair_record(tmp_path):
    source = tmp_path / "demonstrations.jsonl.gz"
    with gzip.open(source, "wt", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                _demo(
                    "adaptive_frontier_widen",
                    "question_path_mismatch:r.alt",
                ).to_dict()
            )
            + "\n"
        )
    source_report = tmp_path / "source_report.json"
    source_report.write_text(
        json.dumps(
            {
                "quality_assessment": {"structurally_ready_for_sft": True},
                "training_demonstrations": 999,
                "sft_rows": 999,
                "decision_sft_rows": 999,
                "validation_rows": 999,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "repaired"

    repair = MODULE.export_repaired_corpus(source, output, source_report)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))

    assert report["quality_assessment"]["structurally_ready_for_sft"] is True
    assert report["capacity_rationale_repair"] == repair
    assert report["training_demonstrations"] == repair["train_trajectory_rows"]
    assert report["sft_rows"] == repair["train_trajectory_rows"]
    assert report["decision_sft_rows"] == repair["train_decision_rows"]
    assert report["validation_rows"] == repair["validation_trajectory_rows"]
