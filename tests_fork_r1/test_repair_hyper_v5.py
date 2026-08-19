import gzip
import importlib.util
import json
from pathlib import Path

import pytest

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


def test_export_rejects_uncertified_v5_corpus(tmp_path):
    source = tmp_path / "demonstrations.jsonl.gz"
    with gzip.open(source, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(_demo("adaptive_frontier_widen", "question_path_mismatch:r.alt").to_dict()))
        handle.write("\n")
    output = tmp_path / "repaired"

    with pytest.raises(RuntimeError, match="cannot be made proof-carrying"):
        MODULE.export_repaired_corpus(source, output)
    assert not output.exists()


def test_uncertified_v5_export_cannot_preserve_stale_ready_status(tmp_path):
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

    with pytest.raises(RuntimeError, match="rebuild the corpus"):
        MODULE.export_repaired_corpus(source, output, source_report)
    assert not (output / "report.json").exists()
