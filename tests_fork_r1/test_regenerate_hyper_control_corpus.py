import importlib.util
import json
from dataclasses import replace
from pathlib import Path

from kbqa_r1.hyper_data import (
    DemonstrationStep,
    ExecutedHypothesis,
    HyperDemonstration,
    RelationProposal,
    decision_sft_records,
    trajectory_sft_record,
)


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "data_process"
    / "regenerate_hyper_control_corpus.py"
)
SPEC = importlib.util.spec_from_file_location("regenerate_hyper_control_corpus", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def node(node_id, answers=("m.value",)):
    return ExecutedHypothesis(
        hypothesis_id=node_id,
        function_state=(f"expression1 = START('m.topic')",),
        target_expression="expression1",
        denotation=answers,
    )


def demo(steps):
    return HyperDemonstration(
        demo_id="demo",
        question_id="question",
        question="Which value is correct?",
        family="recovery",
        hypotheses={
            "H0": node("H0"),
            "H1": node("H1", ("m.sibling",)),
            "H2": node("H2", ()),
        },
        steps=steps,
        gold_answers=("m.answer",),
        private_metadata={"max_turns": 12},
    )


def test_recall_augmentation_parks_and_restores_the_preserved_branch():
    source = demo(
        [
            DemonstrationStep(
                "Select", ("H1",), ("H0", "H1"), supervision="intervention"
            ),
            DemonstrationStep("Find_relation", ("expression1",), ("H0", "H1")),
            DemonstrationStep("Prune", ("H2",), ("H0", "H2")),
            DemonstrationStep("Select", ("H0",), ("H0",)),
        ]
    )

    updated, changed = MODULE._augment_recall(source)

    assert changed
    assert [step.action for step in updated.steps] == [
        "Park",
        "Select",
        "Find_relation",
        "Prune",
        "Recall",
        "Select",
    ]
    assert updated.steps[0].arguments == ("H0",)
    assert updated.steps[1].visible_before == ("H1",)
    assert updated.steps[4].visible_before == ()
    assert updated.steps[5].visible_before == ("H0",)


def test_recalled_training_affordances_follow_runtime_creation_order():
    function_state = ("expression1 = START('m.topic')",)
    source = HyperDemonstration(
        demo_id="recall-order",
        question_id="recall-order",
        question="Which value is correct?",
        family="recovery",
        hypotheses={
            "H0": ExecutedHypothesis(
                "H0", function_state, "expression1", ("m.answer",)
            ),
            "H1": ExecutedHypothesis(
                "H1", function_state + ("expression1 = JOIN('r.other', expression1)",),
                "expression1", ("m.other",)
            ),
        },
        steps=[
            DemonstrationStep("Park", ("H0",), ("H0", "H1")),
            DemonstrationStep("Recall", ("H0",), ("H1",)),
            DemonstrationStep(
                "Commit",
                ("H0",),
                ("H0", "H1"),
                certificate_kind="answer_and_supported_query_equivalent",
            ),
        ],
        gold_answers=("m.answer",),
        private_metadata={
            "runtime_protocol": "lazy_relation_inspection_v1",
            "gold_program": function_state,
            "gold_target_expression": "expression1",
            "max_active": 24,
            "max_nodes": 128,
            "max_execution_attempts": 24,
        },
    )

    record = trajectory_sft_record(source)
    recalled = next(
        message["content"]
        for message in record["messages"]
        if message["role"] == "user" and "Recalled H0" in message["content"]
    )

    assert "Select=[H0,H1]" in recalled
    assert "Park=[H0,H1]" in recalled
    assert "Commit(nonempty active)=[H0,H1]" in recalled
    assert "Combine=[H0|H1]" in recalled


def test_decision_records_keep_question_latest_state_and_target_only():
    source = demo(
        [
            DemonstrationStep("Select", ("H0",), ("H0", "H1")),
            DemonstrationStep(
                "Commit",
                ("H0",),
                ("H0", "H1"),
                certificate_kind="best_attainable_answer_f1",
                certificate_evidence=("answer_f1:0.00000000",),
            ),
        ]
    )
    source = replace(
        source,
        private_metadata={
            **source.private_metadata,
            "best_attainable_supervision": True,
        },
    )

    rows = decision_sft_records(source)

    assert len(rows[0]["messages"]) == 4
    assert len(rows[1]["messages"]) == 4
    assert "Which value is correct?" in rows[1]["messages"][0]["content"]
    assert rows[1]["messages"][1] == {
        "role": "assistant",
        "content": "",
        "loss_mask": 0,
    }
    assert "Selected H0" in rows[1]["messages"][2]["content"]
    assert "<action>Commit [ H0 ]</action>" in rows[1]["messages"][3]["content"]


def test_saved_hypotheses_are_migrated_to_runtime_creation_order():
    source = HyperDemonstration(
        demo_id="inverted-inspection",
        question_id="inverted-inspection",
        question="Which value is correct?",
        family="recovery",
        hypotheses={
            "H0": node("H0"),
            "H1": node("H1", ("m.other",)),
        },
        steps=[
            DemonstrationStep("Inspect", ("P1",), (), ("H1",)),
            DemonstrationStep("Inspect", ("P0",), ("H1",), ("H0",)),
            DemonstrationStep("Commit", ("H0",), ("H1", "H0")),
        ],
        gold_answers=("m.answer",),
    )

    migrated = MODULE._runtime_ordered_demo(source)

    assert list(migrated.hypotheses) == ["H0", "H1"]
    assert migrated.hypotheses["H0"].denotation == ("m.other",)
    assert migrated.hypotheses["H1"].denotation == ("m.value",)
    assert migrated.steps[0].created == ("H0",)
    assert migrated.steps[1].visible_before == ("H0",)
    assert migrated.steps[1].created == ("H1",)
    assert migrated.steps[-1].arguments == ("H1",)
    assert MODULE._hypotheses_follow_runtime_order(migrated)


def test_rendered_runtime_identity_rejects_sparse_node_ids():
    row = {
        "messages": [
            {
                "role": "user",
                "content": """<information>
<hypothesis_graph>
active=1 capacity=24 stored=2 parked=0 execution_attempts=1/24 turns_used=2/32 turns_remaining=30 selected=none committed=none
H6 [active] parents=ROOT operation=expand via=r depth=0 path=r answers=1: m.x
</hypothesis_graph>
</information>""",
            }
        ]
    }

    assert not MODULE._rendered_runtime_identity_is_valid(row)
    row["messages"][0]["content"] = row["messages"][0]["content"].replace(
        "H6", "H1"
    )
    assert MODULE._rendered_runtime_identity_is_valid(row)


def test_deadline_variant_uses_only_available_executed_candidates():
    source = HyperDemonstration(
        demo_id="placeholder",
        question_id="deadline",
        question="Which value is correct?",
        family="deadline",
        hypotheses={
            "H0": node("H0", ("m.partial",)),
            "H1": node("H1", ("m.answer",)),
        },
        steps=[
            DemonstrationStep("Inspect", ("P0",), (), ("H0",)),
            DemonstrationStep("Inspect", ("P1",), ("H0",), ("H1",)),
            DemonstrationStep("Select", ("H1",), ("H0", "H1")),
            DemonstrationStep("Commit", ("H1",), ("H0", "H1")),
        ],
        gold_answers=("m.answer",),
        private_metadata={"max_turns": 32},
    )
    variant = None
    for index in range(10_000):
        candidate = replace(source, demo_id=f"deadline-{index}")
        variant = MODULE._deadline_variant(candidate)
        if variant is not None:
            break

    assert variant is not None
    assert variant.private_metadata["deadline_cutoff_sampled_without_gold"]
    assert variant.steps[-1].action == "Commit"
    assert variant.steps[-1].arguments[0] in variant.steps[-1].visible_before
    assert set(variant.hypotheses).issuperset(variant.steps[-1].visible_before)
    assert variant.private_metadata["max_turns"] == 32
    assert variant.private_metadata["turn_offset"] + len(variant.steps) == 32


def test_deadline_variant_recalls_the_best_parked_candidate_before_commit():
    source = HyperDemonstration(
        demo_id="placeholder",
        question_id="deadline-parked",
        question="Which value is correct?",
        family="deadline",
        hypotheses={
            "H0": node("H0", ("m.partial",)),
            "H1": node("H1", ("m.answer",)),
        },
        steps=[
            DemonstrationStep("Inspect", ("P0",), (), ("H0",)),
            DemonstrationStep("Inspect", ("P1",), ("H0",), ("H1",)),
            DemonstrationStep("Park", ("H1",), ("H0", "H1")),
            DemonstrationStep("Select", ("H0",), ("H0",)),
            DemonstrationStep("Recall", ("H1",), ("H0",)),
            DemonstrationStep("Commit", ("H1",), ("H0", "H1")),
        ],
        gold_answers=("m.answer",),
        private_metadata={"max_turns": 32},
    )
    variant = None
    for index in range(100_000):
        candidate = replace(source, demo_id=f"deadline-parked-{index}")
        possible = MODULE._deadline_variant(candidate)
        if possible is not None and possible.private_metadata.get(
            "deadline_recalled_best"
        ):
            variant = possible
            break

    assert variant is not None
    assert [step.action for step in variant.steps[-2:]] == ["Recall", "Commit"]
    assert variant.steps[-2].arguments == ("H1",)
    assert variant.steps[-1].arguments == ("H1",)
    assert variant.private_metadata["max_turns"] == 32
    assert variant.private_metadata["turn_offset"] + len(variant.steps) == 32


def test_delayed_decoy_comparison_makes_the_certified_target_older():
    start = "expression1 = START('m.topic')"
    gold = ExecutedHypothesis(
        "H1",
        (start, "expression1 = JOIN('r.gold', expression1)"),
        "expression1",
        ("m.answer",),
        relation="r.gold",
    )
    decoy = ExecutedHypothesis(
        "H0",
        (start, "expression1 = JOIN('r.decoy', expression1)"),
        "expression1",
        ("m.decoy",),
        relation="r.decoy",
    )
    source = HyperDemonstration(
        demo_id="comparison",
        question_id="comparison",
        question="Which value is correct?",
        family="frontier_commit",
        hypotheses={"H0": decoy, "H1": gold},
        proposals={
            "P0": RelationProposal("P0", "F0", "m.topic", "r.decoy", 0.8, 0),
            "P1": RelationProposal("P1", "F0", "m.topic", "r.gold", 0.7, 1),
        },
        steps=[
            DemonstrationStep(
                "Find_relation",
                ("m.topic",),
                (),
                exposed=("P0", "P1"),
                relation_page=(0, 2, 2),
            ),
            DemonstrationStep("Inspect", ("P0",), (), ("H0",)),
            DemonstrationStep("Inspect", ("P1",), ("H0",), ("H1",)),
            DemonstrationStep(
                "Commit",
                ("H1",),
                ("H0", "H1"),
                certificate_kind="answer_and_supported_query_equivalent",
            ),
        ],
        gold_answers=("m.answer",),
        private_metadata={
            "runtime_protocol": "lazy_relation_inspection_v1",
            "gold_program": gold.function_state,
            "gold_target_expression": "expression1",
            "max_turns": 32,
            "max_active": 24,
            "max_nodes": 128,
            "max_execution_attempts": 24,
        },
    )

    variant = MODULE._delayed_decoy_comparison(source)

    assert variant is not None
    assert [step.action for step in variant.steps] == [
        "Find_relation",
        "Inspect",
        "Inspect",
        "Commit",
    ]
    assert variant.steps[1].arguments == ("P1",)
    assert variant.steps[2].arguments == ("P0",)
    target = variant.steps[-1].arguments[0]
    decoy_id = variant.private_metadata["comparison_decoy"]
    assert int(target[1:]) < int(decoy_id[1:])
    assert variant.hypotheses[target].denotation == ("m.answer",)
    assert variant.private_metadata["terminal_decision_start"] == 3
    assert not MODULE._runtime_replay_errors(variant)


def test_invalid_recovery_masks_the_bad_action_and_trains_only_valid_target():
    state = """<hypothesis_graph>
active=1 capacity=24 stored=1 parked=0 execution_attempts=1/24 turns_used=3/32 turns_remaining=29 selected=H0 committed=none
H0 [active] parents=ROOT operation=expand via=r depth=0 path=r answers=1: m.x
Available targets: Select=[H0]; Park=[H0]; Commit(nonempty active)=[H0]; Combine=[none]; Prune candidates=[none]; Recall=[none]; Find_relation sources=[m.topic].
</hypothesis_graph>"""
    row = {
        "messages": [
            {"role": "user", "content": "Question"},
            {
                "role": "user",
                "content": f"<information>\nSelected H0.\n{state}\n</information>",
                "loss_mask": 0,
            },
            {
                "role": "assistant",
                "content": "<action>Park [ H0 ]</action>",
                "loss_mask": 1,
            },
        ],
        "extra_info": {"question_id": "question"},
    }

    spec = next(
        spec for spec in MODULE._invalid_recovery_specs(row)
        if spec[0] == "repeat_select"
    )
    recovered = MODULE._invalid_recovery_record(row, spec, 0)

    assert len(recovered["messages"]) == 4
    assert recovered["messages"][1] == {
        "role": "assistant",
        "content": "",
        "loss_mask": 0,
    }
    assert recovered["messages"][-1]["loss_mask"] == 1
    assert "Park [ H0 ]" in recovered["messages"][-1]["content"]
    assert "Select [ H0 ]" not in json.dumps(recovered["messages"])
    assert "Graph action failed" in recovered["messages"][2]["content"]
    assert recovered["messages"][2]["content"].splitlines()[2].startswith(
        "<hypothesis_graph>"
    )
    assert recovered["extra_info"]["invalid_action"] == "Select [ H0 ]"
    assert MODULE._valid_invalid_recovery(recovered)


def test_invalid_recovery_specs_use_only_ids_visible_in_the_current_state():
    current = """<information>
Executed the selected hypothesis operation.
<hypothesis_graph>
active=1 capacity=24 stored=2 parked=0 execution_attempts=2/24 turns_used=4/32 turns_remaining=28 selected=none committed=none
H1 [active] parents=H0 operation=count via=count depth=1 path=r.items answers=1: 2
Available targets: Select=[H1]; Park=[H1]; Commit(nonempty active)=[H1]; Combine=[none]; Prune candidates=[none]; Recall=[none]; Find_relation sources=[m.topic].
</hypothesis_graph>
</information>"""
    row = {
        "messages": [
            {"role": "user", "content": "Question"},
            {"role": "user", "content": current, "loss_mask": 0},
            {
                "role": "assistant",
                "content": "<action>Commit [ H1 ]</action>",
                "loss_mask": 1,
            },
        ],
        "extra_info": {"question_id": "question"},
    }

    specs = MODULE._invalid_recovery_specs(row)
    assert specs
    assert all("H0" not in invalid_action for _, invalid_action, _ in specs)
    assert all("999999" not in invalid_action for _, invalid_action, _ in specs)


def test_invalid_recovery_rejects_an_initial_prompt_without_environment_feedback():
    row = {
        "messages": [
            {"role": "user", "content": "Question"},
            {
                "role": "assistant",
                "content": "<action>Find_relation [ m.topic ]</action>",
                "loss_mask": 1,
            },
        ],
        "extra_info": {"question_id": "question"},
    }

    assert not MODULE._supports_invalid_recovery(row)


def test_streaming_parquet_sink_compresses_and_publishes_atomically(tmp_path):
    import pyarrow.parquet as pq

    row = {
        "messages": [
            {"role": "user", "content": "Question", "loss_mask": 0},
            {
                "role": "assistant",
                "content": "<action>Select [ H0 ]</action>",
                "loss_mask": 1,
            },
        ],
        "data_source": "hyper_r1_verified_decision",
        "extra_info": {
            "demo_id": "demo",
            "question_id": "question",
            "family": "recovery",
            "recovery_stratum": None,
            "replay_verified": True,
            "gold_injected_into_proposals": False,
            "terminal_action": "Commit",
            "decision_index": 0,
            "trajectory_step_index": 0,
            "target_is_graph_action": True,
        },
    }
    path = tmp_path / "decisions.parquet"
    sink = MODULE._ParquetSink(path, batch_size=1)
    sink.append(row)
    assert not path.exists()

    sink.close()

    parquet = pq.ParquetFile(path)
    assert parquet.metadata.num_rows == 1
    assert parquet.metadata.row_group(0).column(1).compression == "SNAPPY"
    assert not path.with_suffix(".parquet.tmp").exists()


def test_source_contract_is_embedded_for_truthful_control_training(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    demonstrations = source / "demonstrations.jsonl"
    demonstrations.write_text("", encoding="utf-8")
    report = {
        "quality_schema": "hyper_r1_v13_state_covered_recovery",
        "relation_page_size": 6,
        "relation_rank_cutoff": None,
        "max_active": 24,
        "max_nodes": 128,
        "max_execution_attempts": 24,
        "max_turns": 32,
        "quality_assessment": {"structurally_ready_for_sft": True},
        "proposal_recall": {"relation_at_frontier": 0.9},
        "families": {"frontier_commit": 1},
    }
    (source / "report.json").write_text(json.dumps(report), encoding="utf-8")

    contract = MODULE._load_source_contract(demonstrations)

    assert contract["quality_schema"] == report["quality_schema"]
    assert contract["max_turns"] == 32
    assert contract["quality_assessment"]["structurally_ready_for_sft"]
    assert len(contract["report_sha256"]) == 64
