import importlib.util
import json
from dataclasses import replace
from pathlib import Path

from kbqa_r1.hyper_data import (
    DemonstrationStep,
    ExecutedHypothesis,
    HyperDemonstration,
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

    assert list(migrated.hypotheses) == ["H1", "H0"]
    assert MODULE._hypotheses_follow_runtime_order(migrated)


def test_abstain_example_requires_prior_execution_and_no_complete_hypothesis():
    source = demo(
        [
            DemonstrationStep("Inspect", ("P0",), (), ("H1",)),
            DemonstrationStep("Inspect", ("P1",), ("H1",), ("H2",)),
        ]
    )

    abstain = MODULE._budget_abstain_demo(source)

    assert abstain is not None
    assert abstain.steps[-1].action == "Abstain"
    assert abstain.private_metadata["max_execution_attempts"] == 1
    assert abstain.private_metadata["verified_abstain_reason"] == (
        "execution_budget_exhausted"
    )
    assert abstain.private_metadata["execution_attempts"] == 1
    assert set(abstain.hypotheses) == {"H1"}


def test_abstain_examples_span_later_exhaustion_states_instead_of_always_one_step():
    source = demo(
        [
            DemonstrationStep("Inspect", ("P0",), (), ("H0",)),
            DemonstrationStep("Inspect", ("P1",), ("H0",), ("H1",)),
            DemonstrationStep("Inspect", ("P2",), ("H0", "H1"), ("H2",)),
        ]
    )

    budgets = {
        MODULE._budget_abstain_demo(replace(source, demo_id=f"demo-{index}"))
        .private_metadata["max_execution_attempts"]
        for index in range(32)
    }

    assert budgets == {1, 2}


def test_abstain_is_not_taught_when_a_complete_parked_hypothesis_can_be_recalled():
    source = HyperDemonstration(
        demo_id="parked-gold",
        question_id="parked-gold",
        question="Which value is correct?",
        family="recovery",
        hypotheses={
            "H0": node("H0", ("m.answer",)),
            "H1": node("H1", ("m.other",)),
        },
        steps=[
            DemonstrationStep("Inspect", ("P0",), (), ("H0",)),
            DemonstrationStep("Park", ("H0",), ("H0",)),
            DemonstrationStep("Inspect", ("P1",), (), ("H1",)),
        ],
        gold_answers=("m.answer",),
        private_metadata={"max_turns": 12},
    )

    assert MODULE._budget_abstain_demo(source) is None

    invalid = replace(
        source,
        steps=[
            *source.steps[:2],
            DemonstrationStep("Abstain", (), ()),
        ],
        private_metadata={
            "max_turns": 3,
            "max_execution_attempts": 1,
            "execution_attempts": 1,
            "verified_abstain_reason": "execution_budget_exhausted",
        },
    )
    assert not MODULE._valid_budget_abstain(invalid)


def test_invalid_recovery_masks_the_bad_action_and_trains_only_valid_target():
    row = {
        "messages": [
            {"role": "user", "content": "Question"},
            {
                "role": "user",
                "content": "<information>\n<hypothesis_graph>state</hypothesis_graph>\n</information>",
                "loss_mask": 0,
            },
            {
                "role": "assistant",
                "content": "<action>Select [ H0 ]</action>",
                "loss_mask": 1,
            },
        ],
        "extra_info": {"question_id": "question"},
    }

    recovered = MODULE._invalid_recovery_record(row, 0)

    assistant = [
        message for message in recovered["messages"] if message["role"] == "assistant"
    ]
    assert assistant[0]["loss_mask"] == 0
    assert "H999999" in assistant[0]["content"]
    assert assistant[-1]["loss_mask"] == 1
    assert "H0" in assistant[-1]["content"]
    assert "Graph action failed" in recovered["messages"][-2]["content"]
    assert recovered["messages"][-2]["content"].splitlines()[2].startswith(
        "<hypothesis_graph>"
    )


def test_invalid_recovery_teaches_commit_of_current_node_after_stale_commit_failure():
    previous = """<information>
Selected H0. Further Find_relation actions now expand this hypothesis.
<hypothesis_graph>
active=1 capacity=24 stored=1 parked=0 execution_attempts=1/24 selected=H0 committed=none
H0 [active] parents=ROOT operation=expand via=r.items depth=0 path=r.items answers=2: m.one, m.two
Available targets: Select=[H0]; Park=[H0]; Commit(nonempty active)=[H0]; Combine=[none]; Prune candidates=[none]; Recall=[none]; Find_relation sources=[expression1].
</hypothesis_graph>
</information>"""
    current = """<information>
Executed the selected hypothesis operation.
<hypothesis_graph>
active=1 capacity=24 stored=2 parked=0 execution_attempts=2/24 selected=none committed=none
H1 [active] parents=H0 operation=count via=count depth=1 path=r.items answers=1: 2
Available targets: Select=[H1]; Park=[H1]; Commit(nonempty active)=[H1]; Combine=[none]; Prune candidates=[none]; Recall=[none]; Find_relation sources=[m.topic].
</hypothesis_graph>
</information>"""
    row = {
        "messages": [
            {"role": "user", "content": "Question"},
            {"role": "user", "content": previous, "loss_mask": 0},
            {
                "role": "assistant",
                "content": "<action>Count [ expression1 ]</action>",
                "loss_mask": 1,
            },
            {"role": "user", "content": current, "loss_mask": 0},
            {
                "role": "assistant",
                "content": "<action>Commit [ H1 ]</action>",
                "loss_mask": 1,
            },
        ],
        "extra_info": {"question_id": "question"},
    }

    recovered = MODULE._invalid_recovery_record(row, 0)

    assert "<action>Commit [ H0 ]</action>" in recovered["messages"][-3]["content"]
    assert recovered["messages"][-3]["loss_mask"] == 0
    assert recovered["extra_info"]["invalid_recovery_kind"] == "stale_commit"
    assert recovered["messages"][-2]["content"].startswith(
        "<information>\nGraph action failed: hypothesis H0 is expanded, not active\n"
        "<hypothesis_graph>"
    )
    assert "Executed the selected hypothesis operation." not in (
        recovered["messages"][-2]["content"]
    )
    assert recovered["messages"][-1]["content"] == "<action>Commit [ H1 ]</action>"
    assert recovered["messages"][-1]["loss_mask"] == 1
    assert MODULE._valid_invalid_recovery(recovered)


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
