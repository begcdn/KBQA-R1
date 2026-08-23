import importlib.util
from pathlib import Path

from kbqa_r1.hyper_data import (
    DemonstrationStep,
    ExecutedHypothesis,
    HyperDemonstration,
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
