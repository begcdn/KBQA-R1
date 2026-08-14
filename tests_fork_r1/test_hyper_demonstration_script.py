import importlib.util
from pathlib import Path
import threading

from kbqa_r1.hyper_data import (
    DemonstrationStep,
    ExecutedHypothesis,
    HyperDemonstration,
)


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "data_process"
    / "build_hyper_demonstrations.py"
)
SPEC = importlib.util.spec_from_file_location("build_hyper_demonstrations", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def demo(name, family):
    return HyperDemonstration(name, name, name, family, {}, [], ())


def test_curriculum_caps_repetitive_commits_but_keeps_all_core_behaviors():
    recoveries = [demo(f"r{index}", "delayed_frontier_recovery") for index in range(2)]
    progress = [demo("p0", "direct_frontier_progress")]
    conjunctions = [demo("c0", "conjunction")]
    commits = [demo(f"d{index}", "frontier_commit") for index in range(20)]

    selected = MODULE._curriculum_take(
        [*commits, *recoveries, *progress, *conjunctions],
        limit=0,
    )

    assert {item.demo_id for item in selected if item.family != "frontier_commit"} == {
        "r0",
        "r1",
        "p0",
        "c0",
    }
    assert sum(item.family == "frontier_commit" for item in selected) == 4


def test_explicit_limit_is_round_robin_across_available_families():
    selected = MODULE._curriculum_take(
        [
            demo("d0", "frontier_commit"),
            demo("d1", "frontier_commit"),
            demo("p0", "direct_frontier_progress"),
            demo("p1", "direct_frontier_progress"),
        ],
        limit=2,
    )

    assert {item.family for item in selected} == {
        "frontier_commit",
        "direct_frontier_progress",
    }


def test_question_split_keeps_all_trajectories_for_a_question_together():
    assert MODULE._question_split("question-1") == MODULE._question_split("question-1")
    assert {
        MODULE._question_split(f"question-{index}") for index in range(100)
    } == {"train", "validation"}


def test_decision_consistency_detects_conflicting_actions_for_same_state():
    shared = [{"role": "user", "content": "same state"}]
    rows = [
        {"messages": [*shared, {"role": "assistant", "content": "<action>Select [ H0 ]</action>"}]},
        {"messages": [*shared, {"role": "assistant", "content": "<action>Select [ H1 ]</action>"}]},
    ]

    result = MODULE._decision_contradictions(rows)

    assert result["decision_states"] == 1
    assert result["contradictory_states"] == 1


def test_unnamed_entity_uses_specific_type_without_claiming_a_name():
    assert MODULE._readable_type_descriptor(
        ["common.topic", "food.cheese"]
    ) == "unnamed cheese"
    assert MODULE._readable_type_descriptor(
        ["common.topic", "base.type_ontology.inanimate"]
    ) is None


def test_entity_display_provider_prefers_real_name_over_type_descriptor():
    provider = object.__new__(MODULE.LiveEntityDisplayProvider)
    provider._cache = {}
    provider._descriptor_ids = set()
    provider._lock = threading.RLock()

    provider.remember([{"x": "m.entity", "type": "food.cheese"}])
    assert provider._cache["m.entity"] == "unnamed cheese"
    assert provider.is_type_descriptor("m.entity")

    provider.remember([{"x": "m.entity", "name": "Brie", "type": "food.cheese"}])
    assert provider._cache["m.entity"] == "Brie"
    assert not provider.is_type_descriptor("m.entity")


def test_frontier_diagnostics_inspect_continuation_frontiers():
    nodes = {
        "H0": ExecutedHypothesis("H0", (), "expression1", ("m.prefix",)),
        "H1": ExecutedHypothesis(
            "H1", (), "expression2", ("m.answer",), parent_id="H0", depth=1
        ),
        "H2": ExecutedHypothesis(
            "H2", (), "expression2", ("m.answer",), parent_id="H0", depth=1,
            role="alternative",
        ),
    }
    demonstration = HyperDemonstration(
        "continuation",
        "question",
        "question",
        "direct_frontier_progress",
        nodes,
        [
            DemonstrationStep("Find_relation", ("m.topic",), (), ("H0",)),
            DemonstrationStep("Select", ("H0",), ("H0",)),
            DemonstrationStep("Find_relation", ("expression1",), ("H0",), ("H1", "H2")),
            DemonstrationStep("Commit", ("H1",), ("H1", "H2")),
        ],
        ("m.answer",),
    )

    result = MODULE._frontier_diagnostics([demonstration])

    assert result["frontiers_with_multiple_gold_answer_sets"] == 1
    assert result["commits_with_answer_equivalent_sibling"] == 1
