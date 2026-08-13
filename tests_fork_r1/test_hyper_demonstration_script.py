import importlib.util
from pathlib import Path

from kbqa_r1.hyper_data import HyperDemonstration


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
