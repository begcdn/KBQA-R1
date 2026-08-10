import pytest

from scripts.compare_hyper_r1_eval import compare, paired_bootstrap


def row(qid, f1, level="i.i.d.", calls=None):
    result = {"metadata": {"id": qid, "level": level}, "mid_f1": f1}
    if calls is not None:
        result["hyper_r1_execution_calls"] = calls
    return result


def test_compare_is_paired_and_reports_generalization_slices():
    baseline = [row("q1", 0.0, calls=2), row("q2", 1.0, "zero-shot", 4)]
    method = [row("q2", 1.0, "zero-shot", 8), row("q1", 1.0, calls=4)]
    report = compare(baseline, method)
    assert report["overall"]["delta_f1"] == pytest.approx(0.5)
    assert report["overall"]["method_wins"] == 1
    assert report["overall"]["ties"] == 1
    assert report["overall"]["method_execution_calls"] == pytest.approx(6.0)
    assert report["by_level"]["zero-shot"]["questions"] == 1


def test_compare_rejects_different_question_populations():
    with pytest.raises(ValueError, match="identical question populations"):
        compare([row("q1", 0.0)], [row("q2", 1.0)])


def test_paired_bootstrap_is_deterministic():
    assert paired_bootstrap([1.0, 0.0, -1.0], samples=100, seed=3) == paired_bootstrap(
        [1.0, 0.0, -1.0], samples=100, seed=3
    )
