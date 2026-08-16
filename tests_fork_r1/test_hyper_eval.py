import pytest

from scripts.analyze_hyper_r1_eval import summarize


def test_summarize_reports_grailqa_levels_and_execution_cost():
    rows = [
        {
            "mid_f1": 1.0,
            "hyper_r1_execution_calls": 4,
            "hyper_r1_commit_valid": 1,
            "hyper_r1_premature_answer": 0,
            "hyper_r1_branch_switch": 1,
            "hyper_r1_used_combine": 0,
            "hyper_r1_used_widen": 1,
            "hyper_r1_preserved_alternatives": 1,
            "hyper_r1_max_active": 3,
            "metadata": {"level": "i.i.d."},
        },
        {
            "mid_f1": 0.5,
            "hyper_r1_execution_attempts": 8,
            "hyper_r1_premature_answer": 1,
            "hyper_r1_used_widen": 0,
            "metadata": {"level": "compositional"},
        },
    ]
    report = summarize(rows)
    assert report["overall"]["mean_f1"] == pytest.approx(0.75)
    assert report["overall"]["exact_match"] == pytest.approx(0.5)
    assert report["overall"]["mean_execution_attempts"] == pytest.approx(6.0)
    assert report["overall"]["premature_answer_rate"] == pytest.approx(0.5)
    assert report["overall"]["commit_valid_rate"] == 1.0
    assert report["overall"]["branch_switch_rate"] == 1.0
    assert report["overall"]["branch_switch_f1"] == 1.0
    assert report["overall"]["mean_max_active_frontier"] == 3.0
    assert report["overall"]["widen_usage_rate"] == 0.5
    assert report["by_level"]["i.i.d."]["questions"] == 1
    assert report["by_level"]["compositional"]["mean_f1"] == pytest.approx(0.5)


def test_summarize_recomputes_f1_without_format_bonus():
    report = summarize(
        [
            {
                "output": "<answer>m.a m.b</answer>",
                "gts": {"target": ["m.a", "m.b"]},
                "score": 1.1,
            }
        ]
    )
    assert report["overall"]["mean_f1"] == 1.0
