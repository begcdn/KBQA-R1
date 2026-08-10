import pytest

from scripts.analyze_hyper_r1_eval import summarize


def test_summarize_reports_grailqa_levels_and_execution_cost():
    rows = [
        {
            "mid_f1": 1.0,
            "hyper_r1_execution_calls": 4,
            "metadata": {"level": "i.i.d."},
        },
        {
            "mid_f1": 0.5,
            "hyper_r1_execution_calls": 8,
            "metadata": {"level": "compositional"},
        },
    ]
    report = summarize(rows)
    assert report["overall"]["mean_f1"] == pytest.approx(0.75)
    assert report["overall"]["exact_match"] == pytest.approx(0.5)
    assert report["overall"]["mean_execution_calls"] == pytest.approx(6.0)
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
