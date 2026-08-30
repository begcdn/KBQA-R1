import json

import pytest

from scripts.summarize_hyper_gate import summarize


def test_gate_summary_separates_policy_and_fallback_f1(tmp_path):
    rows = [
        {
            "hyper_r1_commit_answer_f1": 1.0,
            "hyper_r1_explicit_model_commit": 1,
            "hyper_r1_forced_terminal": 0,
            "metadata": {"id": "q1", "family": "iid"},
        },
        {
            "hyper_r1_commit_answer_f1": 0.5,
            "hyper_r1_explicit_model_commit": 0,
            "hyper_r1_forced_terminal": 1,
            "metadata": {"id": "q2", "family": "iid"},
        },
    ]
    path = tmp_path / "progress.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    report = summarize(path, expected=2)

    assert report["overall"]["fallback_assisted_mean_f1"] == pytest.approx(0.75)
    assert report["overall"]["policy_only_mean_f1"] == pytest.approx(0.5)
    assert report["overall"]["explicit_commit_mean_f1"] == pytest.approx(1.0)
    assert report["overall"]["forced_terminal_mean_f1"] == pytest.approx(0.5)
