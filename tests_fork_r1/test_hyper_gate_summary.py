import json

import pytest

from scripts.summarize_hyper_gate import row_search_diagnostics, summarize


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


def _snapshot(*nodes):
    return {
        "graph": {
            "state": {
                "nodes": list(nodes),
            }
        }
    }


def _node(node_id, answers, parent_id=None):
    return {
        "node_id": node_id,
        "denotation": answers,
        "parent_id": parent_id,
        "parent_ids": [],
    }


def test_search_diagnostics_detect_answer_abandonment_and_regression():
    wrong = _node("H0", ["wrong"])
    exact = _node("H1", ["gold"])
    regressed = _node("H2", ["other"], parent_id="H1")
    row = {
        "gts": {"target": ["gold"]},
        "hyper_r1_commit_answer_f1": 0.0,
        "decisions": [
            {
                "turn": 1,
                "accepted": True,
                "policy_action": "Inspect [ P1 ]",
                "private_execution_state": _snapshot(wrong),
            },
            {
                "turn": 2,
                "accepted": True,
                "policy_action": "Park [ H1 ]",
                "private_execution_state": _snapshot(wrong, exact),
            },
            {
                "turn": 3,
                "accepted": True,
                "policy_action": "Recall [ H1 ]",
                "private_execution_state": _snapshot(wrong, exact),
            },
            {
                "turn": 4,
                "accepted": True,
                "policy_action": "Inspect [ P2 ]",
                "private_execution_state": _snapshot(wrong, exact),
            },
            {
                "turn": 5,
                "accepted": True,
                "policy_action": "Commit [ H2 ]",
                "private_execution_state": _snapshot(wrong, exact, regressed),
            },
        ],
    }

    metrics = row_search_diagnostics(row)

    assert metrics["best_seen_answer_f1"] == 1.0
    assert metrics["commit_regret"] == 1.0
    assert metrics["exact_answer_seen"] == 1.0
    assert metrics["exact_answer_abandoned"] == 1.0
    assert metrics["nonterminal_actions_after_first_exact"] == 3.0
    assert metrics["exact_hypothesis_parks"] == 1.0
    assert metrics["exact_hypothesis_recalls"] == 1.0
    assert metrics["exact_answer_regression_edges"] == 1.0
