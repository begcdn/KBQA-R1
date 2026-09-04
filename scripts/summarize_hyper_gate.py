#!/usr/bin/env python3
"""Summarize durable HyPER-R1 executable-gate records."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping


METRICS = (
    "reward",
    "score",
    "mid_f1",
    "hyper_r1_commit_valid",
    "hyper_r1_commit_protocol_valid",
    "hyper_r1_commit_answer_f1",
    "hyper_r1_commit_intent_equivalent",
    "hyper_r1_premature_answer",
    "hyper_r1_abstained",
    "hyper_r1_explicit_model_commit",
    "hyper_r1_forced_terminal",
    "hyper_r1_forced_empty",
    "hyper_r1_turn_exhausted",
    "hyper_r1_preserved_alternatives",
    "hyper_r1_branch_switch",
    "hyper_r1_used_widen",
    "hyper_r1_used_combine",
    "hyper_r1_execution_attempts",
    "hyper_r1_max_active",
)

SEARCH_DIAGNOSTICS = (
    "best_seen_answer_f1",
    "commit_regret",
    "exact_answer_seen",
    "exact_answer_abandoned",
    "nonterminal_actions_after_first_exact",
    "exact_hypothesis_parks",
    "exact_hypothesis_recalls",
    "exact_answer_regression_edges",
)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(row.get("metadata", {}).get("id", "")) for row in rows]
    duplicates = [item for item, count in Counter(ids).items() if item and count > 1]
    if duplicates:
        raise ValueError(f"duplicate completed question IDs: {duplicates[:10]}")
    return rows


def _metric_means(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    materialized = list(rows)
    result = {}
    for key in METRICS:
        values = [float(row[key]) for row in materialized if row.get(key) is not None]
        if values:
            result[key] = mean(values)
    if any("hyper_r1_explicit_model_commit" in row for row in materialized):
        f1_values = [
            float(row.get("hyper_r1_commit_answer_f1", 0.0))
            for row in materialized
        ]
        explicit = [
            row
            for row in materialized
            if float(row.get("hyper_r1_explicit_model_commit", 0.0)) > 0
        ]
        forced = [
            row
            for row in materialized
            if float(row.get("hyper_r1_forced_terminal", 0.0)) > 0
        ]
        result["fallback_assisted_mean_f1"] = mean(f1_values)
        result["policy_only_mean_f1"] = mean(
            float(row.get("hyper_r1_commit_answer_f1", 0.0))
            if float(row.get("hyper_r1_explicit_model_commit", 0.0)) > 0
            else 0.0
            for row in materialized
        )
        if explicit:
            result["explicit_commit_mean_f1"] = mean(
                float(row.get("hyper_r1_commit_answer_f1", 0.0))
                for row in explicit
            )
        if forced:
            result["forced_terminal_mean_f1"] = mean(
                float(row.get("hyper_r1_commit_answer_f1", 0.0))
                for row in forced
            )
    return result


def _target_values(row: Mapping[str, Any]) -> set[str]:
    ground_truth = row.get("gts", {})
    target = ground_truth.get("target", ()) if isinstance(ground_truth, Mapping) else ()
    if isinstance(target, str):
        target = [target]
    return {str(value) for value in target}


def _answer_set_f1(predicted: Iterable[Any], gold: Iterable[Any]) -> float:
    predicted_values = {str(value) for value in predicted}
    gold_values = {str(value) for value in gold}
    if not predicted_values and not gold_values:
        return 1.0
    if not predicted_values or not gold_values:
        return 0.0
    overlap = len(predicted_values & gold_values)
    if not overlap:
        return 0.0
    precision = overlap / len(predicted_values)
    recall = overlap / len(gold_values)
    return 2.0 * precision * recall / (precision + recall)


def _snapshot_nodes(decision: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    snapshot = decision.get("private_execution_state")
    if not isinstance(snapshot, Mapping):
        return []
    graph = snapshot.get("graph")
    if not isinstance(graph, Mapping):
        return []
    state = graph.get("state")
    if not isinstance(state, Mapping):
        return []
    nodes = state.get("nodes")
    if not isinstance(nodes, list):
        return []
    return [node for node in nodes if isinstance(node, Mapping)]


def row_search_diagnostics(row: Mapping[str, Any]) -> dict[str, float]:
    """Measure search regret from private, evaluator-only graph snapshots."""
    gold = _target_values(row)
    decisions = sorted(
        (
            decision
            for decision in row.get("decisions", ())
            if isinstance(decision, Mapping)
        ),
        key=lambda decision: int(decision.get("turn", -1)),
    )
    node_f1: dict[str, float] = {}
    node_parent_ids: dict[str, tuple[str, ...]] = {}
    first_exact_turn: int | None = None
    exact_parks = 0
    exact_recalls = 0

    for decision in decisions:
        turn = int(decision.get("turn", -1))
        for node in _snapshot_nodes(decision):
            node_id = str(node.get("node_id") or "")
            if not node_id:
                continue
            score = _answer_set_f1(node.get("denotation", ()), gold)
            node_f1[node_id] = score
            parents = node.get("parent_ids") or ()
            if isinstance(parents, str):
                parents = [parents]
            parent_id = str(node.get("parent_id") or "")
            normalized_parents = tuple(str(value) for value in parents if value)
            if not normalized_parents and parent_id:
                normalized_parents = (parent_id,)
            node_parent_ids[node_id] = normalized_parents
            if score >= 1.0 - 1e-9 and first_exact_turn is None:
                first_exact_turn = turn

        if not decision.get("accepted"):
            continue
        action = str(decision.get("policy_action") or "")
        match = re.fullmatch(r"\s*(Park|Recall)\s*\[\s*(H\d+)\s*\]\s*", action)
        if match and node_f1.get(match.group(2), 0.0) >= 1.0 - 1e-9:
            if match.group(1) == "Park":
                exact_parks += 1
            else:
                exact_recalls += 1

    best_seen = max(node_f1.values(), default=0.0)
    commit_f1 = float(row.get("hyper_r1_commit_answer_f1", 0.0))
    exact_seen = best_seen >= 1.0 - 1e-9
    nonterminal_after_exact = 0
    if first_exact_turn is not None:
        nonterminal_after_exact = sum(
            1
            for decision in decisions
            if int(decision.get("turn", -1)) >= first_exact_turn
            and decision.get("accepted")
            and not str(decision.get("policy_action") or "").startswith("Commit [")
        )
    regression_edges = sum(
        1
        for node_id, parents in node_parent_ids.items()
        if node_f1.get(node_id, 0.0) < 1.0 - 1e-9
        and any(node_f1.get(parent, 0.0) >= 1.0 - 1e-9 for parent in parents)
    )
    return {
        "best_seen_answer_f1": best_seen,
        "commit_regret": max(0.0, best_seen - commit_f1),
        "exact_answer_seen": float(exact_seen),
        "exact_answer_abandoned": float(exact_seen and commit_f1 < 1.0 - 1e-9),
        "nonterminal_actions_after_first_exact": float(nonterminal_after_exact),
        "exact_hypothesis_parks": float(exact_parks),
        "exact_hypothesis_recalls": float(exact_recalls),
        "exact_answer_regression_edges": float(regression_edges),
    }


def _search_diagnostic_means(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    materialized = list(rows)
    diagnostics = [row_search_diagnostics(row) for row in materialized]
    result = {
        key: mean(values[key] for values in diagnostics)
        for key in SEARCH_DIAGNOSTICS
        if diagnostics
    }
    exact_rows = [
        values for values in diagnostics if values["exact_answer_seen"] > 0.0
    ]
    if exact_rows:
        result["actions_after_first_exact_when_seen"] = mean(
            values["nonterminal_actions_after_first_exact"] for values in exact_rows
        )
    return result


def _trajectory_diagnostics(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    materialized = list(rows)
    commit_samples = 0
    answer_exact_commits = 0
    commit_attempt_samples = 0
    action_failures = 0
    stale_expanded_failures = 0
    for row in materialized:
        output = str(row.get("output", ""))
        commit_attempt_samples += int(
            re.search(r"<action>\s*Commit\s*\[", output, re.I) is not None
        )
        commits = re.findall(
            r"Committed H\d+\. (?:Return exactly these values in <answer>|Final answer values):\s*([^\n<]+)",
            output,
        )
        if commits:
            commit_samples += 1
            answer_exact_commits += int(set(commits[-1].split()) == _target_values(row))
        action_failures += output.count("Graph action failed:")
        stale_expanded_failures += output.count("is expanded, not active")
    denominator = len(materialized) or 1
    return {
        "samples_with_commit_attempt_rate": commit_attempt_samples / denominator,
        "environment_commit_rate": commit_samples / denominator,
        "environment_commit_answer_exact_rate": answer_exact_commits / denominator,
        "action_failures_per_question": action_failures / denominator,
        "stale_expanded_failures_per_question": stale_expanded_failures / denominator,
    }


def summarize(path: Path, expected: int) -> dict[str, Any]:
    rows = _read_rows(path)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        family = str(row.get("metadata", {}).get("family") or "unknown")
        by_family[family].append(row)
    return {
        "progress_file": str(path),
        "completed": len(rows),
        "expected": expected,
        "completion_rate": len(rows) / expected if expected else 0.0,
        "complete": len(rows) == expected,
        "overall": {
            **_metric_means(rows),
            **_trajectory_diagnostics(rows),
            **_search_diagnostic_means(rows),
        },
        "by_family": {
            family: {
                "examples": len(group),
                **_metric_means(group),
                **_trajectory_diagnostics(group),
                **_search_diagnostic_means(group),
            }
            for family, group in sorted(by_family.items())
        },
    }


def _report(metrics: Mapping[str, Any]) -> str:
    overall = metrics["overall"]
    lines = [
        "# HyPER-R1 Executable Gate",
        "",
        f"Completed: **{metrics['completed']}/{metrics['expected']}** "
        f"({metrics['completion_rate']:.1%})",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in METRICS:
        if key in overall:
            lines.append(f"| {key} | {overall[key]:.4f} |")
    for key in (
        "samples_with_commit_attempt_rate",
        "environment_commit_rate",
        "environment_commit_answer_exact_rate",
        "action_failures_per_question",
        "stale_expanded_failures_per_question",
        *SEARCH_DIAGNOSTICS,
        "actions_after_first_exact_when_seen",
    ):
        if key in overall:
            lines.append(f"| {key} | {overall[key]:.4f} |")
    lines.extend((
        "",
        "## By family",
        "",
        "| Family | N | Protocol-valid Commit | Intent equivalent | Committed-answer F1 |",
    ))
    lines.append("|---|---:|---:|---:|---:|")
    for family, group in metrics["by_family"].items():
        lines.append(
            f"| {family} | {group['examples']} | "
            f"{group.get('hyper_r1_commit_protocol_valid', 0.0):.4f} | "
            f"{group.get('hyper_r1_commit_intent_equivalent', 0.0):.4f} | "
            f"{group.get('hyper_r1_commit_answer_f1', 0.0):.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = summarize(args.progress, args.expected)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "report.md").write_text(_report(metrics), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
