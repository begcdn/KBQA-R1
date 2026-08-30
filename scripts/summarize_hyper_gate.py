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
        "overall": {**_metric_means(rows), **_trajectory_diagnostics(rows)},
        "by_family": {
            family: {
                "examples": len(group),
                **_metric_means(group),
                **_trajectory_diagnostics(group),
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
    ):
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
