#!/usr/bin/env python3
"""Summarize full KBQA-R1/HyPER-R1 validation dumps into stable metrics."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from statistics import mean
from typing import Dict, Iterable, List


def read_rows(path: Path) -> List[dict]:
    files = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
    rows = []
    for file in files:
        with file.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No validation rows found under {path}")
    return rows


def _number(row: dict, key: str, fallback: float = 0.0) -> float:
    try:
        return float(row.get(key, fallback))
    except (TypeError, ValueError):
        return fallback


def _f1(row: dict) -> float:
    if "hyper_r1_commit_answer_f1" in row:
        return _number(row, "hyper_r1_commit_answer_f1")
    if "mid_f1" in row:
        return _number(row, "mid_f1")
    ground_truth = row.get("gts")
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get("target", [])
    if ground_truth is not None and row.get("output") is not None:
        matches = re.findall(
            r"<answer>(.*?)</answer>", str(row["output"]), re.DOTALL | re.IGNORECASE
        )
        predicted = set(matches[-1].strip().split()) if matches else set()
        candidates = ground_truth
        if not candidates:
            candidates = [[]]
        elif not isinstance(candidates[0], list):
            candidates = [candidates]

        best = 0.0
        for candidate in candidates:
            gold = {str(value).strip() for value in candidate if str(value).strip()}
            if not gold:
                score = 1.0 if not predicted else 0.0
            elif not predicted:
                score = 0.0
            else:
                overlap = len(gold & predicted)
                precision = overlap / len(predicted)
                recall = overlap / len(gold)
                score = (
                    2 * precision * recall / (precision + recall)
                    if precision + recall
                    else 0.0
                )
            best = max(best, score)
        return best
    raise ValueError(
        "validation row has neither committed-answer F1, mid_f1, nor output/gold answers"
    )


def summarize(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        metadata = row.get("metadata") or {}
        level = str(metadata.get("level") or "overall").lower()
        grouped[level].append(row)

    def metrics(items: List[dict]) -> dict:
        f1 = [_f1(item) for item in items]
        execution = []
        for item in items:
            if "hyper_r1_execution_attempts" in item:
                execution.append(_number(item, "hyper_r1_execution_attempts"))
            elif "hyper_r1_execution_calls" in item:
                execution.append(_number(item, "hyper_r1_execution_calls"))
        result = {
            "questions": len(items),
            "mean_f1": mean(f1) if f1 else 0.0,
            "exact_match": mean(value == 1.0 for value in f1) if f1 else 0.0,
            "mean_execution_attempts": mean(execution) if execution else None,
        }
        optional_rates = {
            "commit_valid_rate": "hyper_r1_commit_valid",
            "branch_switch_rate": "hyper_r1_branch_switch",
            "combine_usage_rate": "hyper_r1_used_combine",
            "widen_usage_rate": "hyper_r1_used_widen",
            "preserved_alternatives_rate": "hyper_r1_preserved_alternatives",
            "premature_answer_rate": "hyper_r1_premature_answer",
        }
        for output_key, row_key in optional_rates.items():
            values = [_number(item, row_key) for item in items if row_key in item]
            result[output_key] = mean(values) if values else None
        frontier = [
            _number(item, "hyper_r1_max_active")
            for item in items
            if "hyper_r1_max_active" in item
        ]
        result["mean_max_active_frontier"] = mean(frontier) if frontier else None
        for name, key in (
            ("branch_switch_f1", "hyper_r1_branch_switch"),
            ("conjunction_f1", "hyper_r1_used_combine"),
            ("widen_f1", "hyper_r1_used_widen"),
        ):
            selected_items = [item for item in items if _number(item, key) > 0]
            result[name] = (
                mean(_f1(item) for item in selected_items) if selected_items else None
            )
        return result

    result = {"overall": metrics(rows), "by_level": {}}
    for level, items in sorted(grouped.items()):
        if level != "overall":
            result["by_level"][level] = metrics(items)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = summarize(read_rows(Path(args.input)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
