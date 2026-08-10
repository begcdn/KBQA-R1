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
    raise ValueError("validation row has neither mid_f1 nor output/gold answers")


def summarize(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        metadata = row.get("metadata") or {}
        level = str(metadata.get("level") or "overall").lower()
        grouped[level].append(row)

    def metrics(items: List[dict]) -> dict:
        f1 = [_f1(item) for item in items]
        execution = [
            _number(item, "hyper_r1_execution_calls")
            for item in items
            if "hyper_r1_execution_calls" in item
        ]
        return {
            "questions": len(items),
            "mean_f1": mean(f1) if f1 else 0.0,
            "exact_match": mean(value == 1.0 for value in f1) if f1 else 0.0,
            "mean_execution_calls": mean(execution) if execution else None,
        }

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
