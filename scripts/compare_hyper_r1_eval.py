#!/usr/bin/env python3
"""Paired benchmark comparison for HyPER-R1 and executable baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from statistics import mean
from typing import Dict, Iterable, List, Tuple

from scripts.analyze_hyper_r1_eval import _f1, _number, read_rows


def row_id(row: dict) -> str:
    metadata = row.get("metadata") or {}
    value = metadata.get("id") or metadata.get("qid") or metadata.get("question_id")
    if value is None:
        raise ValueError("paired evaluation row is missing metadata.id")
    return str(value)


def paired_bootstrap(
    differences: List[float], samples: int = 10_000, seed: int = 17
) -> Tuple[float, float]:
    if not differences:
        raise ValueError("cannot bootstrap an empty comparison")
    rng = random.Random(seed)
    size = len(differences)
    estimates = sorted(
        mean(differences[rng.randrange(size)] for _ in range(size))
        for _ in range(samples)
    )
    return estimates[int(0.025 * samples)], estimates[int(0.975 * samples)]


def compare(baseline_rows: Iterable[dict], method_rows: Iterable[dict]) -> dict:
    baseline = {row_id(row): row for row in baseline_rows}
    method = {row_id(row): row for row in method_rows}
    common = sorted(set(baseline) & set(method))
    if len(common) != len(baseline) or len(common) != len(method):
        raise ValueError(
            "paired evaluation requires identical question populations: "
            f"baseline={len(baseline)} method={len(method)} shared={len(common)}"
        )

    def group(ids: List[str]) -> dict:
        baseline_f1 = [_f1(baseline[qid]) for qid in ids]
        method_f1 = [_f1(method[qid]) for qid in ids]
        differences = [right - left for left, right in zip(baseline_f1, method_f1)]
        low, high = paired_bootstrap(differences)
        baseline_calls = [
            _number(baseline[qid], "hyper_r1_execution_calls")
            for qid in ids
            if "hyper_r1_execution_calls" in baseline[qid]
        ]
        method_calls = [
            _number(method[qid], "hyper_r1_execution_calls")
            for qid in ids
            if "hyper_r1_execution_calls" in method[qid]
        ]
        return {
            "questions": len(ids),
            "baseline_f1": mean(baseline_f1),
            "method_f1": mean(method_f1),
            "delta_f1": mean(differences),
            "delta_f1_bootstrap_95": [low, high],
            "baseline_exact_match": mean(score == 1.0 for score in baseline_f1),
            "method_exact_match": mean(score == 1.0 for score in method_f1),
            "baseline_execution_calls": mean(baseline_calls) if baseline_calls else None,
            "method_execution_calls": mean(method_calls) if method_calls else None,
            "method_wins": sum(delta > 0 for delta in differences),
            "baseline_wins": sum(delta < 0 for delta in differences),
            "ties": sum(delta == 0 for delta in differences),
        }

    by_level: Dict[str, List[str]] = {}
    for qid in common:
        metadata = method[qid].get("metadata") or {}
        level = str(metadata.get("level") or "unknown").lower()
        by_level.setdefault(level, []).append(qid)

    return {
        "overall": group(common),
        "by_level": {
            level: group(ids) for level, ids in sorted(by_level.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = compare(read_rows(Path(args.baseline)), read_rows(Path(args.method)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
