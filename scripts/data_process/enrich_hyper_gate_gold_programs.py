#!/usr/bin/env python3
"""Attach private gold programs to a HyPER-R1 executable evaluation gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kbqa_r1.hyper_data import compile_gold_plan, normalize_gold_answers


def _mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"expected mapping metadata, got {type(value).__name__}")


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def load_demo_programs(path: Path) -> Dict[str, tuple[str, ...]]:
    programs: Dict[str, tuple[str, ...]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            demo = json.loads(line)
            demo_id = str(demo.get("demo_id", "")).strip()
            private = _mapping(demo.get("private_metadata", {}))
            functions = private.get("gold_program")
            if not demo_id or not functions:
                continue
            program = tuple(compile_gold_plan(functions).executable_functions)
            previous = programs.setdefault(demo_id, program)
            if previous != program:
                raise ValueError(
                    f"conflicting gold programs for {demo_id!r} at line {line_number}"
                )
    return programs


def enrich_gate(gate_path: Path, demos_path: Path, output_path: Path) -> dict:
    programs = load_demo_programs(demos_path)
    frame = pd.read_parquet(gate_path)
    missing = []
    enriched_rows = []

    for row_number, raw_row in frame.iterrows():
        row = raw_row.to_dict()
        extra = _mapping(row.get("extra_info", {}))
        reward = _mapping(row.get("reward_model", {}))
        ground_truth = _mapping(reward.get("ground_truth", {}))
        source_demo_id = str(extra.get("source_demo_id", "")).strip()
        program = programs.get(source_demo_id)
        if program is None:
            missing.append((int(row_number), source_demo_id))
            continue

        raw_answers = ground_truth.get("target", ())
        if hasattr(raw_answers, "tolist"):
            raw_answers = raw_answers.tolist()
        answers = normalize_gold_answers(raw_answers)
        if not answers:
            raise ValueError(f"row {row_number} has no normalized gold answers")

        prompt_text = json.dumps(
            row.get("prompt", []), ensure_ascii=False, default=_json_default
        )
        leaked_statements = [statement for statement in program if statement in prompt_text]
        if leaked_statements:
            raise ValueError(
                f"row {row_number} leaks private gold statements into the prompt: "
                f"{leaked_statements}"
            )

        ground_truth["function_list"] = list(program)
        reward["ground_truth"] = ground_truth
        row["reward_model"] = reward
        enriched_rows.append(row)

    if missing:
        preview = ", ".join(f"row {row}: {demo!r}" for row, demo in missing[:10])
        raise ValueError(f"missing gold programs for {len(missing)} rows: {preview}")
    if len(enriched_rows) != len(frame):
        raise AssertionError("enrichment changed the number of evaluation rows")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(enriched_rows, columns=frame.columns).to_parquet(output_path, index=False)

    reloaded = pd.read_parquet(output_path)
    for row_number, row in reloaded.iterrows():
        reward = _mapping(row["reward_model"])
        ground_truth = _mapping(reward.get("ground_truth", {}))
        functions = ground_truth.get("function_list")
        if functions is None or len(functions) == 0:
            raise ValueError(f"written row {row_number} lost its gold function_list")
        compile_gold_plan(functions)

    return {
        "input": str(gate_path),
        "output": str(output_path),
        "rows": len(reloaded),
        "programs_available": len(programs),
        "all_programs_compile": True,
        "prompt_leakage": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--demonstrations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            enrich_gate(args.gate, args.demonstrations, args.output),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
