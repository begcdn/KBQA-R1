#!/usr/bin/env python3
"""Build replayable HyPER-R1 demonstrations from processed GrailQA programs."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Iterable, List, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from kbqa_r1.hyper_data import (
    DemonstrationBuilder,
    DemonstrationValidator,
    IneligibleProgram,
    ProgramStatement,
    RelationOption,
    step_sft_records,
)


def _read_rows(path: Path) -> List[dict]:
    if path.suffix == ".parquet":
        from datasets import Dataset

        return [dict(row) for row in Dataset.from_parquet(str(path))]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("input JSON must contain a list of processed GrailQA rows")
    return payload


def _flatten_results(results: Iterable[Any]) -> List[str]:
    values: List[str] = []
    for item in results or ():
        if isinstance(item, dict):
            if item.get("x") is not None:
                values.append(str(item["x"]))
            else:
                values.extend(str(value) for value in item.values() if value is not None)
        else:
            values.append(str(item))
    return values


class LiveProgramExecutor:
    def __init__(self, config: Any):
        from kbqa_r1.sexpr.sexpr_executor import SExprExecutor
        from kbqa_r1.sexpr.sexpr_generator import SExprGenerator

        self.generator = SExprGenerator()
        self.executor = SExprExecutor(config, dataset_type="grailqa")

    def __call__(self, functions: Sequence[str], target: str) -> Iterable[str]:
        generated = self.generator.generate_sexpr_from_strings(list(functions), target)
        if not generated.is_valid:
            return []
        executed = self.executor.execute_sexpr(
            generated.sexpr, function_state=list(functions)
        )
        if not executed.is_successful:
            return []
        return _flatten_results(executed.results)


class LiveCandidateProvider:
    def __init__(self, retrieval: Any, topk: int = 20):
        self.retrieval = retrieval
        self.topk = int(topk)

    def __call__(
        self,
        question: str,
        state_before: Sequence[str],
        decision: ProgramStatement,
    ) -> Sequence[RelationOption]:
        if not state_before:
            return []
        candidates = self.retrieval.get_candidate_relations(list(state_before))
        ranked = self.retrieval.rank_relations_no_threshold(
            question, candidates, topk=self.topk
        )
        return [
            RelationOption(candidate.relation_id, float(candidate.score), rank)
            for rank, candidate in enumerate(ranked, 1)
        ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Processed GrailQA JSON or parquet")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-demonstrations", type=int, default=500)
    parser.add_argument("--relation-topk", type=int, default=20)
    args = parser.parse_args()

    from kbqa_r1.sexpr.relation_retrieval import RelationRetrieval
    from kbqa_r1.sparql.odbc_config import ODBCConfig
    from kbqa_r1.sparql.sparql_manager import SPARQLConfig

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    odbc = ODBCConfig.from_env()
    config = SPARQLConfig(use_odbc=True, odbc_config=asdict(odbc))
    executor = LiveProgramExecutor(config)
    retrieval = RelationRetrieval(
        relation_config={
            "relation_topk": args.relation_topk,
            "relation_threshold": 0.0,
            "entity_topk": 1,
            "entity_threshold": 0.5,
            "cmp_relation_threshold": 0.3,
        },
        sparql_config=config,
        dataset="grailqa",
    )
    builder = DemonstrationBuilder(
        executor, LiveCandidateProvider(retrieval, args.relation_topk)
    )
    validator = DemonstrationValidator(executor)

    demonstrations = []
    skipped = Counter()
    for row_number, row in enumerate(_read_rows(Path(args.input)), 1):
        if len(demonstrations) >= args.max_demonstrations:
            break
        try:
            candidates = builder.build(row)
        except IneligibleProgram as exc:
            skipped[f"ineligible:{str(exc).split()[0]}"] += 1
            continue
        except Exception as exc:
            skipped[f"builder_error:{type(exc).__name__}"] += 1
            continue
        if not candidates:
            skipped["no_qualified_demonstration"] += 1
            continue
        for demo in candidates:
            errors = validator.validate(demo)
            if errors:
                skipped["replay_validation_failed"] += 1
                continue
            demonstrations.append(demo)
            if len(demonstrations) >= args.max_demonstrations:
                break
        if row_number % 100 == 0:
            print(f"processed={row_number} accepted={len(demonstrations)}")

    if not demonstrations:
        raise RuntimeError("no demonstrations passed construction and replay validation")

    with (output / "demonstrations.jsonl").open("w", encoding="utf-8") as handle:
        for demo in demonstrations:
            handle.write(json.dumps(demo.to_dict(), ensure_ascii=False) + "\n")
    with (output / "step_sft.jsonl").open("w", encoding="utf-8") as handle:
        for demo in demonstrations:
            for record in step_sft_records(demo):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    families = Counter(demo.family for demo in demonstrations)
    report = {
        "input": args.input,
        "accepted_demonstrations": len(demonstrations),
        "families": dict(sorted(families.items())),
        "skipped": dict(sorted(skipped.items())),
        "teacher_rationales_added": False,
        "explicit_gold_labels_exposed_to_student": False,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
