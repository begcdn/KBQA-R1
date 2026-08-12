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
    ProgramExecutionError,
    ProgramStatement,
    RelationOption,
    step_sft_records,
    trajectory_sft_record,
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


def _balanced_take(demos, limit: int):
    """Round-robin families, then use every remaining verified example."""
    families = {}
    for demo in demos:
        families.setdefault(demo.family, []).append(demo)
    selected = []
    while len(selected) < limit and any(families.values()):
        for family in sorted(families):
            if families[family] and len(selected) < limit:
                selected.append(families[family].pop(0))
    return selected


class LiveProgramExecutor:
    def __init__(self, config: Any):
        from kbqa_r1.sexpr.sexpr_executor import SExprExecutor
        from kbqa_r1.sexpr.sexpr_generator import SExprGenerator

        self.generator = SExprGenerator()
        self.executor = SExprExecutor(config, dataset_type="grailqa")

    def __call__(self, functions: Sequence[str], target: str) -> Iterable[str]:
        generated = self.generator.generate_sexpr_from_strings(list(functions), target)
        if not generated.is_valid:
            raise ProgramExecutionError(generated.error_message)
        executed = self.executor.execute_sexpr(
            generated.sexpr, function_state=list(functions)
        )
        if not executed.is_successful:
            raise ProgramExecutionError(executed.error_message)
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
    parser.add_argument(
        "--relation-model",
        required=True,
        help="Explicit SimCSE checkpoint used to rank frontier relations",
    )
    parser.add_argument("--max-demonstrations", type=int, default=3000)
    parser.add_argument("--max-input-rows", type=int, default=20000)
    parser.add_argument("--relation-topk", type=int, default=20)
    parser.add_argument("--frontier-width", type=int, default=3)
    parser.add_argument("--max-active", type=int, default=6)
    parser.add_argument("--max-turns", type=int, default=10)
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
        similarity_model_path=args.relation_model,
    )
    builder = DemonstrationBuilder(
        executor,
        LiveCandidateProvider(retrieval, args.relation_topk),
        max_active=args.max_active,
        frontier_width=args.frontier_width,
        max_turns=args.max_turns,
    )
    validator = DemonstrationValidator(executor, max_active=args.max_active)

    qualified = []
    skipped = Counter()
    for row_number, row in enumerate(_read_rows(Path(args.input)), 1):
        if row_number > args.max_input_rows:
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
            qualified.append(demo)
        if row_number % 100 == 0:
            print(f"processed={row_number} qualified={len(qualified)}")

    if not qualified:
        raise RuntimeError("no demonstrations passed construction and replay validation")
    demonstrations = _balanced_take(qualified, args.max_demonstrations)

    with (output / "demonstrations.jsonl").open("w", encoding="utf-8") as handle:
        for demo in demonstrations:
            handle.write(json.dumps(demo.to_dict(), ensure_ascii=False) + "\n")
    with (output / "step_sft.jsonl").open("w", encoding="utf-8") as handle:
        for demo in demonstrations:
            for record in step_sft_records(demo):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    trajectory_rows = [trajectory_sft_record(demo) for demo in demonstrations]
    with (output / "trajectory_sft.jsonl").open("w", encoding="utf-8") as handle:
        for record in trajectory_rows:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    try:
        from datasets import Dataset

        Dataset.from_list(trajectory_rows).to_parquet(str(output / "train.parquet"))
        parquet_written = True
    except ImportError:
        parquet_written = False

    families = Counter(demo.family for demo in demonstrations)
    qualified_families = Counter(demo.family for demo in qualified)
    gold_ranks = Counter(
        str(demo.private_metadata["gold_rank"])
        for demo in demonstrations
        if demo.private_metadata.get("gold_rank") is not None
    )
    source_types = Counter()
    for demo in demonstrations:
        for step in demo.steps:
            if step.action != "Find_relation":
                continue
            source = str(step.arguments[0])
            if source.startswith("expression"):
                source_types["expression"] += 1
            elif source.startswith("m.") or source.startswith("g."):
                source_types["mid"] += 1
            else:
                source_types["literal_or_name"] += 1
    proposal_stats = dict(sorted(builder.stats.items()))
    relation_total = proposal_stats.get("relation_decisions", 0)
    conjunction_total = proposal_stats.get("conjunction_decisions", 0)
    report = {
        "input": args.input,
        "relation_model": args.relation_model,
        "accepted_demonstrations": len(demonstrations),
        "qualified_before_balancing": len(qualified),
        "families": dict(sorted(families.items())),
        "qualified_families_before_balancing": dict(sorted(qualified_families.items())),
        "gold_rank_in_selected_teacher_frontiers": dict(sorted(gold_ranks.items())),
        "find_relation_source_types": dict(sorted(source_types.items())),
        "frontier_width": args.frontier_width,
        "max_active": args.max_active,
        "max_turns": args.max_turns,
        "maximum_selected_trajectory_turns": max(
            (len(demo.steps) + 1 for demo in demonstrations), default=0
        ),
        "sft_rows": len(trajectory_rows),
        "train_parquet_written": parquet_written,
        "skipped": dict(sorted(skipped.items())),
        "proposal_statistics": proposal_stats,
        "proposal_recall": {
            "measurement": (
                "teacher-forced semantic relation hints; measures whether the normal "
                "retriever contains the gold action, not inference-time relation recall"
            ),
            "relation_at_frontier": (
                proposal_stats.get("proposal_hit", 0) / relation_total
                if relation_total else None
            ),
            "both_conjuncts_at_frontier": (
                proposal_stats.get("conjunction_proposal_hit", 0) / conjunction_total
                if conjunction_total else None
            ),
        },
        "actions": dict(
            sorted(Counter(step.action for demo in demonstrations for step in demo.steps).items())
        ),
        "teacher_action_selection_uses_gold_program": True,
        "teacher_rationales": "deterministic templates grounded in verified public steps",
        "explicit_gold_labels_exposed_to_student": False,
        "gold_injected_into_proposals": False,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
