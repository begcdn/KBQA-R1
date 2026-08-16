#!/usr/bin/env python3
"""Build replayable HyPER-R1 demonstrations from processed GrailQA programs."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import math
import json
from pathlib import Path
import re
import sys
import threading
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
    compile_gold_plan,
    decision_sft_records,
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


def _curriculum_take(demos, limit: int):
    """Keep the frontier-management signal from being buried by one-hop rows.

    Every verified recovery, multi-hop progression, and conjunction trajectory
    is retained. Direct one-hop commits are necessary positive controls, but
    they are capped at the number of those informative trajectories. This is a
    data-dependent balance rather than an arbitrary corpus-size target.
    """
    families = {}
    for demo in demos:
        families.setdefault(demo.family, []).append(demo)

    informative = [
        demo
        for family, family_demos in families.items()
        if family != "frontier_commit"
        for demo in family_demos
    ]
    commits = list(families.get("frontier_commit", ()))
    if informative:
        commits = commits[: len(informative)]

    selected_families = {}
    for demo in [*informative, *commits]:
        selected_families.setdefault(demo.family, []).append(demo)
    if limit <= 0:
        limit = sum(len(items) for items in selected_families.values())

    selected = []
    while len(selected) < limit and any(selected_families.values()):
        for family in sorted(selected_families):
            if selected_families[family] and len(selected) < limit:
                selected.append(selected_families[family].pop(0))
    return selected


def _question_split(question_id: str) -> str:
    """Deterministic 95/5 split that keeps every trajectory for a question together."""
    bucket = int(hashlib.sha256(str(question_id).encode()).hexdigest()[:8], 16) % 20
    return "validation" if bucket == 0 else "train"


def _decision_contradictions(trajectory_rows):
    """Count identical student observations assigned different graph actions."""
    state_actions = {}
    for row in trajectory_rows:
        history = []
        for message in row.get("messages", ()):
            content = str(message.get("content", ""))
            if message.get("role") == "assistant" and "<action>" in content:
                state = json.dumps(history, sort_keys=True, ensure_ascii=True)
                action = content.split("<action>", 1)[1].split("</action>", 1)[0].strip()
                state_actions.setdefault(state, set()).add(action)
            history.append(message)
    conflicts = [actions for actions in state_actions.values() if len(actions) > 1]
    return {
        "decision_states": len(state_actions),
        "contradictory_states": len(conflicts),
        "contradiction_rate": (
            len(conflicts) / len(state_actions) if state_actions else 0.0
        ),
    }


def _frontier_diagnostics(demonstrations):
    """Measure answer-equivalent alternatives at every executed frontier."""
    frontiers = 0
    multiple_gold = 0
    committed_with_equivalent_sibling = 0
    commits = 0
    for demo in demonstrations:
        node_frontier = {}
        for step in demo.steps:
            if len(step.created) > 1:
                frontiers += 1
                denotations = [demo.hypotheses[node_id].denotation for node_id in step.created]
                multiple_gold += int(sum(value == demo.gold_answers for value in denotations) > 1)
                for node_id in step.created:
                    node_frontier[node_id] = step.created
            if step.action != "Commit":
                continue
            commits += 1
            target = step.arguments[0]
            siblings = node_frontier.get(target, ())
            denotation = demo.hypotheses[target].denotation
            committed_with_equivalent_sibling += int(
                any(
                    sibling != target
                    and demo.hypotheses[sibling].denotation == denotation
                    for sibling in siblings
                )
            )
    return {
        "executed_frontiers": frontiers,
        "frontiers_with_multiple_gold_answer_sets": multiple_gold,
        "multiple_gold_frontier_rate": multiple_gold / frontiers if frontiers else 0.0,
        "commits": commits,
        "commits_with_answer_equivalent_sibling": committed_with_equivalent_sibling,
        "answer_equivalent_commit_rate": (
            committed_with_equivalent_sibling / commits if commits else 0.0
        ),
    }


def _frontier_difficulty_by_family(demonstrations):
    report = {}
    for demo in demonstrations:
        rank = demo.private_metadata.get("gold_rank")
        if rank is None:
            continue
        family = report.setdefault(
            demo.family,
            {"examples": 0, "non_top1_gold": 0, "non_top1_with_nonempty_higher_sibling": 0},
        )
        family["examples"] += 1
        family["non_top1_gold"] += int(int(rank) > 1)
        initial = demo.steps[0].created if demo.steps else ()
        gold_position = next(
            (
                index
                for index, node_id in enumerate(initial)
                if demo.hypotheses[node_id].role == "gold"
            ),
            None,
        )
        if gold_position is not None and gold_position > 0:
            family["non_top1_with_nonempty_higher_sibling"] += int(
                any(demo.hypotheses[node_id].denotation for node_id in initial[:gold_position])
            )
    for values in report.values():
        values["non_top1_gold_rate"] = values["non_top1_gold"] / values["examples"]
        values["non_top1_with_nonempty_higher_sibling_rate"] = (
            values["non_top1_with_nonempty_higher_sibling"] / values["examples"]
        )
    return dict(sorted(report.items()))


_GENERIC_ENTITY_TYPES = {
    "common.topic",
    "type.object",
    "type.type",
    "base.type_ontology.inanimate",
    "base.type_ontology.non_agent",
    "base.type_ontology.physically_instantiable",
}


def _readable_type_descriptor(type_ids: Sequence[str]):
    """Describe an unnamed MID by its most useful asserted Freebase type."""
    candidates = {
        str(type_id).strip()
        for type_id in type_ids
        if str(type_id).strip() and str(type_id).strip() not in _GENERIC_ENTITY_TYPES
    }
    if not candidates:
        return None

    def priority(type_id: str):
        if type_id.startswith("user."):
            namespace = 2
        elif type_id.startswith("base."):
            namespace = 1
        else:
            namespace = 0
        return namespace, len(type_id), type_id

    chosen = min(candidates, key=priority)
    kind = chosen.rsplit(".", 1)[-1].replace("_", " ").strip()
    return f"unnamed {kind}" if kind else None


class LiveEntityDisplayProvider:
    """Cache readable labels while preserving MIDs as executable identities."""

    _MID = re.compile(r"^[mg]\.[A-Za-z0-9_]+$")

    def __init__(self, config: Any):
        from kbqa_r1.sparql.sparql_manager import SPARQLExecutionManager

        self.manager = SPARQLExecutionManager(config)
        self._cache = {}
        self._descriptor_ids = set()
        self._attempted = set()
        self._lock = threading.RLock()

    def remember(self, results: Any) -> None:
        rows = results if isinstance(results, list) else [results]
        types = {}
        with self._lock:
            for row in rows:
                if not isinstance(row, dict):
                    continue
                identity = row.get("x")
                label = row.get("name") or row.get("label")
                if identity and label and str(identity) != str(label):
                    self._cache[str(identity)] = str(label)
                    self._descriptor_ids.discard(str(identity))
                if identity and row.get("type"):
                    types.setdefault(str(identity), set()).add(str(row["type"]))
            for identity, type_ids in types.items():
                if identity in self._cache:
                    continue
                descriptor = _readable_type_descriptor(tuple(type_ids))
                if descriptor:
                    self._cache[identity] = descriptor
                    self._descriptor_ids.add(identity)

    def is_type_descriptor(self, identity: str) -> bool:
        with self._lock:
            return str(identity) in self._descriptor_ids

    def __call__(self, values: Sequence[str]):
        identities = [str(value) for value in values]
        with self._lock:
            missing = [
                value for value in identities
                if self._MID.fullmatch(value)
                and value not in self._cache
                and value not in self._attempted
            ]
            self._attempted.update(missing)
            if missing:
                terms = " ".join(f"ns:{value}" for value in missing)
                query = f"""SPARQL
PREFIX ns: <http://rdf.freebase.com/ns/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?x ?name ?type WHERE {{
  VALUES ?x {{ {terms} }}
  OPTIONAL {{ ?x rdfs:label ?name . FILTER(langMatches(lang(?name), 'en')) }}
  OPTIONAL {{ ?x ns:type.object.type ?type . }}
}}"""
                payload = self.manager.execute_batch([query])
                envelope = (payload.get("results") or [{}])[0] or {}
                self.remember(envelope.get("results") or [])

            for identity in identities:
                if identity in self._cache:
                    continue
                if not self._MID.fullmatch(identity) and "." in identity:
                    self._cache[identity] = identity.rsplit(".", 1)[-1].replace("_", " ")
            return {
                identity: self._cache[identity]
                for identity in identities
                if identity in self._cache
            }


class LiveProgramExecutor:
    def __init__(self, config: Any, display_provider: LiveEntityDisplayProvider):
        from kbqa_r1.sexpr.sexpr_executor import SExprExecutor
        from kbqa_r1.sexpr.sexpr_generator import SExprGenerator

        self.generator = SExprGenerator()
        self.executor = SExprExecutor(config, dataset_type="grailqa")
        self.display_provider = display_provider

    def __call__(self, functions: Sequence[str], target: str) -> Iterable[str]:
        generated = self.generator.generate_sexpr_from_strings(list(functions), target)
        if not generated.is_valid:
            raise ProgramExecutionError(generated.error_message)
        executed = self.executor.execute_sexpr(
            generated.sexpr, function_state=list(functions)
        )
        if not executed.is_successful:
            raise ProgramExecutionError(executed.error_message)
        self.display_provider.remember(executed.results)
        return _flatten_results(executed.results)


class LiveCandidateProvider:
    def __init__(self, retrieval: Any, topk: int = 20):
        self.retrieval = retrieval
        self.topk = int(topk)
        self._rank_lock = threading.Lock()

    def __call__(
        self,
        question: str,
        state_before: Sequence[str],
        decision: ProgramStatement,
    ) -> Sequence[RelationOption]:
        if not state_before:
            return []
        candidates = self.retrieval.get_candidate_relations(list(state_before))
        # Graph adjacency is fetched concurrently. A single model instance
        # ranks the returned vocabularies, so serialize only this short GPU call.
        with self._rank_lock:
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
    parser.add_argument(
        "--max-demonstrations", type=int, default=0,
        help="Maximum saved trajectories; 0 uses every verified trajectory",
    )
    parser.add_argument(
        "--max-input-rows", type=int, default=0,
        help="Maximum input questions; 0 processes the complete training split",
    )
    parser.add_argument("--relation-topk", type=int, default=20)
    parser.add_argument("--frontier-width", type=int, default=3)
    parser.add_argument("--max-frontier-width", type=int, default=6)
    parser.add_argument("--max-active", type=int, default=6)
    parser.add_argument("--max-turns", type=int, default=14)
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent graph-query workers; relation ranking remains serialized",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    from kbqa_r1.sexpr.relation_retrieval import RelationRetrieval
    from kbqa_r1.sparql.odbc_config import ODBCConfig
    from kbqa_r1.sparql.sparql_manager import SPARQLConfig

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    odbc = ODBCConfig.from_env()
    config = SPARQLConfig(use_odbc=True, odbc_config=asdict(odbc))
    display_provider = LiveEntityDisplayProvider(config)
    executor = LiveProgramExecutor(config, display_provider)
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
    provider = LiveCandidateProvider(retrieval, args.relation_topk)

    def build_row(row):
        builder = DemonstrationBuilder(
            executor,
            provider,
            max_active=args.max_active,
            frontier_width=args.frontier_width,
            max_frontier_width=args.max_frontier_width,
            max_turns=args.max_turns,
            entity_display_provider=display_provider,
        )
        demonstrations = builder.build(row)
        validator = DemonstrationValidator(
            executor,
            max_active=args.max_active,
            execution_cache=builder._execution_cache,
        )
        for demo in demonstrations:
            errors = validator.validate(demo)
            if errors:
                raise RuntimeError(
                    f"replay validation failed for {demo.demo_id}: "
                    + "; ".join(errors)
                )
        return demonstrations, builder.stats

    qualified = []
    skipped = Counter()
    proposal_stats = Counter()
    rows_by_hop = Counter()
    accepted_rows_by_hop = Counter()
    rows = _read_rows(Path(args.input))
    if args.max_input_rows > 0:
        rows = rows[: args.max_input_rows]

    def prepared_rows():
        for row in rows:
            try:
                compile_gold_plan(row.get("function_list") or ())
            except IneligibleProgram as exc:
                yield row, None, f"ineligible:{str(exc).split()[0]}"
            else:
                yield row, row, None

    prepared = list(prepared_rows())
    eligible = [row for _, row, error in prepared if row is not None and error is None]
    result_iterator = iter(())
    pool = None
    if eligible:
        if args.workers > 1:
            pool = ThreadPoolExecutor(max_workers=args.workers)
            result_iterator = pool.map(build_row, eligible)
        else:
            result_iterator = map(build_row, eligible)
    result_iterator = iter(result_iterator)

    for row_number, (original, eligible_row, prefilter_error) in enumerate(prepared, 1):
        row = original
        hop_count = sum("JOIN(" in str(item) for item in row.get("function_list", ()))
        rows_by_hop[str(hop_count)] += 1
        if prefilter_error is not None:
            skipped[prefilter_error] += 1
            continue
        try:
            candidates, row_stats = next(result_iterator)
        except Exception as exc:
            question_id = row.get("ID") or row.get("id") or row_number
            raise RuntimeError(
                f"demonstration construction failed for row {question_id}"
            ) from exc
        proposal_stats.update(row_stats)
        if not candidates:
            skipped["no_qualified_demonstration"] += 1
            continue
        accepted_this_row = False
        for demo in candidates:
            qualified.append(demo)
            accepted_this_row = True
        if accepted_this_row:
            accepted_rows_by_hop[str(hop_count)] += 1
        if row_number % 100 == 0:
            print(f"processed={row_number} qualified={len(qualified)}")
    if pool is not None:
        pool.shutdown(wait=True)

    if not qualified:
        raise RuntimeError("no demonstrations passed construction and replay validation")
    unique_qualified = []
    seen_trajectories = set()
    duplicates_removed = 0
    for demo in qualified:
        serialized = json.dumps(
            trajectory_sft_record(demo)["messages"],
            sort_keys=True,
            ensure_ascii=True,
        )
        if serialized in seen_trajectories:
            duplicates_removed += 1
            continue
        seen_trajectories.add(serialized)
        unique_qualified.append(demo)
    demonstrations = _curriculum_take(
        unique_qualified, args.max_demonstrations
    )

    with (output / "demonstrations.jsonl").open("w", encoding="utf-8") as handle:
        for demo in demonstrations:
            handle.write(json.dumps(demo.to_dict(), ensure_ascii=False) + "\n")
    with (output / "step_sft.jsonl").open("w", encoding="utf-8") as handle:
        for demo in demonstrations:
            for record in step_sft_records(demo):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    trajectory_rows = [trajectory_sft_record(demo) for demo in demonstrations]
    decision_rows = [
        row for demo in demonstrations for row in decision_sft_records(demo)
    ]
    with (output / "trajectory_sft.jsonl").open("w", encoding="utf-8") as handle:
        for record in trajectory_rows:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (output / "decision_sft.jsonl").open("w", encoding="utf-8") as handle:
        for record in decision_rows:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    train_pairs = [
        (demo, row) for demo, row in zip(demonstrations, trajectory_rows)
        if _question_split(demo.question_id) == "train"
    ]
    validation_pairs = [
        (demo, row) for demo, row in zip(demonstrations, trajectory_rows)
        if _question_split(demo.question_id) == "validation"
    ]
    train_demonstrations = [demo for demo, _ in train_pairs]
    validation_demonstrations = [demo for demo, _ in validation_pairs]
    train_rows = [row for _, row in train_pairs]
    validation_rows = [row for _, row in validation_pairs]
    train_decision_rows = [
        row for demo in train_demonstrations for row in decision_sft_records(demo)
    ]
    validation_decision_rows = [
        row for demo in validation_demonstrations for row in decision_sft_records(demo)
    ]
    for name, rows_for_split in (
        ("train_trajectory_sft.jsonl", train_rows),
        ("validation_trajectory_sft.jsonl", validation_rows),
        ("train_decision_sft.jsonl", train_decision_rows),
        ("validation_decision_sft.jsonl", validation_decision_rows),
    ):
        with (output / name).open("w", encoding="utf-8") as handle:
            for record in rows_for_split:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    try:
        from datasets import Dataset

        Dataset.from_list(train_rows).to_parquet(str(output / "train.parquet"))
        Dataset.from_list(train_decision_rows).to_parquet(
            str(output / "train_decision.parquet")
        )
        if validation_rows:
            Dataset.from_list(validation_rows).to_parquet(
                str(output / "validation.parquet")
            )
            Dataset.from_list(validation_decision_rows).to_parquet(
                str(output / "validation_decision.parquet")
            )
        parquet_written = True
    except ImportError:
        parquet_written = False

    families = Counter(demo.family for demo in train_demonstrations)
    validation_families = Counter(demo.family for demo in validation_demonstrations)
    qualified_families = Counter(demo.family for demo in unique_qualified)
    gold_ranks = Counter(
        str(demo.private_metadata["gold_rank"])
        for demo in train_demonstrations
        if demo.private_metadata.get("gold_rank") is not None
    )
    source_types = Counter()
    runtime_incompatible_expression_sources = 0
    for demo in train_demonstrations:
        for step in demo.steps:
            if step.action not in {"Find_relation", "Widen"}:
                continue
            source = str(step.arguments[0])
            runtime_incompatible_expression_sources += int(source == "expression")
            if source.startswith("expression"):
                source_types["expression"] += 1
            elif source.startswith("m.") or source.startswith("g."):
                source_types["mid"] += 1
            else:
                source_types["literal_or_name"] += 1
    proposal_stats = dict(sorted(proposal_stats.items()))
    relation_total = proposal_stats.get("relation_decisions", 0)
    continuation_total = proposal_stats.get("continuation_decisions", 0)
    recovery_probe_total = proposal_stats.get("recovery_probe_decisions", 0)
    conjunction_total = proposal_stats.get("conjunction_decisions", 0)
    relation_recall = (
        proposal_stats.get("proposal_hit", 0) / relation_total
        if relation_total else None
    )
    relation_max_recall = (
        proposal_stats.get("proposal_max_frontier_hit", 0) / relation_total
        if relation_total else None
    )
    relation_top1 = (
        proposal_stats.get("proposal_top1_hit", 0) / relation_total
        if relation_total else None
    )
    margins = [
        float(demo.private_metadata["gold_vs_best_alternative_margin"])
        for demo in train_demonstrations
        if demo.private_metadata.get("gold_vs_best_alternative_margin") is not None
    ]
    hypothesis_nodes = [
        node for demo in train_demonstrations for node in demo.hypotheses.values()
    ]
    visible_mid_count = 0
    labeled_mid_count = 0
    descriptor_mid_count = 0
    for node in hypothesis_nodes:
        labels = dict(node.denotation_labels)
        for identity in node.denotation[:4]:
            if LiveEntityDisplayProvider._MID.fullmatch(identity):
                visible_mid_count += 1
                labeled_mid_count += int(identity in labels)
                descriptor_mid_count += int(
                    identity in labels and display_provider.is_type_descriptor(identity)
                )
    candidate_mid_count = 0
    candidate_labeled_count = 0
    candidate_descriptor_count = 0
    conjunction_source_positions = Counter()
    for demo in train_demonstrations:
        candidates = list(demo.private_metadata.get("candidate_entities", ()))
        candidate_ids = [str(entity[-1]) for entity in candidates if entity]
        for entity in candidates:
            if not entity:
                continue
            name, identity = str(entity[0]), str(entity[-1])
            if LiveEntityDisplayProvider._MID.fullmatch(identity):
                candidate_mid_count += 1
                candidate_labeled_count += int(name != identity)
                candidate_descriptor_count += int(
                    name != identity and display_provider.is_type_descriptor(identity)
                )
        if demo.family == "conjunction":
            roots = [
                str(step.arguments[0])
                for step in demo.steps
                if step.action == "Find_relation"
                and not str(step.arguments[0]).startswith("expression")
            ]
            positions = tuple(
                candidate_ids.index(root) for root in roots if root in candidate_ids
            )
            if positions:
                conjunction_source_positions[str(positions)] += 1
    serialized_trajectories = [
        json.dumps(row["messages"], sort_keys=True, ensure_ascii=True)
        for row in train_rows
    ]
    duplicate_rate = (
        1 - len(set(serialized_trajectories)) / len(serialized_trajectories)
        if serialized_trajectories else None
    )
    decision_consistency = _decision_contradictions(train_rows)
    frontier_diagnostics = _frontier_diagnostics(train_demonstrations)
    family_difficulty = _frontier_difficulty_by_family(train_demonstrations)
    recovery_count = (
        families.get("delayed_frontier_recovery", 0)
        + families.get("semantic_frontier_recovery", 0)
    )
    multi_hop_count = sum(
        any(node.depth > 0 for node in demo.hypotheses.values())
        for demo in train_demonstrations
    )
    minimum_recoveries = max(500, math.ceil(0.10 * multi_hop_count))
    entity_display_total = visible_mid_count + candidate_mid_count
    entity_display_labeled = labeled_mid_count + candidate_labeled_count
    entity_display_described = descriptor_mid_count + candidate_descriptor_count
    entity_display_rate = (
        entity_display_labeled / entity_display_total
        if entity_display_total else 1.0
    )
    varied_conjunction_order = (
        len(conjunction_source_positions) >= 2
        if families.get("conjunction", 0) >= 2 else True
    )
    deep_input_rows = sum(
        count for hop, count in rows_by_hop.items() if int(hop) >= 3
    )
    deep_training_trajectories = (
        families.get("deep_frontier_progress", 0)
        + families.get("deep_conjunction_progress", 0)
    )
    minimum_deep_trajectories = (
        min(deep_input_rows, max(100, math.ceil(0.10 * deep_input_rows)))
        if deep_input_rows else 0
    )
    quality_checks = {
        "all_saved_trajectories_replay": True,
        "contains_direct_progress": bool(
            families.get("frontier_commit", 0)
            or families.get("direct_frontier_progress", 0)
        ),
        "recovery_skill_has_training_mass": recovery_count >= minimum_recoveries,
        "adaptive_widening_is_represented": (
            proposal_stats.get("proposal_recovered_by_widen", 0) == 0
            or families.get("adaptive_frontier_widen", 0) > 0
        ),
        "contains_required_conjunction": (
            families.get("conjunction", 0)
            + families.get("deep_conjunction_progress", 0)
        ) > 0,
        "deep_progress_has_training_mass": (
            deep_training_trajectories >= minimum_deep_trajectories
        ),
        "natural_frontier_has_non_top1_gold": any(
            int(rank) > 1 and count > 0 for rank, count in gold_ranks.items()
        ),
        "no_exact_duplicate_trajectories": duplicate_rate == 0.0,
        "readable_entity_evidence": entity_display_rate >= 0.95,
        "conjunction_roots_are_not_position_fixed": varied_conjunction_order,
        "all_expression_sources_match_runtime_vocabulary": (
            runtime_incompatible_expression_sources == 0
        ),
        "no_conflicting_teacher_actions_for_same_observation": (
            decision_consistency["contradictory_states"] == 0
        ),
    }
    report = {
        "quality_schema": "hyper_r1_v6_depth_aware",
        "input": args.input,
        "relation_model": args.relation_model,
        "accepted_demonstrations": len(demonstrations),
        "training_demonstrations": len(train_demonstrations),
        "qualified_before_balancing": len(qualified),
        "qualified_after_exact_deduplication": len(unique_qualified),
        "curriculum_selection": (
            "retain every verified recovery, multi-hop progression, and conjunction; "
            "cap repetitive one-hop commits at the informative-trajectory count"
        ),
        "families": dict(sorted(families.items())),
        "validation_families": dict(sorted(validation_families.items())),
        "qualified_families_before_balancing": dict(sorted(qualified_families.items())),
        "gold_rank_in_selected_teacher_frontiers": dict(sorted(gold_ranks.items())),
        "gold_rank_by_family": family_difficulty,
        "find_relation_source_types": dict(sorted(source_types.items())),
        "runtime_incompatible_expression_sources": (
            runtime_incompatible_expression_sources
        ),
        "frontier_width": args.frontier_width,
        "max_frontier_width": args.max_frontier_width,
        "max_active": args.max_active,
        "max_turns": args.max_turns,
        "graph_query_workers": args.workers,
        "maximum_selected_trajectory_turns": max(
            (len(demo.steps) + 1 for demo in demonstrations), default=0
        ),
        "sft_rows": len(train_rows),
        "decision_sft_rows": len(train_decision_rows),
        "validation_rows": len(validation_rows),
        "question_disjoint_split": "sha256_question_id_95_5",
        "train_parquet_written": parquet_written,
        "skipped": dict(sorted(skipped.items())),
        "proposal_statistics": proposal_stats,
        "proposal_recall": {
            "measurement": (
                "question-derived intent with the runtime relation ranker; gold is used "
                "only to measure whether the natural frontier contains the required action"
            ),
            "relation_at_frontier": (
                relation_recall
            ),
            "relation_at_max_frontier": relation_max_recall,
            "recovered_by_widen": (
                relation_max_recall - relation_recall
                if relation_max_recall is not None and relation_recall is not None
                else None
            ),
            "relation_top1": relation_top1,
            "topk_opportunity_over_top1": (
                relation_recall - relation_top1
                if relation_recall is not None and relation_top1 is not None else None
            ),
            "both_conjuncts_at_frontier": (
                proposal_stats.get("conjunction_proposal_hit", 0) / conjunction_total
                if conjunction_total else None
            ),
            "continuation_at_frontier": (
                proposal_stats.get("continuation_proposal_hit", 0)
                / continuation_total
                if continuation_total else None
            ),
            "continuation_top1": (
                proposal_stats.get("continuation_top1_hit", 0)
                / continuation_total
                if continuation_total else None
            ),
            "recovery_probe_at_frontier": (
                proposal_stats.get("recovery_probe_natural_frontier", 0)
                / recovery_probe_total
                if recovery_probe_total else None
            ),
            "recovery_probe_visible_failure": (
                proposal_stats.get("recovery_probe_visible_empty", 0)
                / recovery_probe_total
                if recovery_probe_total else None
            ),
        },
        "acceptance_by_hop_count": {
            hop: {
                "rows": count,
                "accepted_rows": accepted_rows_by_hop.get(hop, 0),
                "rate": accepted_rows_by_hop.get(hop, 0) / count,
            }
            for hop, count in sorted(rows_by_hop.items(), key=lambda item: int(item[0]))
        },
        "frontier_difficulty": {
            "gold_vs_best_alternative_margin_mean": (
                sum(margins) / len(margins) if margins else None
            ),
            "non_top1_gold_rate": (
                sum(count for rank, count in gold_ranks.items() if int(rank) > 1)
                / sum(gold_ranks.values()) if gold_ranks else None
            ),
            "by_family": family_difficulty,
        },
        "empty_hypothesis_rate": (
            sum(not node.denotation for node in hypothesis_nodes) / len(hypothesis_nodes)
            if hypothesis_nodes else None
        ),
        "entity_display": {
            "labeled_mid_rate": entity_display_rate,
            "labeled_mids": entity_display_labeled,
            "visible_mids": entity_display_total,
            "named_mids": entity_display_labeled - entity_display_described,
            "typed_descriptor_mids": entity_display_described,
            "candidate_labeled_mids": candidate_labeled_count,
            "candidate_mids": candidate_mid_count,
            "denotation_labeled_mids": labeled_mid_count,
            "denotation_mids": visible_mid_count,
        },
        "candidate_entity_order": {
            "method": "stable_question_hash",
            "conjunction_source_position_patterns": dict(
                sorted(conjunction_source_positions.items())
            ),
        },
        "denotational_ambiguity": {
            **frontier_diagnostics,
            "interpretation": (
                "Answer equality is diagnostic only; it does not turn a different "
                "relation path into an additional semantic positive."
            ),
        },
        "teacher_decision_consistency": decision_consistency,
        "exact_duplicate_trajectory_rate": (
            duplicate_rate
        ),
        "exact_duplicate_trajectories_removed": duplicates_removed,
        "actions": dict(
            sorted(
                Counter(
                    step.action
                    for demo in train_demonstrations
                    for step in demo.steps
                ).items()
            )
        ),
        "teacher_action_selection_uses_gold_program": True,
        "retrieval_intent_source": "question_and_visible_state_only",
        "teacher_rationales": "deterministic templates grounded in verified public steps",
        "explicit_gold_labels_exposed_to_student": False,
        "gold_injected_into_proposals": False,
        "quality_assessment": {
            "structurally_ready_for_sft": all(quality_checks.values()),
            "checks": quality_checks,
            "minimum_recovery_trajectories": minimum_recoveries,
            "multi_hop_training_trajectories": multi_hop_count,
            "deep_input_rows": deep_input_rows,
            "deep_training_trajectories": deep_training_trajectories,
            "minimum_deep_trajectories": minimum_deep_trajectories,
            "note": (
                "Validity failures block export. Expensive SFT additionally requires "
                "readable entity evidence, non-positional conjunction roots, and enough "
                "verified recovery and deep trajectories to teach the method-specific behavior. "
                "Decision-level SFT rows exclude the final answer-copy turn."
            ),
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
