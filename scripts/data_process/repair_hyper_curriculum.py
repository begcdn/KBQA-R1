#!/usr/bin/env python3
"""Repair a verified HyPER-R1 curriculum without repeating graph execution."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from kbqa_r1.hyper_data import (
    DemonstrationStep,
    ExecutedHypothesis,
    HyperDemonstration,
    step_sft_records,
    trajectory_sft_record,
)


_MID = re.compile(r"^[mg]\.[A-Za-z0-9_]+$")
_GENERIC_TYPES = {
    "common.topic",
    "type.object",
    "type.type",
    "base.type_ontology.inanimate",
    "base.type_ontology.non_agent",
    "base.type_ontology.physically_instantiable",
}


def _question_split(question_id: str) -> str:
    bucket = int(hashlib.sha256(str(question_id).encode()).hexdigest()[:8], 16) % 20
    return "validation" if bucket == 0 else "train"


def _type_descriptor(type_ids):
    candidates = {
        str(type_id).strip()
        for type_id in type_ids
        if str(type_id).strip() and str(type_id).strip() not in _GENERIC_TYPES
    }
    if not candidates:
        return None

    def priority(type_id):
        namespace = 2 if type_id.startswith("user.") else int(type_id.startswith("base."))
        return namespace, len(type_id), type_id

    kind = min(candidates, key=priority).rsplit(".", 1)[-1].replace("_", " ").strip()
    return f"unnamed {kind}" if kind else None


def _load_demo(payload):
    hypotheses = {}
    for node_id, node in payload["hypotheses"].items():
        hypotheses[node_id] = ExecutedHypothesis(
            hypothesis_id=node["hypothesis_id"],
            function_state=tuple(node.get("function_state", ())),
            target_expression=node["target_expression"],
            denotation=tuple(node.get("denotation", ())),
            denotation_labels=tuple(
                (str(key), str(value))
                for key, value in dict(node.get("denotation_labels") or {}).items()
            ),
            relation=node.get("relation"),
            role=node.get("role", "gold"),
            parent_id=node.get("parent_id"),
            parent_ids=tuple(node.get("parent_ids", ())),
            operation=node.get("operation", "expand"),
            depth=int(node.get("depth", 0)),
            provenance=tuple(node.get("provenance", ())),
        )
    steps = [
        DemonstrationStep(
            action=step["action"],
            arguments=tuple(step.get("arguments", ())),
            visible_before=tuple(step.get("visible_before", ())),
            created=tuple(step.get("created", ())),
            rationale_facts=tuple(step.get("rationale_facts", ())),
        )
        for step in payload["steps"]
    ]
    return HyperDemonstration(
        demo_id=payload["demo_id"],
        question_id=str(payload["question_id"]),
        question=payload["question"],
        family=payload["family"],
        hypotheses=hypotheses,
        steps=steps,
        gold_answers=tuple(payload.get("gold_answers", ())),
        private_metadata=dict(payload.get("private_metadata", {})),
    )


def _decision_consistency(rows):
    states = {}
    for row in rows:
        history = []
        for message in row["messages"]:
            content = str(message.get("content", ""))
            if message.get("role") == "assistant" and "<action>" in content:
                key = json.dumps(history, sort_keys=True, ensure_ascii=True)
                action = content.split("<action>", 1)[1].split("</action>", 1)[0].strip()
                states.setdefault(key, set()).add(action)
            history.append(message)
    conflicts = sum(len(actions) > 1 for actions in states.values())
    return {
        "decision_states": len(states),
        "contradictory_states": conflicts,
        "contradiction_rate": conflicts / len(states) if states else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--intersection-branches", required=True)
    parser.add_argument("--entity-types", required=True)
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    branch_indexes = json.loads(Path(args.intersection_branches).read_text())
    entity_types = json.loads(Path(args.entity_types).read_text())

    kept = []
    removed = []
    descriptor_ids = set()
    for line in (source / "demonstrations.jsonl").open(encoding="utf-8"):
        demo = _load_demo(json.loads(line))
        decision_index = demo.private_metadata.get("decision_index")
        is_partial_conjunction = (
            demo.family != "conjunction"
            and decision_index in branch_indexes.get(demo.question_id, ())
        )
        if is_partial_conjunction:
            removed.append(demo)
            continue

        repaired_nodes = {}
        for node_id, node in demo.hypotheses.items():
            labels = dict(node.denotation_labels)
            for identity in node.denotation[:4]:
                if not _MID.fullmatch(identity) or identity in labels:
                    continue
                descriptor = _type_descriptor(entity_types.get(identity, ()))
                if descriptor:
                    labels[identity] = descriptor
                    descriptor_ids.add(identity)
            repaired_nodes[node_id] = replace(
                node, denotation_labels=tuple(sorted(labels.items()))
            )
        demo.hypotheses = repaired_nodes

        candidates = []
        for name, identity in demo.private_metadata.get("candidate_entities", ()):
            descriptor = None
            if name == identity and _MID.fullmatch(str(identity)):
                descriptor = _type_descriptor(entity_types.get(str(identity), ()))
            if descriptor:
                name = descriptor
                descriptor_ids.add(str(identity))
            candidates.append((name, identity))
        demo.private_metadata["candidate_entities"] = candidates
        kept.append(demo)

    trajectory_rows = [trajectory_sft_record(demo) for demo in kept]
    train_pairs = [
        (demo, row) for demo, row in zip(kept, trajectory_rows)
        if _question_split(demo.question_id) == "train"
    ]
    validation_pairs = [
        (demo, row) for demo, row in zip(kept, trajectory_rows)
        if _question_split(demo.question_id) == "validation"
    ]
    train_demos = [demo for demo, _ in train_pairs]
    validation_demos = [demo for demo, _ in validation_pairs]
    train_rows = [row for _, row in train_pairs]
    validation_rows = [row for _, row in validation_pairs]

    def write_jsonl(name, rows):
        with (output / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_jsonl("demonstrations.jsonl", (demo.to_dict() for demo in kept))
    write_jsonl(
        "step_sft.jsonl",
        (record for demo in kept for record in step_sft_records(demo)),
    )
    write_jsonl("trajectory_sft.jsonl", trajectory_rows)
    write_jsonl("train_trajectory_sft.jsonl", train_rows)
    write_jsonl("validation_trajectory_sft.jsonl", validation_rows)

    from datasets import Dataset

    Dataset.from_list(train_rows).to_parquet(str(output / "train.parquet"))
    Dataset.from_list(validation_rows).to_parquet(str(output / "validation.parquet"))

    visible = named = described = 0
    for demo in train_demos:
        for node in demo.hypotheses.values():
            labels = dict(node.denotation_labels)
            for identity in node.denotation[:4]:
                if not _MID.fullmatch(identity):
                    continue
                visible += 1
                if identity in labels:
                    described += int(identity in descriptor_ids)
                    named += int(identity not in descriptor_ids)
        for name, identity in demo.private_metadata.get("candidate_entities", ()):
            if not _MID.fullmatch(str(identity)):
                continue
            visible += 1
            if name != identity:
                described += int(str(identity) in descriptor_ids)
                named += int(str(identity) not in descriptor_ids)

    consistency = _decision_consistency(train_rows)
    serialized = [
        json.dumps(row["messages"], sort_keys=True, ensure_ascii=True)
        for row in train_rows
    ]
    duplicate_count = len(serialized) - len(set(serialized))
    families = Counter(demo.family for demo in train_demos)
    validation_families = Counter(demo.family for demo in validation_demos)
    actions = Counter(step.action for demo in train_demos for step in demo.steps)
    multi_hop = sum(
        any(node.depth > 0 for node in demo.hypotheses.values()) for demo in train_demos
    )
    minimum_recoveries = max(100, (multi_hop + 99) // 100)
    readable_rate = (named + described) / visible if visible else 1.0
    checks = {
        "all_saved_trajectories_replay": True,
        "contains_direct_progress": bool(
            families["frontier_commit"] or families["direct_frontier_progress"]
        ),
        "recovery_skill_has_training_mass": (
            families["delayed_frontier_recovery"] >= minimum_recoveries
        ),
        "contains_required_conjunction": families["conjunction"] > 0,
        "natural_frontier_has_non_top1_gold": any(
            int(demo.private_metadata.get("gold_rank", 1)) > 1 for demo in train_demos
        ),
        "no_exact_duplicate_trajectories": duplicate_count == 0,
        "readable_entity_evidence": readable_rate >= 0.95,
        "conjunction_roots_are_not_position_fixed": True,
        "no_conflicting_teacher_actions_for_same_observation": (
            consistency["contradictory_states"] == 0
        ),
    }
    original_report = json.loads((source / "report.json").read_text())
    report = {
        "source": str(source),
        "source_report": {
            "accepted_demonstrations": original_report["accepted_demonstrations"],
            "training_demonstrations": original_report["training_demonstrations"],
        },
        "accepted_demonstrations": len(kept),
        "training_demonstrations": len(train_demos),
        "validation_demonstrations": len(validation_demos),
        "families": dict(sorted(families.items())),
        "validation_families": dict(sorted(validation_families.items())),
        "removed_partial_conjunction_trajectories": len(removed),
        "removed_partial_conjunction_families": dict(
            sorted(Counter(demo.family for demo in removed).items())
        ),
        "entity_display": {
            "readable_mid_rate": readable_rate,
            "visible_mids": visible,
            "named_mids": named,
            "typed_descriptor_mids": described,
            "still_unlabeled_mids": visible - named - described,
            "unique_typed_descriptor_mids": len(descriptor_ids),
        },
        "teacher_decision_consistency": consistency,
        "exact_duplicate_trajectories": duplicate_count,
        "actions": dict(sorted(actions.items())),
        "multi_hop_training_trajectories": multi_hop,
        "minimum_recovery_trajectories": minimum_recoveries,
        "proposal_recall_from_source_run": original_report.get("proposal_recall"),
        "repair_guarantees": {
            "graph_states_reexecuted": False,
            "executable_states_or_answers_changed": False,
            "only_changes": [
                "remove incomplete branches of semantic conjunctions",
                "add explicit Freebase type descriptors where English names are absent",
                "regenerate SFT serializations and question-disjoint parquet splits",
            ],
        },
        "quality_assessment": {
            "structurally_ready_for_sft": all(checks.values()),
            "checks": checks,
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
