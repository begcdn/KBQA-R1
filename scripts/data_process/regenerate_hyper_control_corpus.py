#!/usr/bin/env python3
"""Regenerate HyPER-R1 SFT from structured demonstrations.

This exporter adds only transitions whose legality can be reconstructed from
the stored graph state: reversible Park/Recall recovery and masked
invalid-action recovery contexts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Iterator

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from kbqa_r1.hyper_data import (  # noqa: E402
    _public_graph,
    answer_set_f1,
    DemonstrationStep,
    HyperDemonstration,
    decision_sft_records,
)
from kbqa_r1.hyper_r1 import HypothesisGraph, render_hyper_information  # noqa: E402
from scripts.data_process.build_hyper_demonstrations import _question_split  # noqa: E402
from scripts.data_process.repair_hyper_curriculum import _load_demo  # noqa: E402


_EXECUTING_ACTIONS = {
    "Inspect",
    "Combine",
    "Merge",
    "Order",
    "Compare",
    "Time_constraint",
    "Count",
}

_SOURCE_CONTRACT_FIELDS = (
    "quality_schema",
    "relation_page_size",
    "relation_rank_cutoff",
    "max_active",
    "max_nodes",
    "max_execution_attempts",
    "max_turns",
    "quality_assessment",
    "proposal_recall",
    "families",
)


def _load_source_contract(input_path: Path) -> dict:
    report_path = input_path.parent / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(
            f"source corpus contract is missing: {report_path}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    inherited = report.get("source_contract", {})
    missing = [
        field
        for field in _SOURCE_CONTRACT_FIELDS
        if field not in report and field not in inherited
    ]
    if missing:
        raise ValueError(
            "source corpus report is missing contract fields: "
            + ", ".join(missing)
        )
    contract = {
        field: report[field] if field in report else inherited[field]
        for field in _SOURCE_CONTRACT_FIELDS
    }
    contract["report"] = str(report_path)
    contract["report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    return contract


def _read_demos(path: Path) -> list[HyperDemonstration]:
    with path.open(encoding="utf-8") as handle:
        return [_load_demo(json.loads(line)) for line in handle if line.strip()]


def _iter_demos(path: Path) -> Iterator[HyperDemonstration]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield _load_demo(json.loads(line))


_HYPOTHESIS_ID = re.compile(r"\bH\d+\b")


def _rewrite_hypothesis_ids(value: Any, mapping: dict[str, str]) -> Any:
    """Rewrite exact hypothesis references inside structured metadata."""
    if isinstance(value, str):
        return _HYPOTHESIS_ID.sub(lambda match: mapping.get(match.group(0), match.group(0)), value)
    if isinstance(value, tuple):
        return tuple(_rewrite_hypothesis_ids(item, mapping) for item in value)
    if isinstance(value, list):
        return [_rewrite_hypothesis_ids(item, mapping) for item in value]
    if isinstance(value, dict):
        return {
            _rewrite_hypothesis_ids(key, mapping): _rewrite_hypothesis_ids(item, mapping)
            for key, item in value.items()
        }
    return value


def _runtime_creation_order(demo: HyperDemonstration) -> list[str]:
    """Return the order in which the live graph would mint node IDs."""
    known = set(demo.hypotheses)
    order: list[str] = []
    seen: set[str] = set()
    if demo.steps:
        for node_id in demo.steps[0].visible_before:
            if node_id in known and node_id not in seen:
                order.append(node_id)
                seen.add(node_id)
    for step in demo.steps:
        for node_id in step.created:
            if node_id in known and node_id not in seen:
                order.append(node_id)
                seen.add(node_id)
    for node_id in demo.hypotheses:
        if node_id not in seen:
            order.append(node_id)
            seen.add(node_id)
    return order


def _runtime_ordered_demo(demo: HyperDemonstration) -> HyperDemonstration:
    """Rename every node and reference to the IDs minted by live runtime."""
    order = _runtime_creation_order(demo)
    mapping = {node_id: f"H{index}" for index, node_id in enumerate(order)}
    hypotheses = {}
    for old_id in order:
        node = demo.hypotheses[old_id]
        new_id = mapping[old_id]
        hypotheses[new_id] = replace(
            node,
            hypothesis_id=new_id,
            parent_id=mapping.get(node.parent_id, node.parent_id),
            parent_ids=tuple(mapping.get(value, value) for value in node.parent_ids),
            provenance=_rewrite_hypothesis_ids(node.provenance, mapping),
        )
    steps = [
        replace(
            step,
            arguments=_rewrite_hypothesis_ids(step.arguments, mapping),
            visible_before=tuple(
                sorted(
                    _rewrite_hypothesis_ids(step.visible_before, mapping),
                    key=lambda node_id: int(str(node_id)[1:]),
                )
            ),
            created=_rewrite_hypothesis_ids(step.created, mapping),
            rationale_facts=_rewrite_hypothesis_ids(step.rationale_facts, mapping),
            certificate_evidence=_rewrite_hypothesis_ids(
                step.certificate_evidence, mapping
            ),
        )
        for step in demo.steps
    ]
    return replace(
        demo,
        hypotheses=hypotheses,
        steps=steps,
        private_metadata=_rewrite_hypothesis_ids(demo.private_metadata, mapping),
    )


def _hypotheses_follow_runtime_order(demo: HyperDemonstration) -> bool:
    expected = [f"H{index}" for index in range(len(demo.hypotheses))]
    if list(demo.hypotheses) != expected:
        return False
    known = set(demo.steps[0].visible_before if demo.steps else ())
    next_index = len(known)
    if known != set(expected[:next_index]):
        return False
    for step in demo.steps:
        if len(step.visible_before) != len(set(step.visible_before)):
            return False
        for node_id in step.created:
            if node_id != f"H{next_index}":
                return False
            known.add(node_id)
            next_index += 1
        referenced = {
            value
            for value in step.arguments
            if _HYPOTHESIS_ID.fullmatch(str(value))
        }
        if not referenced.issubset(known):
            return False
    return known == set(expected)


def _runtime_replay_errors(demo: HyperDemonstration) -> list[str]:
    """Replay a teacher trajectory through the real graph state machine."""
    graph = HypothesisGraph(
        max_active=int(demo.private_metadata.get("max_active", 24)),
        max_nodes=int(demo.private_metadata.get("max_nodes", 128)),
        max_execution_attempts=int(
            demo.private_metadata.get("max_execution_attempts", 24)
        ),
    )
    graph.register_public_question(0, demo.question)
    max_turns = int(demo.private_metadata.get("max_turns", 32))
    turn_offset = int(demo.private_metadata.get("turn_offset", 0))
    if turn_offset < 0 or turn_offset + len(demo.steps) > max_turns:
        return ["invalid time-shifted turn clock"]
    frontiers: dict[str, str | None] = {}
    errors: list[str] = []
    candidate_sources = [
        str(candidate[-1])
        for key in ("candidate_entities", "candidate_literals")
        for candidate in demo.private_metadata.get(key, ())
        if candidate
    ]

    for step_index, step in enumerate(demo.steps):
        graph.set_clock(
            0,
            turns_used=turn_offset + step_index,
            max_turns=max_turns,
        )
        active = tuple(node.node_id for node in graph.active_nodes(0))
        if active != tuple(step.visible_before):
            errors.append(
                f"step {step_index} {step.action}: runtime active={active} "
                f"teacher active={step.visible_before}"
            )
            break
        known = tuple(graph.state(0).nodes)
        parked = tuple(node.node_id for node in graph.parked_nodes(0))
        expected_graph = _public_graph(
            demo,
            active,
            selected=graph.state(0).selected_id,
            executions=graph.state(0).execution_attempts,
            known_ids=known,
            parked_ids=parked,
            turns_used=turn_offset + step_index,
        )
        actual_graph = graph.serialize(0, candidate_sources=candidate_sources)
        if actual_graph != expected_graph:
            errors.append(
                f"step {step_index} {step.action}: rendered runtime state differs"
            )
            break

        try:
            if step.action == "Find_relation":
                for proposal_id in step.exposed:
                    frontiers[demo.proposals[proposal_id].frontier_id] = (
                        graph.state(0).selected_id
                    )
                graph.clear_selection(0)
            elif step.action == "Widen":
                pass
            elif step.action == "Inspect":
                node = demo.hypotheses[step.created[0]]
                proposal = demo.proposals[step.arguments[0]]
                parent_id = frontiers.get(proposal.frontier_id)
                graph.record_execution_attempt(0)
                created = graph.add_executed(
                    sample_id=0,
                    function_state=node.function_state,
                    target_expression=node.target_expression,
                    sexpr="\n".join(node.function_state),
                    denotation=node.denotation,
                    denotation_labels=dict(node.denotation_labels),
                    parent_id=parent_id,
                    parent_ids=node.parent_ids,
                    operation=node.operation,
                    relation_id=node.relation,
                    provenance=node.provenance,
                )
                if created.node_id != step.created[0]:
                    errors.append(
                        f"step {step_index} Inspect: created {created.node_id}, "
                        f"teacher expects {step.created[0]}"
                    )
                    break
            elif step.action == "Select":
                graph.select(0, step.arguments[0])
            elif step.action == "Park":
                graph.park(0, step.arguments[0])
            elif step.action == "Recall":
                graph.recall(0, step.arguments[0])
            elif step.action == "Prune":
                graph.prune(0, step.arguments[0])
            elif step.action == "Combine":
                node = demo.hypotheses[step.created[0]]
                left, right = graph.combination_parents(0, *step.arguments)
                graph.mark_expanded(0, left.node_id)
                graph.mark_expanded(0, right.node_id)
                graph.record_execution_attempt(0)
                created = graph.add_executed(
                    sample_id=0,
                    function_state=node.function_state,
                    target_expression=node.target_expression,
                    sexpr="\n".join(node.function_state),
                    denotation=node.denotation,
                    denotation_labels=dict(node.denotation_labels),
                    parent_id=node.parent_id,
                    parent_ids=node.parent_ids,
                    operation=node.operation,
                    relation_id=node.relation,
                    provenance=node.provenance,
                )
                if created.node_id != step.created[0]:
                    errors.append(f"step {step_index} Combine: node ID mismatch")
                    break
            elif step.action in {
                "Merge",
                "Order",
                "Compare",
                "Time_constraint",
                "Count",
            }:
                node = demo.hypotheses[step.created[0]]
                parent_id = graph.state(0).selected_id
                if parent_id is not None:
                    graph.mark_expanded(0, parent_id)
                graph.record_execution_attempt(0)
                created = graph.add_executed(
                    sample_id=0,
                    function_state=node.function_state,
                    target_expression=node.target_expression,
                    sexpr="\n".join(node.function_state),
                    denotation=node.denotation,
                    denotation_labels=dict(node.denotation_labels),
                    parent_id=node.parent_id,
                    parent_ids=node.parent_ids,
                    operation=node.operation,
                    relation_id=node.relation,
                    provenance=node.provenance,
                )
                if created.node_id != step.created[0]:
                    errors.append(f"step {step_index} {step.action}: node ID mismatch")
                    break
            elif step.action == "Commit":
                graph.commit(0, step.arguments[0])
            else:
                errors.append(f"step {step_index}: unsupported action {step.action}")
                break
        except (KeyError, RuntimeError, ValueError) as exc:
            errors.append(f"step {step_index} {step.action}: {exc}")
            break
    return errors


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _augment_recall(demo: HyperDemonstration) -> tuple[HyperDemonstration, bool]:
    """Park the preserved branch during a real probe, then Recall it."""
    steps = list(demo.steps)
    max_turns = int(demo.private_metadata.get("max_turns", 32))
    if len(steps) + 2 > max_turns:
        return demo, False

    for start, step in enumerate(steps):
        if step.action != "Select" or step.supervision != "intervention":
            continue
        resolution_seen = False
        resume = None
        resume_index = None
        for index in range(start + 1, len(steps)):
            candidate = steps[index]
            resolution_seen = resolution_seen or candidate.action in {"Park", "Prune"}
            if resolution_seen and candidate.action == "Select":
                resume = candidate.arguments[0]
                resume_index = index
                break
            if candidate.action in {"Commit", "Abstain"}:
                break
        if resume is None or resume_index is None:
            continue
        if resume == step.arguments[0] or resume not in step.visible_before:
            continue

        parked_state = tuple(value for value in step.visible_before if value != resume)
        rewritten = steps[:start]
        rewritten.append(
            DemonstrationStep(
                "Park",
                (resume,),
                step.visible_before,
                (),
                (
                    "preserve_required_branch_during_probe",
                    "storage_only_not_semantic_rejection",
                ),
            )
        )
        for index in range(start, resume_index):
            current = steps[index]
            rewritten.append(
                replace(
                    current,
                    visible_before=tuple(
                        value for value in current.visible_before if value != resume
                    ),
                )
            )
        resume_step = steps[resume_index]
        recall_before = tuple(
            value for value in resume_step.visible_before if value != resume
        )
        rewritten.append(
            DemonstrationStep(
                "Recall",
                (resume,),
                recall_before,
                (),
                ("restore_preserved_branch_after_probe",),
            )
        )
        rewritten.extend(steps[resume_index:])
        metadata = dict(demo.private_metadata)
        metadata["recall_augmented"] = True
        return replace(demo, steps=rewritten, private_metadata=metadata), True
    return demo, False


def _delayed_decoy_comparison(demo: HyperDemonstration) -> HyperDemonstration | None:
    """Move one unused decoy Inspect immediately before the gold Commit.

    The original committed hypothesis retains its answer-and-intent
    certificate. Delaying an independent active leaf makes that certified
    hypothesis older than a freshly executed plausible alternative without
    inventing a denotation or changing any graph operation.
    """
    if len(demo.steps) < 3 or demo.steps[-1].action != "Commit":
        return None
    commit = demo.steps[-1]
    target = commit.arguments[0]
    creators = {
        node_id: (index, step)
        for index, step in enumerate(demo.steps[:-1])
        for node_id in step.created
    }
    movable = []
    for node_id in commit.visible_before:
        if node_id == target or node_id not in creators:
            continue
        creator_index, creator = creators[node_id]
        if creator.action != "Inspect" or len(creator.created) != 1:
            continue
        if any(node_id in step.arguments for step in demo.steps[creator_index + 1 : -1]):
            continue
        if any(
            node_id == node.parent_id or node_id in node.parent_ids
            for candidate_id, node in demo.hypotheses.items()
            if candidate_id != node_id
        ):
            continue
        if creator.arguments[0] not in demo.proposals:
            continue
        movable.append((creator_index, node_id, creator))
    if not movable:
        return None

    digest = int(hashlib.sha256((demo.demo_id + ":comparison").encode()).hexdigest()[:16], 16)
    creator_index, decoy_id, creator = movable[digest % len(movable)]
    rewritten = []
    for index, step in enumerate(demo.steps[:-1]):
        if index == creator_index:
            continue
        visible = (
            tuple(value for value in step.visible_before if value != decoy_id)
            if index > creator_index
            else step.visible_before
        )
        rewritten.append(replace(step, visible_before=visible))
    delayed_visible = tuple(
        value for value in commit.visible_before if value != decoy_id
    )
    rewritten.append(replace(creator, visible_before=delayed_visible))
    rewritten.append(commit)
    metadata = dict(demo.private_metadata)
    metadata.update(
        {
            "comparison_variant": True,
            "comparison_decoy": decoy_id,
            "terminal_decision_start": len(rewritten) - 1,
        }
    )
    return _runtime_ordered_demo(
        replace(
            demo,
            demo_id=f"{demo.demo_id}:delayed-decoy",
            steps=rewritten,
            private_metadata=metadata,
        )
    )


def _deadline_variant(demo: HyperDemonstration) -> HyperDemonstration | None:
    """Create a time-shifted deadline state with an offline F1 teacher."""
    if len(demo.steps) < 4 or demo.steps[-1].action != "Commit":
        return None
    digest = int(hashlib.sha256(demo.demo_id.encode()).hexdigest()[:16], 16)
    if digest % 1000 >= 80:
        return None

    # The deadline is sampled without looking at gold or candidate quality.
    cutoff = 1 + (digest // 1000) % (len(demo.steps) - 1)
    active = tuple(demo.steps[cutoff].visible_before)
    parked: set[str] = set()
    for step in demo.steps[:cutoff]:
        if step.action == "Park":
            parked.add(step.arguments[0])
        elif step.action == "Recall":
            parked.discard(step.arguments[0])
    available = tuple(active) + tuple(
        node_id for node_id in demo.hypotheses if node_id in parked
    )
    candidates = [
        demo.hypotheses[node_id]
        for node_id in available
        if node_id in demo.hypotheses and demo.hypotheses[node_id].denotation
    ]
    if not candidates:
        return None
    chosen = max(
        candidates,
        key=lambda node: (
            answer_set_f1(node.denotation, demo.gold_answers),
            node.depth,
            -int(node.hypothesis_id[1:]),
        ),
    )
    chosen_f1 = answer_set_f1(chosen.denotation, demo.gold_answers)
    prefix = list(demo.steps[:cutoff])
    deadline_recalled_best = chosen.hypothesis_id in parked
    if deadline_recalled_best:
        if len(active) >= int(demo.private_metadata.get("max_active", 24)):
            return None
        prefix.append(
            DemonstrationStep(
                "Recall",
                (chosen.hypothesis_id,),
                active,
                (),
                ("restore_best_available_candidate_at_deadline",),
            )
        )
        order = {node_id: index for index, node_id in enumerate(demo.hypotheses)}
        active = tuple(sorted((*active, chosen.hypothesis_id), key=order.__getitem__))
    prefix.append(
        DemonstrationStep(
            "Commit",
            (chosen.hypothesis_id,),
            active,
            (),
            (
                "deadline_reached",
                "best_attainable_visible_candidate",
                f"answer_f1:{chosen_f1:.8f}",
            ),
            certificate_kind="best_attainable_answer_f1",
            certificate_evidence=(f"answer_f1:{chosen_f1:.8f}",),
        )
    )
    known = set(demo.steps[0].visible_before if demo.steps else ())
    for step in prefix:
        known.update(step.created)
    max_turns = int(demo.private_metadata.get("max_turns", 32))
    if len(prefix) > max_turns:
        return None
    metadata = dict(demo.private_metadata)
    metadata.update(
        {
            "best_attainable_supervision": True,
            "deadline_cutoff_sampled_without_gold": True,
            "deadline_recalled_best": deadline_recalled_best,
            "terminal_answer_f1": chosen_f1,
            "max_turns": max_turns,
            "turn_offset": max_turns - len(prefix),
            "terminal_decision_start": len(prefix) - (2 if deadline_recalled_best else 1),
            "execution_attempts": sum(
                step.action in _EXECUTING_ACTIONS for step in prefix
            ),
        }
    )
    return replace(
        demo,
        demo_id=f"{demo.demo_id}:deadline:{cutoff}",
        hypotheses={
            node_id: node
            for node_id, node in demo.hypotheses.items()
            if node_id in known
        },
        steps=prefix,
        private_metadata=metadata,
    )


def _target_action(row: dict) -> str:
    content = str(row["messages"][-1]["content"])
    match = re.search(r"<action>\s*([^\s\[]+)", content)
    if match is None:
        raise ValueError("decision row has no target action")
    return match.group(1)


def _rendered_clock(row: dict) -> tuple[int, int, int] | None:
    """Return the latest public (used, maximum, remaining) clock."""
    clocks = []
    for message in row.get("messages", ()):
        if message.get("role") != "user":
            continue
        clocks.extend(
            tuple(map(int, match))
            for match in re.findall(
                r"turns_used=(\d+)/(\d+)\s+turns_remaining=(\d+)",
                str(message.get("content", "")),
            )
        )
    return clocks[-1] if clocks else None


def _commit_recency(row: dict) -> tuple[bool, bool] | None:
    """Return (eligible, target_is_newest) for a rendered Commit decision."""
    target = re.search(
        r"<action>\s*Commit\s*\[\s*(H\d+)\s*\]",
        str(row["messages"][-1].get("content", "")),
    )
    if target is None or len(row.get("messages", ())) < 2:
        return None
    observation = str(row["messages"][-2].get("content", ""))
    active = [
        int(value)
        for value in re.findall(r"(?m)^H(\d+) \[active\]", observation)
    ]
    if len(active) < 2:
        return (False, False)
    target_index = int(target.group(1)[1:])
    return (True, target_index == max(active))


def _observation_state(observation: str) -> str:
    start = observation.find("<hypothesis_graph>")
    stop = observation.rfind("</information>")
    if start < 0 or stop < start:
        raise ValueError("environment observation has no rendered hypothesis state")
    return observation[start:stop].rstrip()


def _rendered_runtime_identity_is_valid(row: dict) -> bool:
    """Reject sparse/impossible node IDs and observations without a clock."""
    for message in row.get("messages", ()):
        if message.get("role") != "user":
            continue
        content = str(message.get("content", ""))
        if "<hypothesis_graph>" not in content:
            continue
        graph = _observation_state(content)
        header = re.search(
            r"stored=(\d+).*turns_used=(\d+)/(\d+).*turns_remaining=(\d+)",
            graph,
        )
        if header is None:
            return False
        stored, used, maximum, remaining = map(int, header.groups())
        if used > maximum or remaining != maximum - used:
            return False
        ids = [
            int(value)
            for value in re.findall(r"(?<![A-Za-z0-9_-])H(\d+)\b", graph)
        ]
        if ids and max(ids) >= stored:
            return False
    return True


def _invalid_recovery_specs(row: dict) -> list[tuple[str, str, str]]:
    """Return mistakes whose rejection follows from the visible live state."""
    messages = row.get("messages", ())
    if len(messages) < 2 or messages[-2].get("role") != "user":
        return []
    observation = str(messages[-2].get("content", ""))
    if "<hypothesis_graph>" not in observation:
        return []
    state = _observation_state(observation)
    specs: list[tuple[str, str, str]] = []
    selected = re.search(r"\bselected=(H\d+)\b", state)
    if selected:
        node_id = selected.group(1)
        specs.append(
            (
                "repeat_select",
                f"Select [ {node_id} ]",
                f"hypothesis {node_id} is already selected; choose a progress action",
            )
        )
    active = re.findall(
        r"(?m)^(H\d+) \[active\].*? answers=(\d+):", state
    )
    if active:
        node_id = active[0][0]
        specs.append(
            (
                "recall_active",
                f"Recall [ {node_id} ]",
                f"hypothesis {node_id} is active, not parked",
            )
        )
    empty = next((node_id for node_id, size in active if int(size) == 0), None)
    if empty:
        specs.append(
            (
                "commit_empty",
                f"Commit [ {empty} ]",
                "cannot commit an empty hypothesis",
            )
        )
    parked = re.search(r"(?ms)<parked_hypotheses>.*?^(H\d+) path=", state)
    if parked:
        node_id = parked.group(1)
        specs.append(
            (
                "select_parked",
                f"Select [ {node_id} ]",
                f"hypothesis {node_id} is parked, not active",
            )
        )
    budget = re.search(
        r"active=(\d+) capacity=(\d+) stored=(\d+).*?"
        r"execution_attempts=(\d+)/(\d+)",
        state,
    )
    selected_id = re.search(r"\bselected=(\S+)\b", state)
    if budget:
        active_count, capacity, _, used, maximum = map(int, budget.groups())
        for catalog in re.findall(
            r"(?s)<proposal_catalog>.*?</proposal_catalog>", state
        ):
            header = re.search(r"source=(\S+) exposed=(\d+)/(\d+)", catalog)
            if (
                header
                and int(header.group(2)) >= int(header.group(3))
                and active_count < capacity
                and used < maximum
                and selected_id
                and selected_id.group(1) == "none"
            ):
                source = header.group(1)
                specs.append(
                    (
                        "widen_exhausted",
                        f"Widen [ {source} ]",
                        f"no open frontier from {source} has further ranked proposals",
                    )
                )
                break
    visible_proposal = re.search(r"(?m)^(P\d+) rank=", state)
    if budget and visible_proposal:
        active_count, capacity, _, used, maximum = map(int, budget.groups())
        proposal_id = visible_proposal.group(1)
        if active_count >= capacity:
            specs.append(
                (
                    "inspect_full_workspace",
                    f"Inspect [ {proposal_id} ]",
                    "the visible hypothesis workspace is full; Park an unresolved hypothesis first",
                )
            )
        if used >= maximum:
            specs.append(
                (
                    "inspect_execution_exhausted",
                    f"Inspect [ {proposal_id} ]",
                    "the execution budget is exhausted; select or recall the strongest supported nonempty hypothesis and Commit it",
                )
            )
    return specs


def _invalid_recovery_record(
    row: dict,
    spec: tuple[str, str, str],
    ordinal: int,
) -> dict:
    """Show the live Markov failure state and supervise one legal recovery."""
    messages = [dict(message) for message in row["messages"]]
    recovery_kind, invalid_action, failure = spec
    observation = str(messages[-2]["content"])
    state = _observation_state(observation)
    graph, separator, page_state = state.partition("\n<proposal_catalog>")
    if separator:
        page_state = "<proposal_catalog>" + page_state
    failed_observation = render_hyper_information(
        f"Graph action failed: {failure}", graph, page_state
    )
    messages = [
        dict(messages[0]),
        {"role": "assistant", "content": "", "loss_mask": 0},
        {"role": "user", "content": failed_observation, "loss_mask": 0},
        dict(messages[-1]),
    ]
    extra = dict(row.get("extra_info", {}))
    extra.update(
        {
            "invalid_recovery": True,
            "invalid_recovery_ordinal": ordinal,
            "invalid_recovery_kind": recovery_kind,
            "invalid_action": invalid_action,
            "invalid_failure": failure,
        }
    )
    return {**row, "messages": messages, "extra_info": extra}


def _supports_invalid_recovery(row: dict) -> bool:
    return bool(_invalid_recovery_specs(row))


def _decision_contradictions(rows: list[dict]) -> dict:
    state_actions: dict[str, set[str]] = {}
    for row in rows:
        history = []
        for message in row["messages"]:
            content = str(message.get("content", ""))
            if (
                message.get("role") == "assistant"
                and message.get("loss_mask") == 1
                and "<action>" in content
            ):
                key = json.dumps(history, sort_keys=True, ensure_ascii=True)
                state_actions.setdefault(key, set()).add(content)
            history.append(message)
    conflicts = sum(len(actions) > 1 for actions in state_actions.values())
    return {
        "decision_states": len(state_actions),
        "contradictory_states": conflicts,
        "contradiction_rate": conflicts / len(state_actions) if state_actions else 0.0,
    }


def _normalized_decision_row(row: dict) -> dict:
    extra = dict(row.get("extra_info", {}))
    extra.setdefault("recovery_stratum", None)
    extra.setdefault("invalid_recovery", False)
    extra.setdefault("invalid_recovery_ordinal", -1)
    extra.setdefault("invalid_recovery_kind", None)
    extra.setdefault("invalid_action", None)
    extra.setdefault("invalid_failure", None)
    return {**row, "extra_info": extra}


def _decision_schema():
    import pyarrow as pa

    return pa.schema(
        [
            pa.field(
                "messages",
                pa.list_(
                    pa.struct(
                        [
                            pa.field("role", pa.string()),
                            pa.field("content", pa.string()),
                            pa.field("loss_mask", pa.int64()),
                        ]
                    )
                ),
            ),
            pa.field("data_source", pa.string()),
            pa.field(
                "extra_info",
                pa.struct(
                    [
                        pa.field("demo_id", pa.string()),
                        pa.field("question_id", pa.string()),
                        pa.field("family", pa.string()),
                        pa.field("recovery_stratum", pa.string()),
                        pa.field("replay_verified", pa.bool_()),
                        pa.field("gold_injected_into_proposals", pa.bool_()),
                        pa.field("terminal_action", pa.string()),
                        pa.field("decision_index", pa.int64()),
                        pa.field("trajectory_step_index", pa.int64()),
                        pa.field("target_is_graph_action", pa.bool_()),
                        pa.field("invalid_recovery", pa.bool_()),
                        pa.field("invalid_recovery_ordinal", pa.int64()),
                        pa.field("invalid_recovery_kind", pa.string()),
                        pa.field("invalid_action", pa.string()),
                        pa.field("invalid_failure", pa.string()),
                    ]
                ),
            ),
        ]
    )


class _ParquetSink:
    """Write bounded compressed row groups and publish only a complete file."""

    def __init__(self, path: Path, batch_size: int = 512):
        import pyarrow.parquet as pq

        self.path = path
        self.temporary = path.with_suffix(path.suffix + ".tmp")
        self.temporary.unlink(missing_ok=True)
        self.schema = _decision_schema()
        self.writer = pq.ParquetWriter(
            self.temporary,
            self.schema,
            compression="snappy",
            use_dictionary=True,
        )
        self.batch_size = int(batch_size)
        self.rows: list[dict] = []
        self.count = 0

    def append(self, row: dict) -> None:
        self.rows.append(_normalized_decision_row(row))
        self.count += 1
        if len(self.rows) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        import pyarrow as pa

        self.writer.write_table(pa.Table.from_pylist(self.rows, schema=self.schema))
        self.rows.clear()

    def close(self) -> None:
        self.flush()
        self.writer.close()
        os.replace(self.temporary, self.path)

    def abort(self) -> None:
        self.writer.close()
        self.temporary.unlink(missing_ok=True)


class _DecisionConsistency:
    def __init__(self):
        self.actions: dict[bytes, str] = {}
        self.conflicts: set[bytes] = set()

    def add(self, row: dict) -> None:
        messages = row["messages"]
        if not messages or messages[-1].get("loss_mask") != 1:
            raise ValueError("decision row must end in exactly one trained action")
        state = json.dumps(messages[:-1], sort_keys=True, ensure_ascii=True)
        key = hashlib.sha256(state.encode()).digest()
        action = str(messages[-1]["content"])
        previous = self.actions.setdefault(key, action)
        if previous != action:
            self.conflicts.add(key)

    def report(self) -> dict:
        states = len(self.actions)
        conflicts = len(self.conflicts)
        return {
            "decision_states": states,
            "contradictory_states": conflicts,
            "contradiction_rate": conflicts / states if states else 0.0,
        }


def _valid_recall_augmentation(demo: HyperDemonstration) -> bool:
    parked = set()
    recalls = 0
    for step in demo.steps:
        if step.action == "Park":
            parked.add(step.arguments[0])
        elif step.action == "Recall":
            node_id = step.arguments[0]
            if node_id not in parked or node_id not in demo.hypotheses:
                return False
            parked.remove(node_id)
            recalls += 1
    return recalls == 1


def _valid_invalid_recovery(row: dict) -> bool:
    messages = row["messages"]
    extra = row.get("extra_info", {})
    failed = str(messages[2].get("content", "")) if len(messages) == 4 else ""
    kind = extra.get("invalid_recovery_kind")
    supported = {
        "repeat_select",
        "recall_active",
        "commit_empty",
        "select_parked",
        "widen_exhausted",
        "inspect_full_workspace",
        "inspect_execution_exhausted",
    }
    return (
        len(messages) == 4
        and messages[0].get("role") == "user"
        and messages[1] == {"role": "assistant", "content": "", "loss_mask": 0}
        and kind in supported
        and messages[2].get("role") == "user"
        and messages[2].get("loss_mask") == 0
        and failed.startswith("<information>\nGraph action failed:")
        and "\n<hypothesis_graph>" in failed
        and messages[3].get("role") == "assistant"
        and messages[3].get("loss_mask") == 1
        and extra.get("invalid_action")
        and extra.get("invalid_failure")
        and "P999999" not in json.dumps(row)
        and "H999999" not in json.dumps(row)
    )


def _regenerate_unlocked(input_path: Path, output: Path) -> dict:
    source_contract = _load_source_contract(input_path)
    source_demonstrations = 0
    action_counts: Counter = Counter()
    for demo in _iter_demos(input_path):
        source_demonstrations += 1
        action_counts.update(step.action for step in demo.steps)
    rare_action_target = max(action_counts["Widen"], action_counts["Prune"], 1)
    output.mkdir(parents=True, exist_ok=True)
    demo_path = output / "demonstrations.jsonl"
    demo_temporary = demo_path.with_suffix(demo_path.suffix + ".tmp")
    demo_temporary.unlink(missing_ok=True)
    train_sink = _ParquetSink(output / "train_decision.parquet")
    validation_sink = _ParquetSink(output / "validation_decision.parquet")
    consistency = _DecisionConsistency()
    recovery_candidates: dict[
        str, list[tuple[dict, tuple[str, str, str]]]
    ] = defaultdict(list)
    train_questions = set()
    validation_questions = set()
    final_actions: Counter = Counter()
    training_actions: Counter = Counter()
    counters: Counter = Counter()
    recall_valid = True
    runtime_order_valid = True
    rendered_runtime_identity_valid = True
    runtime_replay_valid = True
    runtime_replay_failures: list[dict] = []
    clock_contract_valid = True
    clocked_decisions = 0
    near_deadline_decisions = 0
    deadline_decisions = 0
    deadline_near_horizon = 0
    comparison_decisions = 0
    commit_recency: Counter = Counter()
    comparison_commit_recency: Counter = Counter()

    def write_decision(row: dict, *, collect_recovery: bool) -> None:
        nonlocal rendered_runtime_identity_valid, clock_contract_valid
        nonlocal clocked_decisions, near_deadline_decisions
        nonlocal deadline_decisions, deadline_near_horizon, comparison_decisions
        rendered_runtime_identity_valid = (
            rendered_runtime_identity_valid
            and _rendered_runtime_identity_is_valid(row)
        )
        demo_id = str(row["extra_info"]["demo_id"])
        is_invalid_recovery = bool(
            row.get("extra_info", {}).get("invalid_recovery")
        )
        is_comparison_primary = (
            ":delayed-decoy" in demo_id and not is_invalid_recovery
        )
        clock = _rendered_clock(row)
        if clock is not None:
            used, maximum, remaining = clock
            clocked_decisions += 1
            near_deadline_decisions += int(remaining <= 3)
            clock_contract_valid = (
                clock_contract_valid
                and maximum == int(source_contract["max_turns"])
                and remaining == maximum - used
            )
            if ":deadline:" in demo_id:
                deadline_decisions += 1
                deadline_near_horizon += int(remaining <= 3)
        if is_comparison_primary:
            comparison_decisions += 1
        recency = _commit_recency(row)
        if recency is not None and recency[0]:
            commit_recency["eligible"] += 1
            commit_recency["newest"] += int(recency[1])
            if is_comparison_primary:
                comparison_commit_recency["eligible"] += 1
                comparison_commit_recency["newest"] += int(recency[1])
        split = _question_split(row["extra_info"]["question_id"])
        (train_sink if split == "train" else validation_sink).append(row)
        training_actions[_target_action(row)] += 1
        consistency.add(row)
        if collect_recovery:
            for spec in _invalid_recovery_specs(row):
                candidates = recovery_candidates[spec[0]]
                if len(candidates) < rare_action_target:
                    candidates.append((row, spec))

    try:
        with demo_temporary.open("w", encoding="utf-8") as demo_handle:
            for source_demo in _iter_demos(input_path):
                source_demo = _runtime_ordered_demo(source_demo)
                updated, augmented = _augment_recall(source_demo)
                variants = [updated]
                comparison = _delayed_decoy_comparison(updated)
                if comparison is not None:
                    variants.append(comparison)
                    counters["comparison"] += 1
                deadline = _deadline_variant(updated)
                if deadline is not None:
                    variants.append(deadline)
                    counters["deadline"] += 1
                    counters["deadline_recalled_best"] += int(
                        deadline.private_metadata.get("deadline_recalled_best", False)
                    )
                    counters[
                        "deadline_exact"
                        if float(deadline.private_metadata["terminal_answer_f1"]) == 1.0
                        else "deadline_partial"
                        if float(deadline.private_metadata["terminal_answer_f1"]) > 0.0
                        else "deadline_zero"
                    ] += 1
                counters["recall"] += int(augmented)
                if augmented:
                    recall_valid = recall_valid and _valid_recall_augmentation(updated)

                for demo in variants:
                    replay_errors = _runtime_replay_errors(demo)
                    runtime_replay_valid = runtime_replay_valid and not replay_errors
                    if replay_errors and len(runtime_replay_failures) < 20:
                        runtime_replay_failures.append(
                            {"demo_id": demo.demo_id, "errors": replay_errors}
                        )
                    runtime_order_valid = (
                        runtime_order_valid
                        and _hypotheses_follow_runtime_order(demo)
                    )
                    split = _question_split(demo.question_id)
                    (train_questions if split == "train" else validation_questions).add(
                        demo.question_id
                    )
                    counters[f"{split}_demonstrations"] += 1
                    final_actions.update(step.action for step in demo.steps)
                    demo_handle.write(
                        json.dumps(demo.to_dict(), ensure_ascii=False) + "\n"
                    )
                    rows = decision_sft_records(demo)
                    terminal_start = demo.private_metadata.get(
                        "terminal_decision_start"
                    )
                    if terminal_start is not None:
                        rows = [
                            row
                            for row in rows
                            if int(row["extra_info"]["trajectory_step_index"])
                            >= int(terminal_start)
                        ]
                    for row in rows:
                        write_decision(row, collect_recovery=True)

        selected: list[tuple[dict, tuple[str, str, str]]] = []
        recovery_kinds = sorted(recovery_candidates)
        recovery_offsets = {kind: 0 for kind in recovery_kinds}
        while len(selected) < rare_action_target and any(recovery_candidates.values()):
            made_progress = False
            for kind in recovery_kinds:
                offset = recovery_offsets[kind]
                candidates = recovery_candidates[kind]
                if offset < len(candidates) and len(selected) < rare_action_target:
                    selected.append(candidates[offset])
                    recovery_offsets[kind] = offset + 1
                    made_progress = True
            if not made_progress:
                break
        invalid_rows = [
            _invalid_recovery_record(row, spec, index)
            for index, (row, spec) in enumerate(selected)
        ]
        invalid_count_complete = len(invalid_rows) == rare_action_target
        invalid_masks_valid = all(
            _valid_invalid_recovery(row) for row in invalid_rows
        )
        invalid_recovery_kinds = Counter(
            row["extra_info"]["invalid_recovery_kind"] for row in invalid_rows
        )
        for row in invalid_rows:
            write_decision(row, collect_recovery=False)

        consistency_report = consistency.report()
        assertions = {
            "regenerated_from_structured_demonstrations": source_demonstrations > 0,
            "invalid_recovery_count_reached": invalid_count_complete,
            "invalid_actions_have_zero_loss": invalid_masks_valid,
            "recall_restores_a_parked_executable_hypothesis": recall_valid,
            "no_abstain_supervision_under_f1_objective": final_actions["Abstain"] == 0,
            "runtime_aligned_invalid_recovery_observations": invalid_masks_valid,
            "hypotheses_follow_runtime_creation_order": runtime_order_valid,
            "rendered_ids_and_clock_match_runtime": rendered_runtime_identity_valid,
            "all_rendered_clocks_use_source_horizon": clock_contract_valid,
            "deadline_states_reach_the_live_horizon": (
                source_demonstrations < 100
                or deadline_decisions > 0
                and deadline_near_horizon == deadline_decisions
            ),
            "all_teacher_actions_replay_in_live_graph": runtime_replay_valid,
            "deadline_commit_supervision_present": (
                source_demonstrations < 100 or counters["deadline"] > 0
            ),
            "invalid_recovery_uses_only_runtime_visible_ids": all(
                "P999999" not in json.dumps(row)
                and "H999999" not in json.dumps(row)
                for row in invalid_rows
            ),
            "invalid_recovery_has_multiple_real_failure_modes": (
                source_demonstrations < 100 or len(invalid_recovery_kinds) >= 3
            ),
            "comparison_variants_train_only_the_terminal_decision": (
                comparison_decisions == counters["comparison"]
            ),
            "comparison_commits_break_the_recency_shortcut": (
                source_demonstrations < 100
                or comparison_commit_recency["eligible"] > 0
                and comparison_commit_recency["newest"] == 0
            ),
            "question_disjoint_split": not train_questions.intersection(
                validation_questions
            ),
        }
        if not all(assertions.values()) or consistency_report["contradictory_states"]:
            raise RuntimeError(
                "generated corpus failed a truthfulness invariant: "
                f"assertions={assertions}, consistency={consistency_report}"
            )

        train_sink.close()
        validation_sink.close()
        os.replace(demo_temporary, demo_path)
    except Exception:
        train_sink.abort()
        validation_sink.abort()
        demo_temporary.unlink(missing_ok=True)
        raise

    report = {
        "quality_schema": "hyper_r1_v23_runtime_comparison",
        "source": str(input_path),
        "source_contract": source_contract,
        "output": str(output),
        "source_demonstrations": source_demonstrations,
        "output_demonstrations": (
            source_demonstrations + counters["comparison"] + counters["deadline"]
        ),
        "delayed_decoy_comparisons": counters["comparison"],
        "recall_augmented_demonstrations": counters["recall"],
        "deadline_commit_demonstrations": counters["deadline"],
        "deadline_commit_exact": counters["deadline_exact"],
        "deadline_commit_partial": counters["deadline_partial"],
        "deadline_commit_zero": counters["deadline_zero"],
        "deadline_recalled_best": counters["deadline_recalled_best"],
        "masked_invalid_recovery_decisions": len(invalid_rows),
        "invalid_recovery_kinds": dict(sorted(invalid_recovery_kinds.items())),
        "clock_audit": {
            "clocked_decisions": clocked_decisions,
            "near_deadline_decisions": near_deadline_decisions,
            "deadline_decisions": deadline_decisions,
            "deadline_near_horizon": deadline_near_horizon,
        },
        "commit_recency": {
            "eligible": commit_recency["eligible"],
            "newest": commit_recency["newest"],
            "newest_rate": (
                commit_recency["newest"] / commit_recency["eligible"]
                if commit_recency["eligible"]
                else None
            ),
            "comparison_eligible": comparison_commit_recency["eligible"],
            "comparison_newest": comparison_commit_recency["newest"],
        },
        "rare_action_reference_count": rare_action_target,
        "demonstration_action_counts": dict(sorted(final_actions.items())),
        "action_counts": dict(sorted(training_actions.items())),
        "train_demonstrations": counters["train_demonstrations"],
        "validation_demonstrations": counters["validation_demonstrations"],
        "train_decisions": train_sink.count,
        "validation_decisions": validation_sink.count,
        "decision_consistency": consistency_report,
        "runtime_replay_failures": runtime_replay_failures,
        "assertions": assertions,
    }
    report_path = output / "report.json"
    report_temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    report_temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(report_temporary, report_path)
    return report


def regenerate(input_path: Path, output: Path) -> dict:
    """Regenerate under an exclusive lock so concurrent exports cannot mix."""
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{output.name}.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"another corpus export owns {lock_path}") from exc
    try:
        os.write(lock_fd, f"pid={os.getpid()}\n".encode())
        return _regenerate_unlocked(input_path, output)
    finally:
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(regenerate(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
