#!/usr/bin/env python3
"""Add explicit live-action targets to an existing HyPER-R1 SFT corpus."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd


_GRAPH_RE = re.compile(r"<hypothesis_graph>.*?</hypothesis_graph>", re.DOTALL)
_CATALOG_RE = re.compile(r"<proposal_catalog>.*?</proposal_catalog>", re.DOTALL)
_ACTIVE_RE = re.compile(r"(?m)^(H\d+) \[active\].*? answers=(\d+):")
_PARKED_RE = re.compile(r"(?m)^(H\d+) path=.*? answers=\d+$")
_PROPOSAL_RE = re.compile(r"(?m)^(P\d+) .*? status=visible$")
_CATALOG_HEADER_RE = re.compile(
    r"source=(\S+) exposed=(\d+)/(\d+) page_size=\d+"
)
_INSTRUCTION_ANCHOR = (
    "- Hypothesis IDs and execution results are owned by the environment. "
    "Never invent or edit them."
)
_INSTRUCTION = (
    "- Every observation lists current action targets. Never reuse a hypothesis "
    "or proposal ID that is absent from the corresponding target list, even if "
    "it appeared earlier in the conversation."
)
_OLD_COMMIT_INSTRUCTION = (
    "- Use `Commit [ Hn ]` when one hypothesis expresses the complete question. "
    "After the environment confirms it, return its values inside <answer>."
)
_NEW_COMMIT_INSTRUCTION = (
    "- Use `Commit [ Hn ]` when one hypothesis expresses the complete question. "
    "Commit is terminal; the environment returns that hypothesis's values."
)
_OLD_CLOSING_INSTRUCTION = (
    "Preserve plausible alternatives until later execution distinguishes them. "
    "Select is not Commit: selecting one hypothesis for expansion does not reject "
    "the others. After Commit, perform no more graph actions and copy the committed "
    "values exactly into <answer>."
)
_NEW_CLOSING_INSTRUCTION = (
    "Preserve plausible alternatives until later execution distinguishes them. "
    "Select is not Commit: selecting one hypothesis for expansion does not reject "
    "the others. Commit ends the search."
)


def _show(values) -> str:
    return ",".join(values) if values else "none"


def add_action_affordances(text: str) -> Tuple[str, int, int]:
    """Enrich public observations without consulting target actions or gold data."""
    value = str(text)
    graph_updates = 0
    catalog_updates = 0
    parked = _PARKED_RE.findall(value)

    def graph_replacement(match: re.Match[str]) -> str:
        nonlocal graph_updates
        block = match.group(0)
        if "Available targets:" in block:
            return block
        active_rows = _ACTIVE_RE.findall(block)
        active = [node_id for node_id, _ in active_rows]
        commit = [node_id for node_id, count in active_rows if int(count) > 0]
        line = (
            "Available targets: "
            f"Select/Park=[{_show(active)}]; "
            f"Commit(nonempty active)=[{_show(commit)}]; "
            f"Combine/Prune candidates=[{_show(active)}]; "
            f"Recall=[{_show(parked)}]. "
            "Do not use an ID absent from its list; Prune still requires a public contradiction."
        )
        anchor = "Actions: Select, Find_relation [ source ]"
        if anchor not in block:
            raise ValueError("hypothesis graph is missing its action anchor")
        graph_updates += 1
        return block.replace(anchor, line + "\n" + anchor, 1)

    value = _GRAPH_RE.sub(graph_replacement, value)

    def catalog_replacement(match: re.Match[str]) -> str:
        nonlocal catalog_updates
        block = match.group(0)
        if "Available proposal targets:" in block:
            return block
        header = _CATALOG_HEADER_RE.search(block)
        if header is None:
            raise ValueError("proposal catalog is missing source/exposure metadata")
        source, exposed, total = header.groups()
        proposals = _PROPOSAL_RE.findall(block)
        widen = source if int(exposed) < int(total) else "none"
        line = (
            "Available proposal targets: "
            f"Inspect=[{_show(proposals)}]; Widen=[{widen}]."
        )
        anchor = "Use Inspect [ Pn ]"
        if anchor not in block:
            raise ValueError("proposal catalog is missing its instruction anchor")
        catalog_updates += 1
        return block.replace(anchor, line + "\n" + anchor, 1)

    value = _CATALOG_RE.sub(catalog_replacement, value)
    value = value.replace(_OLD_COMMIT_INSTRUCTION, _NEW_COMMIT_INSTRUCTION)
    value = value.replace(_OLD_CLOSING_INSTRUCTION, _NEW_CLOSING_INSTRUCTION)
    if _INSTRUCTION not in value and _INSTRUCTION_ANCHOR in value:
        value = value.replace(
            _INSTRUCTION_ANCHOR,
            _INSTRUCTION_ANCHOR + "\n" + _INSTRUCTION,
            1,
        )
    return value, graph_updates, catalog_updates


def migrate(input_path: Path, output_path: Path) -> Dict[str, int]:
    frame = pd.read_parquet(input_path)
    rows_changed = 0
    graphs = 0
    catalogs = 0
    migrated_messages = []
    for messages in frame["messages"]:
        updated = []
        row_changed = False
        for message in messages:
            item: Dict[str, Any] = dict(message)
            if item.get("role") == "user":
                content, graph_count, catalog_count = add_action_affordances(
                    item.get("content", "")
                )
                item["content"] = content
                graphs += graph_count
                catalogs += catalog_count
                row_changed = row_changed or bool(graph_count or catalog_count)
            updated.append(item)
        migrated_messages.append(updated)
        rows_changed += int(row_changed)
    frame = frame.copy()
    frame["messages"] = migrated_messages
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False, row_group_size=1000)
    return {
        "rows": len(frame),
        "rows_changed": rows_changed,
        "graphs_enriched": graphs,
        "catalogs_enriched": catalogs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(migrate(args.input, args.output))


if __name__ == "__main__":
    main()
