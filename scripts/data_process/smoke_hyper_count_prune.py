#!/usr/bin/env python3
"""Exercise Count-aware HyPER transitions against the configured Freebase."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from kbqa_r1.hyper_r1 import HypothesisGraph


def _is_zero(value: Any) -> bool:
    lexical = str(value).split("^^", 1)[0].strip().strip('"')
    try:
        return Decimal(lexical) == 0
    except InvalidOperation:
        return False


def run_smoke(
    executor: Callable[[Sequence[str], str], Iterable[str]],
    *,
    root_mid: str = "m.hyper_r1_intentionally_missing",
    relation: str = "type.object.type",
) -> dict:
    """Verify real empty execution, Count zero, and both Prune outcomes."""
    empty_state = (
        f"expression1 = START('{root_mid}')",
        f"expression1 = JOIN('{relation}', expression1)",
    )
    empty_values = tuple(str(value) for value in executor(empty_state, "expression1"))
    if empty_values:
        raise RuntimeError(
            "smoke fixture unexpectedly returned values; choose an absent root MID"
        )

    count_state = (*empty_state, "expression1 = COUNT(expression1)")
    count_values = tuple(str(value) for value in executor(count_state, "expression1"))
    if not count_values or not any(_is_zero(value) for value in count_values):
        raise RuntimeError(f"COUNT(empty) did not return zero: {count_values!r}")

    count_graph = HypothesisGraph(max_active=3, max_nodes=6)
    count_graph.register_public_question(
        0, "How many types belong to the deliberately absent entity?"
    )
    empty = count_graph.add_executed(
        sample_id=0,
        function_state=empty_state,
        target_expression="expression1",
        sexpr=f"(JOIN {relation} {root_mid})",
        denotation=empty_values,
        parent_id=None,
        operation="expand",
        relation_id=relation,
        provenance=["live_smoke_inspect"],
    )
    try:
        count_graph.prune(0, empty.node_id)
    except ValueError as exc:
        if "Count can produce" not in str(exc):
            raise
    else:
        raise RuntimeError("Count-capable empty hypothesis was incorrectly pruned")

    count_graph.select(0, empty.node_id)
    transition_error = count_graph.execution_error(
        0,
        opens_frontier=False,
        frontier_width=6,
        operation="Count",
    )
    if transition_error:
        raise RuntimeError(transition_error)
    count_graph.mark_expanded(0, empty.node_id)
    counted = count_graph.add_executed(
        sample_id=0,
        function_state=count_state,
        target_expression="expression1",
        sexpr=f"(COUNT (JOIN {relation} {root_mid}))",
        denotation=count_values,
        parent_id=empty.node_id,
        operation="count",
        provenance=["live_smoke_count"],
    )
    count_graph.commit(0, counted.node_id)

    noncount_graph = HypothesisGraph(max_active=3, max_nodes=6)
    noncount_graph.register_public_question(
        0, "Which types belong to the deliberately absent entity?"
    )
    noncount_empty = noncount_graph.add_executed(
        sample_id=0,
        function_state=empty_state,
        target_expression="expression1",
        sexpr=f"(JOIN {relation} {root_mid})",
        denotation=empty_values,
        parent_id=None,
        operation="expand",
        relation_id=relation,
        provenance=["live_smoke_inspect"],
    )
    certificate = noncount_graph.prune(0, noncount_empty.node_id)

    return {
        "status": "passed",
        "root_mid": root_mid,
        "relation": relation,
        "empty_execution": True,
        "count_prune_rejected": True,
        "count_result": list(count_values),
        "count_commit": count_graph.state(0).committed_id == counted.node_id,
        "noncount_prune_certificate": asdict(certificate),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-mid", default="m.hyper_r1_intentionally_missing")
    parser.add_argument("--relation", default="type.object.type")
    parser.add_argument("--output")
    args = parser.parse_args()

    from kbqa_r1.sparql.odbc_config import ODBCConfig
    from kbqa_r1.sparql.sparql_manager import SPARQLConfig
    from build_hyper_demonstrations import (
        LiveEntityDisplayProvider,
        LiveProgramExecutor,
    )

    odbc = ODBCConfig.from_env()
    config = SPARQLConfig(use_odbc=True, odbc_config=asdict(odbc))
    display_provider = LiveEntityDisplayProvider(config)
    result = run_smoke(
        LiveProgramExecutor(config, display_provider),
        root_mid=args.root_mid,
        relation=args.relation,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
