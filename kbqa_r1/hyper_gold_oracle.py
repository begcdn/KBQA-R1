"""State-conditioned gold continuation for HyPER-R1 recovery audits.

The oracle uses private gold programs only to select among actions that the
production runtime exposes.  It never inserts relations, entities, proposal
IDs, or graph nodes.  Unsupported or unreachable states fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

from .action_constraints import HyPERActionConstraintSpec
from .hyper_data import (
    GoldPlan,
    IneligibleProgram,
    ProgramStatement,
    compile_gold_plan,
    logical_program_signature,
)
from .hyper_r1 import HypothesisNode, HypothesisStatus


class GoldContinuationUnavailable(RuntimeError):
    """The exact live state cannot be continued by the supported gold oracle."""


@dataclass(frozen=True)
class GoldOracleChoice:
    action: str
    reason: str
    statement_index: int


def _response(action: str) -> str:
    return f"<action>{action}</action>"


def _is_ontology_type(value: str) -> bool:
    parts = str(value).strip().split(".")
    return len(parts) >= 2 and all(
        part and (part[0].isalpha() or part[0] == "_")
        and all(char.isalnum() or char == "_" for char in part)
        for part in parts
    )


def _statement_signatures(plan: GoldPlan) -> dict[int, Tuple[Any, ...]]:
    signatures: dict[int, Tuple[Any, ...]] = {}
    prefix = []
    for statement in plan.statements:
        if statement.kind == "stop":
            continue
        prefix.append(statement.raw)
        signatures[statement.index] = logical_program_signature(
            prefix, statement.target
        )
    return signatures


def _node_signature(node: HypothesisNode) -> Optional[Tuple[Any, ...]]:
    try:
        return logical_program_signature(
            node.function_state, node.target_expression
        )
    except IneligibleProgram:
        return None


def _represented_signatures(node: HypothesisNode) -> frozenset[Tuple[Any, ...]]:
    """Return every logical prefix embedded in a node's executable program."""
    try:
        plan = compile_gold_plan(node.function_state)
        return frozenset(_statement_signatures(plan).values())
    except IneligibleProgram:
        return frozenset()


class GoldContinuationOracle:
    """Choose the next runtime-valid action toward one compiled gold program."""

    def __init__(self, function_list: Sequence[str], target_expression: str):
        self.plan = compile_gold_plan(function_list)
        if str(target_expression) != self.plan.target_expression:
            raise GoldContinuationUnavailable(
                "gold target does not match the compiled program"
            )
        self.signatures = _statement_signatures(self.plan)
        self.gold_signatures = frozenset(self.signatures.values())

    @classmethod
    def from_contract(cls, contract: Mapping[str, Any]) -> "GoldContinuationOracle":
        functions = contract.get("function_list")
        target = str(contract.get("target_expression") or "")
        if not isinstance(functions, (list, tuple)) or not functions or not target:
            raise GoldContinuationUnavailable("missing private gold program")
        return cls(tuple(str(value) for value in functions), target)

    def _nodes_by_signature(
        self, nodes: Sequence[HypothesisNode]
    ) -> tuple[
        dict[Tuple[Any, ...], list[HypothesisNode]],
        dict[Tuple[Any, ...], list[HypothesisNode]],
        frozenset[Tuple[Any, ...]],
    ]:
        all_nodes: dict[Tuple[Any, ...], list[HypothesisNode]] = {}
        usable: dict[Tuple[Any, ...], list[HypothesisNode]] = {}
        represented: set[Tuple[Any, ...]] = set()
        for node in nodes:
            represented.update(_represented_signatures(node))
            signature = _node_signature(node)
            if signature is None:
                continue
            all_nodes.setdefault(signature, []).append(node)
            if node.status in {HypothesisStatus.ACTIVE, HypothesisStatus.PARKED}:
                usable.setdefault(signature, []).append(node)
        return all_nodes, usable, frozenset(represented)

    @staticmethod
    def _preferred_node(
        candidates: Sequence[HypothesisNode], selected_id: Optional[str]
    ) -> HypothesisNode:
        return sorted(
            candidates,
            key=lambda node: (
                node.node_id != selected_id,
                node.status != HypothesisStatus.ACTIVE,
                -node.depth,
                node.node_id,
            ),
        )[0]

    @staticmethod
    def _frontier_relations(frontier: Mapping[str, Any]) -> Tuple[str, ...]:
        decision = frontier.get("decision")
        ranked = (
            decision.ranked_relations
            if hasattr(decision, "ranked_relations")
            else decision.get("ranked_relations", ())
        )
        return tuple(
            str(
                candidate.relation
                if hasattr(candidate, "relation")
                else candidate.get("relation")
            )
            for candidate in ranked
        )

    @staticmethod
    def _proposal_relation(proposal: Mapping[str, Any]) -> str:
        candidate = proposal.get("candidate")
        return str(
            candidate.relation
            if hasattr(candidate, "relation")
            else candidate.get("relation")
        )

    def _legal_choice(
        self,
        action: str,
        reason: str,
        statement_index: int,
        constraint: HyPERActionConstraintSpec,
    ) -> GoldOracleChoice:
        if not constraint.accepts_response(_response(action)):
            raise GoldContinuationUnavailable(
                f"oracle action is outside the live constraint: {action}"
            )
        return GoldOracleChoice(action, reason, statement_index)

    def _make_room(
        self,
        *,
        nodes: Sequence[HypothesisNode],
        protected: Sequence[str],
        constraint: HyPERActionConstraintSpec,
        statement_index: int,
    ) -> GoldOracleChoice:
        protected_ids = set(protected)
        active = [
            node
            for node in nodes
            if node.status == HypothesisStatus.ACTIVE
            and node.node_id not in protected_ids
            and f"Park [ {node.node_id} ]" in constraint.exact_actions
        ]
        if not active:
            raise GoldContinuationUnavailable(
                "gold continuation needs workspace capacity but no safe Park is legal"
            )
        non_gold = [
            node for node in active if _node_signature(node) not in self.gold_signatures
        ]
        node = sorted(non_gold or active, key=lambda value: value.node_id)[0]
        return self._legal_choice(
            f"Park [ {node.node_id} ]",
            "preserve a competing hypothesis while freeing visible workspace",
            statement_index,
            constraint,
        )

    def _ensure_usable(
        self,
        node: HypothesisNode,
        *,
        nodes: Sequence[HypothesisNode],
        constraint: HyPERActionConstraintSpec,
        statement_index: int,
        protected: Sequence[str] = (),
    ) -> Optional[GoldOracleChoice]:
        if node.status == HypothesisStatus.PARKED:
            action = f"Recall [ {node.node_id} ]"
            if action in constraint.exact_actions:
                return self._legal_choice(
                    action,
                    "restore a retained gold-program prefix",
                    statement_index,
                    constraint,
                )
            return self._make_room(
                nodes=nodes,
                protected=(*protected, node.node_id),
                constraint=constraint,
                statement_index=statement_index,
            )
        if node.status != HypothesisStatus.ACTIVE:
            raise GoldContinuationUnavailable(
                f"required gold prefix {node.node_id} is not recoverable"
            )
        return None

    def _relation_action(
        self,
        *,
        source: str,
        relation: str,
        frontiers: Sequence[Mapping[str, Any]],
        constraint: HyPERActionConstraintSpec,
        statement_index: int,
    ) -> GoldOracleChoice:
        matching = [
            frontier
            for frontier in reversed(frontiers)
            if not frontier.get("closed")
            and str(frontier.get("source")) == source
            and relation in self._frontier_relations(frontier)
        ]
        if matching:
            frontier = matching[0]
            for proposal_id, proposal in frontier.get("proposals", {}).items():
                if self._proposal_relation(proposal) != relation:
                    continue
                status = str(proposal.get("status"))
                if status == "visible":
                    return self._legal_choice(
                        f"Inspect [ {proposal_id} ]",
                        "execute the visible gold relation proposal",
                        statement_index,
                        constraint,
                    )
                if status in {"failed", "inspected"}:
                    raise GoldContinuationUnavailable(
                        f"required proposal {proposal_id} is already {status}"
                    )
            exposed = int(frontier.get("next_offset", 0))
            total = len(self._frontier_relations(frontier))
            if exposed < total:
                return self._legal_choice(
                    f"Widen [ {source} ]",
                    "reveal the page containing the ranked gold relation",
                    statement_index,
                    constraint,
                )
            raise GoldContinuationUnavailable(
                "gold relation is ranked but no longer inspectable"
            )

        return self._legal_choice(
            f"Find_relation [ {source} ]",
            "open the live ranked catalog for the next gold relation",
            statement_index,
            constraint,
        )

    def _has_matching_frontier(
        self,
        frontiers: Sequence[Mapping[str, Any]],
        source: str,
        relation: str,
    ) -> bool:
        return any(
            not frontier.get("closed")
            and str(frontier.get("source")) == source
            and relation in self._frontier_relations(frontier)
            for frontier in frontiers
        )

    def _definition(self, target: str, before: int) -> ProgramStatement:
        statement = next(
            (
                value
                for value in reversed(self.plan.statements)
                if value.kind != "stop"
                and value.index < before
                and value.target == target
            ),
            None,
        )
        if statement is None:
            raise GoldContinuationUnavailable(
                f"gold source {target} has no prior definition"
            )
        return statement

    def _source_node(
        self,
        statement: ProgramStatement,
        usable: Mapping[Tuple[Any, ...], Sequence[HypothesisNode]],
        selected_id: Optional[str],
    ) -> Optional[HypothesisNode]:
        if statement.kind == "start":
            return None
        candidates = usable.get(self.signatures[statement.index], ())
        return self._preferred_node(candidates, selected_id) if candidates else None

    def choose(
        self,
        *,
        nodes: Sequence[HypothesisNode],
        selected_id: Optional[str],
        frontiers: Sequence[Mapping[str, Any]],
        constraint: HyPERActionConstraintSpec,
    ) -> GoldOracleChoice:
        all_nodes, usable, represented = self._nodes_by_signature(nodes)

        final_statement = self._definition(
            self.plan.target_expression, len(self.plan.statements) + 1
        )
        final_candidates = usable.get(self.signatures[final_statement.index], ())
        if final_candidates:
            final = self._preferred_node(final_candidates, selected_id)
            preparation = self._ensure_usable(
                final,
                nodes=nodes,
                constraint=constraint,
                statement_index=final_statement.index,
            )
            if preparation is not None:
                return preparation
            return self._legal_choice(
                f"Commit [ {final.node_id} ]",
                "commit the exact complete gold-program hypothesis",
                final_statement.index,
                constraint,
            )

        for statement in self.plan.statements:
            if statement.kind in {"start", "stop"}:
                continue
            signature = self.signatures[statement.index]
            if signature in represented:
                continue

            if statement.kind == "join":
                if statement.sources:
                    source_statement = self._definition(
                        statement.sources[0], statement.index
                    )
                    parent = self._source_node(
                        source_statement, usable, selected_id
                    )
                    if parent is not None:
                        preparation = self._ensure_usable(
                            parent,
                            nodes=nodes,
                            constraint=constraint,
                            statement_index=statement.index,
                        )
                        if preparation is not None:
                            return preparation
                        if selected_id != parent.node_id:
                            if self._has_matching_frontier(
                                frontiers,
                                parent.target_expression,
                                str(statement.relation),
                            ):
                                return self._relation_action(
                                    source=parent.target_expression,
                                    relation=str(statement.relation),
                                    frontiers=frontiers,
                                    constraint=constraint,
                                    statement_index=statement.index,
                                )
                            return self._legal_choice(
                                f"Select [ {parent.node_id} ]",
                                "restore the deepest retained gold-program prefix",
                                statement.index,
                                constraint,
                            )
                        source = parent.target_expression
                    elif source_statement.kind == "start":
                        source = source_statement.arguments[0]
                        if selected_id is not None:
                            selected = next(
                                node for node in nodes if node.node_id == selected_id
                            )
                            return self._legal_choice(
                                f"Park [ {selected.node_id} ]",
                                "preserve the selected branch before opening another gold root",
                                statement.index,
                                constraint,
                            )
                    else:
                        raise GoldContinuationUnavailable(
                            "the next gold join has no retained executable parent"
                        )
                else:
                    source = statement.arguments[1].strip("'")
                    if selected_id is not None:
                        selected = next(
                            node for node in nodes if node.node_id == selected_id
                        )
                        return self._legal_choice(
                            f"Park [ {selected.node_id} ]",
                            "preserve the selected branch before opening another gold root",
                            statement.index,
                            constraint,
                        )
                return self._relation_action(
                    source=source,
                    relation=str(statement.relation),
                    frontiers=frontiers,
                    constraint=constraint,
                    statement_index=statement.index,
                )

            if statement.kind == "and":
                sources = [
                    self._definition(source, statement.index)
                    for source in statement.sources
                ]
                source_nodes = [
                    self._source_node(source, usable, selected_id)
                    for source in sources
                ]
                bare = [
                    source.arguments[0]
                    if source.kind == "start" and _is_ontology_type(source.arguments[0])
                    else None
                    for source in sources
                ]
                if sum(node is not None for node in source_nodes) == 1 and sum(
                    value is not None for value in bare
                ) == 1:
                    parent = next(node for node in source_nodes if node is not None)
                    ontology = next(value for value in bare if value is not None)
                    preparation = self._ensure_usable(
                        parent,
                        nodes=nodes,
                        constraint=constraint,
                        statement_index=statement.index,
                    )
                    if preparation is not None:
                        return preparation
                    if selected_id != parent.node_id:
                        return self._legal_choice(
                            f"Select [ {parent.node_id} ]",
                            "select the retained branch before applying its gold type",
                            statement.index,
                            constraint,
                        )
                    return self._legal_choice(
                        f"Merge [ {parent.target_expression} | {ontology} ]",
                        "apply the exact ontology-type intersection from the gold program",
                        statement.index,
                        constraint,
                    )
                if all(node is not None for node in source_nodes):
                    left, right = source_nodes
                    for parent in (left, right):
                        preparation = self._ensure_usable(
                            parent,
                            nodes=nodes,
                            constraint=constraint,
                            statement_index=statement.index,
                            protected=(left.node_id, right.node_id),
                        )
                        if preparation is not None:
                            return preparation
                    return self._legal_choice(
                        f"Combine [ {left.node_id} | {right.node_id} ]",
                        "intersect both retained gold-program branches",
                        statement.index,
                        constraint,
                    )
                raise GoldContinuationUnavailable(
                    "the next gold intersection has a missing retained branch"
                )

            source = statement.sources[0]
            start = self._definition(source, statement.index)
            parent = self._source_node(start, usable, selected_id)
            root_operator = parent is None and statement.kind in {"compare", "order"}
            if not root_operator:
                if parent is None:
                    raise GoldContinuationUnavailable(
                        f"the next gold {statement.kind} has no retained parent"
                    )
                preparation = self._ensure_usable(
                    parent,
                    nodes=nodes,
                    constraint=constraint,
                    statement_index=statement.index,
                )
                if preparation is not None:
                    return preparation
                if selected_id != parent.node_id:
                    return self._legal_choice(
                        f"Select [ {parent.node_id} ]",
                        f"select the retained branch before {statement.kind}",
                        statement.index,
                        constraint,
                    )

            if statement.kind == "order":
                mode, _, relation = statement.arguments
                operand = parent.target_expression if parent is not None else start.arguments[0]
                action = f"Order [ {mode} | {operand} | {relation} ]"
            elif statement.kind == "compare":
                if start is None or start.kind != "start":
                    raise GoldContinuationUnavailable("comparison literal is unavailable")
                mode, relation, _ = statement.arguments
                action = f"Compare [ {mode} | {relation} | {start.arguments[0]} ]"
            elif statement.kind == "time_constraint":
                relation, time = statement.arguments
                action = f"Time_constraint [ {relation} | {time} ]"
            elif statement.kind == "count":
                action = f"Count [ {parent.target_expression} ]"
            else:
                raise GoldContinuationUnavailable(
                    f"unsupported gold statement kind: {statement.kind}"
                )
            return self._legal_choice(
                action,
                f"execute the exact {statement.kind} required by the gold program",
                statement.index,
                constraint,
            )

        raise GoldContinuationUnavailable(
            "no executable gold continuation can be derived from this state"
        )
