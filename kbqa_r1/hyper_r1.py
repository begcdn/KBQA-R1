"""Executable hypothesis graph used by HyPER-R1.

The graph is deliberately independent of the LLM and the SPARQL backend.  The
environment executes a candidate first and then records the resulting logical
form and denotation here.  This keeps graph bookkeeping deterministic and makes
the same state usable during SFT, RL, and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import re

import torch

from .answer_utils import extract_last_answer_values


class HypothesisStatus(str, Enum):
    ACTIVE = "active"
    EXPANDED = "expanded"
    PRUNED = "pruned"
    MERGED = "merged"
    COMMITTED = "committed"


class HypothesisEdgeKind(str, Enum):
    EXPANSION = "expansion"
    CONTRAST = "contrast"
    COMPOSITION = "composition"
    EQUIVALENCE = "equivalence"


@dataclass(frozen=True)
class HypothesisEdge:
    source: str
    target: str
    kind: HypothesisEdgeKind
    label: str = ""


@dataclass
class HypothesisNode:
    node_id: str
    sample_id: int
    function_state: Tuple[str, ...]
    target_expression: str
    sexpr: str
    denotation: Tuple[str, ...]
    parent_id: Optional[str]
    operation: str
    denotation_labels: Dict[str, str] = field(default_factory=dict)
    parent_ids: Tuple[str, ...] = ()
    relation_id: Optional[str] = None
    relation_prompt: Optional[str] = None
    resolver_score: float = 0.0
    depth: int = 0
    status: HypothesisStatus = HypothesisStatus.ACTIVE
    equivalent_to: Optional[str] = None
    provenance: List[str] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        return self.status == HypothesisStatus.ACTIVE


@dataclass
class HypothesisGraphState:
    sample_id: int
    nodes: Dict[str, HypothesisNode] = field(default_factory=dict)
    edges: List[HypothesisEdge] = field(default_factory=list)
    selected_id: Optional[str] = None
    committed_id: Optional[str] = None
    execution_attempts: int = 0
    execution_calls: int = 0
    next_node_index: int = 0


def normalize_denotation(values: Iterable[str]) -> Tuple[str, ...]:
    """Canonicalize an executor result without erasing entity identity."""
    normalized = []
    for value in values or ():
        text = str(value).strip()
        if text:
            normalized.append(text)
    return tuple(sorted(set(normalized)))


def normalize_display_labels(labels: Optional[Mapping[str, str]]) -> Dict[str, str]:
    """Keep only useful executor-owned labels for public graph observations."""
    normalized: Dict[str, str] = {}
    for identity, label in (labels or {}).items():
        key = str(identity).strip()
        value = str(label).strip()
        if key and value and value != key:
            normalized[key] = value
    return normalized


def result_display_labels(results: Any) -> Dict[str, str]:
    """Extract entity labels already returned by the executable SPARQL query."""
    labels: Dict[str, str] = {}
    rows = results if isinstance(results, list) else [results]
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        identity = row.get("x")
        label = row.get("name") or row.get("label")
        if identity is not None and label is not None:
            labels[str(identity).strip()] = str(label).strip()
        elif identity is not None:
            value = str(identity).strip()
            if not re.fullmatch(r"[mg]\.[A-Za-z0-9_]+", value) and "." in value:
                labels[value] = value.rsplit(".", 1)[-1].replace("_", " ")
    return normalize_display_labels(labels)


def result_denotation_values(results: Any, fallback: Iterable[str] = ()) -> Tuple[str, ...]:
    """Keep executor identities separate from optional human-readable labels."""
    values: List[str] = []
    rows = results if isinstance(results, list) else [results]
    for row in rows:
        if isinstance(row, Mapping) and row.get("x") is not None:
            values.append(str(row["x"]))
    return normalize_denotation(values if values else fallback)


_MID_RE = re.compile(r"(?:[A-Za-z_][A-Za-z0-9_]*:)?([mg]\.[A-Za-z0-9_]+)")


def _display_answer(value: str, labels: Mapping[str, str]) -> str:
    text = str(value)
    match = _MID_RE.search(text)
    identity = match.group(1) if match else text
    label = labels.get(text) or labels.get(identity)
    if not label:
        return text
    return f"{label} [{identity}]"


def extract_answer_values(text: str) -> Tuple[str, ...]:
    """Read the final answer tag using the same identity-preserving normalization."""
    return extract_last_answer_values(text) or ()


def relation_path(function_state: Sequence[str]) -> Tuple[str, ...]:
    """Expose the executed relation sequence without leaking private labels."""
    relations: List[str] = []
    pattern = re.compile(r"\bJOIN\(\s*['\"](.+?)['\"]\s*,")
    for statement in function_state:
        match = pattern.search(str(statement))
        if match:
            relations.append(match.group(1))
    return tuple(relations)


def _node_source(provenance: Sequence[str]) -> str:
    if "policy_choice" in provenance:
        return "policy"
    if "ranked_alternative" in provenance or "hard_sibling" in provenance:
        return "alternative"
    return "derived"


def serialize_frontier(
    nodes: Sequence[Mapping[str, Any]],
    *,
    active_ids: Sequence[str],
    selected_id: Optional[str],
    committed_id: Optional[str],
    max_active: int,
    node_count: int,
    execution_calls: int,
    max_answers: int = 4,
) -> str:
    """Canonical policy observation shared by demonstrations and live rollouts."""
    active = set(active_ids)
    visible = active | ({committed_id} if committed_id else set())
    lines = [
        "<hypothesis_graph>",
        f"active={len(active)} capacity={max_active} nodes={node_count} "
        f"execution_attempts={execution_calls} selected={selected_id or 'none'} "
        f"committed={committed_id or 'none'}",
    ]
    for node in nodes:
        node_id = str(node["node_id"])
        if node_id not in visible:
            continue
        denotation = normalize_denotation(node.get("denotation", ()))
        labels = normalize_display_labels(node.get("denotation_labels", {}))
        answers = ", ".join(
            _display_answer(value, labels) for value in denotation[:max_answers]
        ) or "empty"
        if len(denotation) > max_answers:
            answers += f", ... (+{len(denotation) - max_answers})"
        status = "committed" if node_id == committed_id else "active"
        path = " -> ".join(relation_path(node.get("function_state", ()))) or "root"
        provenance = tuple(node.get("provenance", ()))
        parents = tuple(node.get("parent_ids", ()))
        parent_text = "+".join(parents) if parents else node.get("parent_id") or "ROOT"
        operation = str(node.get("operation") or "expand")
        lines.append(
            f"{node_id} [{status}] parents={parent_text} operation={operation} "
            f"via={node.get('relation_id') or node.get('operation') or 'derived'} "
            f"source={_node_source(provenance)} depth={int(node.get('depth', 0))} "
            f"path={path} answers={len(denotation)}: {answers}"
        )
    lines.append(
        "Actions: Select, Find_relation [ source ], Widen [ source ], Combine, Prune, or Commit. "
        "Widen exposes the next stable ranked page. Prune only for a visible path or "
        "execution contradiction; low rank or limited capacity is not a contradiction."
    )
    lines.append("</hypothesis_graph>")
    return "\n".join(lines)


_EXPRESSION_RE = re.compile(r"\bexpression\d*\b")


def _expression_number(name: str) -> int:
    suffix = str(name)[len("expression") :]
    return int(suffix) if suffix else 0


def dependency_function_state(
    function_state: Sequence[str], target: str
) -> Tuple[str, ...]:
    """Keep only assignments that determine ``target``.

    The live executor may retain definitions from an earlier root while it
    opens another branch. Those definitions are harmless to SPARQL execution,
    but storing them in the new hypothesis would make two independent branches
    look like one path. Resolve overwritten expression variables by assignment
    position and retain the exact dependency closure of the requested target.
    """
    assignments = []
    for index, raw_value in enumerate(function_state):
        raw = str(raw_value).strip()
        if "=" not in raw:
            continue
        lhs, rhs = (part.strip() for part in raw.split("=", 1))
        if not _EXPRESSION_RE.fullmatch(lhs):
            continue
        assignments.append((index, lhs, rhs, raw))

    retained = set()

    def visit(expression: str, before: int) -> None:
        match = next(
            (
                item
                for item in reversed(assignments)
                if item[0] < before and item[1] == expression
            ),
            None,
        )
        if match is None or match[0] in retained:
            return
        index, _, rhs, _ = match
        for source in _EXPRESSION_RE.findall(rhs):
            visit(source, index)
        retained.add(index)

    visit(str(target), len(function_state) + 1)
    return tuple(raw for index, _, _, raw in assignments if index in retained)


def combine_function_states(
    left_state: Sequence[str],
    left_target: str,
    right_state: Sequence[str],
    right_target: str,
) -> Tuple[List[str], str]:
    """Combine two branch programs without expression-name collisions.

    Branches normally share a prefix and then overwrite the same expression
    variable in different ways.  The right suffix is replayed with fresh
    expression names before an AND node is appended.
    """
    common = 0
    for left, right in zip(left_state, right_state):
        if left != right:
            break
        common += 1

    merged = list(left_state)
    existing_ids = [
        _expression_number(name)
        for statement in merged
        for name in _EXPRESSION_RE.findall(statement)
    ]
    next_id = max(existing_ids, default=0) + 1

    # At the fork point, references in the right suffix still point to the
    # common-prefix values.  Each subsequent assignment updates the mapping.
    mapping = {name: name for statement in right_state[:common] for name in _EXPRESSION_RE.findall(statement)}
    for statement in right_state[common:]:
        if "=" not in statement:
            raise ValueError(f"invalid function statement: {statement}")
        lhs, rhs = (part.strip() for part in statement.split("=", 1))
        if not _EXPRESSION_RE.fullmatch(lhs):
            raise ValueError(f"invalid expression assignment: {statement}")
        rewritten_rhs = _EXPRESSION_RE.sub(lambda match: mapping.get(match.group(0), match.group(0)), rhs)
        fresh_lhs = f"expression{next_id}"
        next_id += 1
        merged.append(f"{fresh_lhs} = {rewritten_rhs}")
        mapping[lhs] = fresh_lhs

    right_renamed = mapping.get(right_target, right_target)
    target = f"expression{next_id}"
    merged.append(f"{target} = AND({left_target}, {right_renamed})")
    return merged, target


class HypothesisGraph:
    """Owns persistent executable alternatives for all samples in a rollout."""

    def __init__(self, max_active: int = 24, max_nodes: int = 24):
        if max_active < 2:
            raise ValueError("HyPER-R1 requires at least two active hypotheses")
        if max_nodes < max_active:
            raise ValueError("max_nodes must be at least max_active")
        self.max_active = int(max_active)
        self.max_nodes = int(max_nodes)
        self._samples: Dict[int, HypothesisGraphState] = {}

    def state(self, sample_id: int) -> HypothesisGraphState:
        if sample_id not in self._samples:
            self._samples[sample_id] = HypothesisGraphState(sample_id=sample_id)
        return self._samples[sample_id]

    def has_capacity(self, sample_id: int, required: int = 1) -> bool:
        """Return whether another executed hypothesis can be retained."""
        if required < 0:
            raise ValueError("required must be non-negative")
        return len(self.state(sample_id).nodes) + required <= self.max_nodes

    def clear(self, sample_id: Optional[int] = None) -> None:
        if sample_id is None:
            self._samples.clear()
        else:
            self._samples.pop(sample_id, None)

    def add_executed(
        self,
        *,
        sample_id: int,
        function_state: Sequence[str],
        target_expression: str,
        sexpr: str,
        denotation: Iterable[str],
        parent_id: Optional[str],
        parent_ids: Sequence[str] = (),
        operation: str,
        denotation_labels: Optional[Mapping[str, str]] = None,
        relation_id: Optional[str] = None,
        relation_prompt: Optional[str] = None,
        resolver_score: float = 0.0,
        contrast_group: Optional[str] = None,
        provenance: Optional[Sequence[str]] = None,
    ) -> HypothesisNode:
        graph = self.state(sample_id)
        if len(graph.nodes) >= self.max_nodes:
            raise RuntimeError(
                f"HyPER-R1 node budget exhausted for sample {sample_id}: "
                f"{len(graph.nodes)}/{self.max_nodes}"
            )
        all_parent_ids = tuple(parent_ids or ((parent_id,) if parent_id else ()))
        unknown_parents = [value for value in all_parent_ids if value not in graph.nodes]
        if unknown_parents:
            raise KeyError(f"unknown parent hypotheses: {unknown_parents}")
        if len(self.active_nodes(sample_id)) >= self.max_active:
            raise RuntimeError(
                f"HyPER-R1 active frontier is full for sample {sample_id}: "
                f"{len(self.active_nodes(sample_id))}/{self.max_active}; prune before exploring"
            )

        node_id = f"H{graph.next_node_index}"
        graph.next_node_index += 1
        parent_depth = graph.nodes[parent_id].depth if parent_id is not None else -1
        node = HypothesisNode(
            node_id=node_id,
            sample_id=sample_id,
            function_state=tuple(function_state),
            target_expression=str(target_expression),
            sexpr=str(sexpr),
            denotation=normalize_denotation(denotation),
            denotation_labels=normalize_display_labels(denotation_labels),
            parent_id=parent_id,
            parent_ids=all_parent_ids,
            operation=str(operation),
            relation_id=relation_id,
            relation_prompt=relation_prompt,
            resolver_score=float(resolver_score or 0.0),
            depth=parent_depth + 1,
            provenance=list(provenance or ()),
        )
        graph.nodes[node_id] = node
        graph.execution_calls += 1

        for edge_parent in all_parent_ids:
            graph.edges.append(
                HypothesisEdge(
                    source=edge_parent,
                    target=node_id,
                    kind=(
                        HypothesisEdgeKind.COMPOSITION
                        if operation in {"combine", "count", "order", "compare", "time"}
                        else HypothesisEdgeKind.EXPANSION
                    ),
                    label=relation_id or operation,
                )
            )

        if contrast_group:
            for sibling in graph.nodes.values():
                if sibling.node_id == node_id or not sibling.is_active:
                    continue
                if contrast_group in sibling.provenance:
                    graph.edges.append(
                        HypothesisEdge(
                            source=sibling.node_id,
                            target=node_id,
                            kind=HypothesisEdgeKind.CONTRAST,
                            label=contrast_group,
                        )
                    )
            node.provenance.append(contrast_group)

        equivalent = self._find_equivalent(graph, node)
        if equivalent is not None:
            node.status = HypothesisStatus.MERGED
            node.equivalent_to = equivalent.node_id
            equivalent.provenance.extend(
                item for item in node.provenance if item not in equivalent.provenance
            )
            graph.edges.append(
                HypothesisEdge(
                    source=node_id,
                    target=equivalent.node_id,
                    kind=HypothesisEdgeKind.EQUIVALENCE,
                    label="same_program",
                )
            )

        return node

    def record_execution_attempt(self, sample_id: int) -> int:
        """Count every proposed graph execution, including failed proposals."""
        graph = self.state(sample_id)
        graph.execution_attempts += 1
        return graph.execution_attempts

    def _find_equivalent(
        self, graph: HypothesisGraphState, candidate: HypothesisNode
    ) -> Optional[HypothesisNode]:
        # Equal answers at one step do not imply equal meaning. HyPER-R1 keeps
        # such paths separate and deduplicates only byte-for-byte programs.
        for node in graph.nodes.values():
            if node.node_id == candidate.node_id or not node.is_active:
                continue
            if (
                node.function_state == candidate.function_state
                and node.target_expression == candidate.target_expression
                and node.sexpr == candidate.sexpr
            ):
                return node
        return None

    def select(self, sample_id: int, node_id: str) -> HypothesisNode:
        graph = self.state(sample_id)
        if graph.committed_id is not None:
            raise ValueError("the hypothesis graph is already committed")
        node = self.require_active(sample_id, node_id)
        if not node.denotation:
            raise ValueError("cannot select an empty hypothesis")
        graph.selected_id = node_id
        return node

    def selected_node(self, sample_id: int) -> Optional[HypothesisNode]:
        graph = self.state(sample_id)
        if graph.selected_id is None:
            return None
        return graph.nodes.get(graph.selected_id)

    def execution_error(
        self,
        sample_id: int,
        *,
        opens_frontier: bool,
        frontier_width: int,
        opens_new_root: bool = False,
    ) -> Optional[str]:
        """Return why an executable policy action is illegal in this state."""
        graph = self.state(sample_id)
        if graph.committed_id is not None:
            return "The graph is committed. Return the committed values in <answer>."
        active = self.active_nodes(sample_id)
        if active and graph.selected_id is None and not opens_new_root:
            return "Select an active hypothesis before any executable continuation."
        if not active and graph.nodes and graph.selected_id is None:
            return "No active hypothesis can be continued."
        if not graph.nodes and not opens_frontier and not opens_new_root:
            return "Begin the executable frontier with Find_relation."
        if opens_frontier:
            active_after = len(active) - (1 if graph.selected_id is not None else 0)
            if (
                active_after + frontier_width > self.max_active
                or not self.has_capacity(sample_id, frontier_width)
            ):
                return (
                    "The next complete relation page does not fit the remaining "
                    "uniform node budget; no partial page was executed."
                )
        elif not self.has_capacity(sample_id):
            return "The executed-node budget is exhausted; Commit an active hypothesis."
        return None

    def relation_source_error(
        self,
        sample_id: int,
        source: str,
        candidate_sources: Sequence[str],
    ) -> Optional[str]:
        """Require relation retrieval to continue from a public valid source."""
        graph = self.state(sample_id)
        candidate_sources = {str(value) for value in candidate_sources}
        if graph.selected_id is not None:
            selected = self.require_active(sample_id, graph.selected_id)
            if str(source) != selected.target_expression:
                return (
                    f"Find_relation source must be the selected hypothesis target "
                    f"{selected.target_expression}."
                )
            return None
        if str(source) not in candidate_sources:
            return "The initial Find_relation source must be a supplied candidate entity or literal."
        return None

    def mark_expanded(self, sample_id: int, node_id: Optional[str]) -> None:
        """Replace an explored leaf with its children without deleting history."""
        if node_id is None:
            return
        node = self.require_active(sample_id, node_id)
        node.status = HypothesisStatus.EXPANDED
        node.provenance.append("expanded")
        if self.state(sample_id).selected_id == node_id:
            self.state(sample_id).selected_id = None

    def available_active_slots(self, sample_id: int) -> int:
        return self.max_active - len(self.active_nodes(sample_id))

    def combination_parents(
        self, sample_id: int, left_id: str, right_id: str
    ) -> Tuple[HypothesisNode, HypothesisNode]:
        """Validate a composition before execution or graph mutation."""
        if left_id == right_id:
            raise ValueError("Combine requires two distinct active hypotheses")
        if not self.has_capacity(sample_id):
            raise RuntimeError("HyPER-R1 node budget exhausted before Combine")
        return (
            self.require_active(sample_id, left_id),
            self.require_active(sample_id, right_id),
        )

    def prune(self, sample_id: int, node_id: str, reason: str = "policy") -> None:
        node = self.require_active(sample_id, node_id)
        node.status = HypothesisStatus.PRUNED
        node.provenance.append(reason)
        if self.state(sample_id).selected_id == node_id:
            self.state(sample_id).selected_id = None

    def commit(self, sample_id: int, node_id: str) -> HypothesisNode:
        graph = self.state(sample_id)
        node = self.require_active(sample_id, node_id)
        if not node.denotation:
            raise ValueError("cannot commit an empty hypothesis")
        for other in graph.nodes.values():
            if other.is_active and other.node_id != node_id:
                other.status = HypothesisStatus.PRUNED
                other.provenance.append("commit")
        node.status = HypothesisStatus.COMMITTED
        graph.selected_id = None
        graph.committed_id = node_id
        return node

    def committed_node(self, sample_id: int) -> Optional[HypothesisNode]:
        graph = self.state(sample_id)
        if graph.committed_id is None:
            return None
        return graph.nodes.get(graph.committed_id)

    def answer_matches_commit(self, sample_id: int, response: str) -> bool:
        node = self.committed_node(sample_id)
        return bool(node is not None and extract_answer_values(response) == node.denotation)

    def answer_values_match_commit(
        self, sample_id: int, values: Optional[Sequence[str]]
    ) -> bool:
        node = self.committed_node(sample_id)
        return bool(
            node is not None
            and values is not None
            and normalize_denotation(values) == node.denotation
        )

    def decision_state_key(
        self,
        sample_id: int,
        turn: Optional[int] = None,
        legal_context: Sequence[Any] = (),
    ) -> str:
        """Stable semantic key for comparing decisions from the same state."""
        graph = self.state(sample_id)
        payload = {
            "selected": self._node_program_key(graph.nodes.get(graph.selected_id)),
            "active": sorted(
                (self._node_program_key(node) for node in self.active_nodes(sample_id)),
                key=repr,
            ),
            "node_count": len(graph.nodes),
            "execution_attempts": graph.execution_attempts,
            "execution_calls": graph.execution_calls,
            "turn": None if turn is None else int(turn),
            "legal_context": list(legal_context),
        }
        return sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()[:20]

    @staticmethod
    def _node_program_key(node: Optional[HypothesisNode]) -> Optional[Tuple[Any, ...]]:
        if node is None:
            return None
        return (node.function_state, node.target_expression, node.operation, node.relation_id)

    def action_key(
        self,
        sample_id: int,
        action: str,
        node_ids: Sequence[str],
        details: Sequence[str] = (),
    ) -> str:
        graph = self.state(sample_id)
        programs = [
            self._node_program_key(graph.nodes.get(node_id)) for node_id in node_ids
        ]
        if str(action).lower() == "combine":
            programs = sorted(programs, key=repr)
        payload = {
            "action": str(action),
            "programs": programs,
            "details": [str(value) for value in details],
        }
        return sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:20]

    def require_active(self, sample_id: int, node_id: str) -> HypothesisNode:
        """Return an active node or raise a user-facing graph-action error."""
        graph = self.state(sample_id)
        if node_id not in graph.nodes:
            raise KeyError(f"unknown hypothesis: {node_id}")
        node = graph.nodes[node_id]
        if not node.is_active:
            raise ValueError(f"hypothesis {node_id} is {node.status.value}, not active")
        return node

    def active_nodes(self, sample_id: int) -> List[HypothesisNode]:
        return [node for node in self.state(sample_id).nodes.values() if node.is_active]

    def lineage(self, sample_id: int, node_id: Optional[str] = None) -> List[str]:
        graph = self.state(sample_id)
        current = node_id or graph.committed_id
        lineage = []
        seen = set()
        while current is not None:
            if current in seen or current not in graph.nodes:
                raise ValueError(f"invalid hypothesis lineage at {current}")
            seen.add(current)
            lineage.append(current)
            current = graph.nodes[current].parent_id
        return lineage

    def serialize(self, sample_id: int, max_answers: int = 4) -> str:
        """Compact policy observation; names remain the executor's responsibility."""
        graph = self.state(sample_id)
        return serialize_frontier(
            [node.__dict__ for node in graph.nodes.values()],
            active_ids=[node.node_id for node in self.active_nodes(sample_id)],
            selected_id=graph.selected_id,
            committed_id=graph.committed_id,
            max_active=self.max_active,
            node_count=len(graph.nodes),
            execution_calls=graph.execution_attempts,
            max_answers=max_answers,
        )

    def to_dict(self, sample_id: int) -> dict:
        graph = self.state(sample_id)
        return {
            "sample_id": sample_id,
            "selected_id": graph.selected_id,
            "committed_id": graph.committed_id,
            "execution_calls": graph.execution_calls,
            "execution_attempts": graph.execution_attempts,
            "nodes": [
                {
                    **node.__dict__,
                    "function_state": list(node.function_state),
                    "denotation": list(node.denotation),
                    "denotation_labels": dict(node.denotation_labels),
                    "parent_ids": list(node.parent_ids),
                    "status": node.status.value,
                }
                for node in graph.nodes.values()
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "kind": edge.kind.value,
                    "label": edge.label,
                }
                for edge in graph.edges
            ],
        }


def graph_action_token_mask(
    token_ids: Sequence[int], action_token_ids: Sequence[int]
) -> List[int]:
    """Mask all exact occurrences of a structured graph action in a response."""
    mask = [0] * len(token_ids)
    if not action_token_ids or len(action_token_ids) > len(token_ids):
        return mask
    width = len(action_token_ids)
    for start in range(len(token_ids) - width + 1):
        if list(token_ids[start : start + width]) == list(action_token_ids):
            mask[start : start + width] = [1] * width
    return mask


def apply_grouped_decision_credit(
    advantages: torch.Tensor,
    action_ids: torch.Tensor,
    terminal_rewards: torch.Tensor,
    group_ids: Sequence[str],
    action_records: Sequence[Sequence[Mapping[str, Any]]],
    weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Credit actions only when sibling rollouts tried a different decision.

    GRPO already samples several trajectories per question.  For every exact
    semantic frontier state, this compares an action's terminal reward with the
    rewards of *different* actions sampled from that same state. Unique actions
    receive no invented local credit.
    """
    if advantages.shape != action_ids.shape:
        raise ValueError("advantages and action_ids must have the same shape")
    if terminal_rewards.ndim != 1 or terminal_rewards.shape[0] != advantages.shape[0]:
        raise ValueError("terminal_rewards must contain one value per rollout")
    if len(group_ids) != advantages.shape[0] or len(action_records) != advantages.shape[0]:
        raise ValueError("group ids and records must align with rollouts")

    outcomes: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    for row, records in enumerate(action_records):
        reward = float(terminal_rewards[row].item())
        for record in records:
            key = (str(group_ids[row]), str(record["state_key"]))
            outcomes.setdefault(key, {}).setdefault(str(record["action_key"]), []).append(reward)

    result = advantages.clone()
    compared = torch.zeros_like(action_ids, dtype=torch.float32)
    for row, records in enumerate(action_records):
        reward = float(terminal_rewards[row].item())
        for fallback_index, record in enumerate(records, 1):
            action_index = int(record.get("action_index", fallback_index))
            alternatives = outcomes.get(
                (str(group_ids[row]), str(record["state_key"])), {}
            )
            other = [
                value
                for action_key, values in alternatives.items()
                if action_key != str(record["action_key"])
                for value in values
            ]
            if not other:
                continue
            delta = max(-1.0, min(1.0, reward - sum(other) / len(other)))
            token_mask = action_ids == action_index
            result[row][token_mask[row]] += float(weight) * delta
            compared[row][token_mask[row]] = 1.0
    return result, compared


def penalize_invalid_actions(
    advantages: torch.Tensor,
    invalid_action_mask: torch.Tensor,
    penalty: float = 0.25,
) -> torch.Tensor:
    """Prevent malformed or rejected actions from receiving positive credit."""
    if advantages.shape != invalid_action_mask.shape:
        raise ValueError("advantages and invalid_action_mask must have the same shape")
    result = advantages.clone()
    mask = invalid_action_mask.to(dtype=torch.bool)
    floor = torch.full_like(result, -abs(float(penalty)))
    result[mask] = torch.minimum(result[mask], floor[mask])
    return result


def enforce_commit_reward(
    token_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    commit_valid: torch.Tensor,
    invalid_penalty: float = 0.25,
) -> torch.Tensor:
    """Make an answer-grounded Commit a necessary condition for reward."""
    if token_rewards.shape != response_mask.shape:
        raise ValueError("token_rewards and response_mask must have the same shape")
    if commit_valid.ndim != 1 or commit_valid.shape[0] != token_rewards.shape[0]:
        raise ValueError("commit_valid must contain one value per rollout")
    result = token_rewards.clone()
    invalid = commit_valid.to(dtype=torch.bool).logical_not()
    result[invalid] = 0
    lengths = response_mask.long().sum(dim=-1).clamp_min(1) - 1
    rows = torch.arange(result.shape[0], device=result.device)
    result[rows[invalid], lengths[invalid]] = -abs(float(invalid_penalty))
    return result


def charge_execution_budget(
    token_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    execution_counts: torch.Tensor,
    max_nodes: int,
    cost: float,
    group_ids: Optional[Sequence[str]] = None,
) -> torch.Tensor:
    """Charge only executions beyond the cheapest sibling rollout."""
    if token_rewards.shape != response_mask.shape:
        raise ValueError("token_rewards and response_mask must have the same shape")
    if execution_counts.ndim != 1 or execution_counts.shape[0] != token_rewards.shape[0]:
        raise ValueError("execution_counts must be one value per rollout")
    if max_nodes <= 0:
        raise ValueError("max_nodes must be positive")
    result = token_rewards.clone()
    lengths = response_mask.long().sum(dim=-1).clamp_min(1) - 1
    baseline = torch.zeros_like(execution_counts)
    if group_ids is not None:
        if len(group_ids) != execution_counts.shape[0]:
            raise ValueError("group_ids must align with rollouts")
        minima: Dict[str, float] = {}
        for group_id, count in zip(group_ids, execution_counts.tolist()):
            key = str(group_id)
            minima[key] = min(minima.get(key, float("inf")), float(count))
        baseline = torch.tensor(
            [minima[str(group_id)] for group_id in group_ids],
            device=execution_counts.device,
            dtype=execution_counts.dtype,
        )
    excess = (execution_counts - baseline).clamp_min(0)
    penalties = float(cost) * excess.to(result.dtype) / float(max_nodes)
    rows = torch.arange(result.shape[0], device=result.device)
    result[rows, lengths] -= penalties
    return result


def required_hyper_relation_model(config: Any) -> Optional[str]:
    """Return HyPER's explicit ranker path or fail before a rollout starts."""
    if not bool(getattr(config, "hyper_r1_enable", False)):
        return None
    path = getattr(config, "hyper_r1_relation_model", None)
    if not path:
        raise ValueError(
            "HyPER-R1 requires hyper_r1_relation_model; "
            "silent relation-ranker fallback is disabled"
        )
    return str(path)
