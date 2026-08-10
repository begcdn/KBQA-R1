"""Executable hypothesis graph used by HyPER-R1.

The graph is deliberately independent of the LLM and the SPARQL backend.  The
environment executes a candidate first and then records the resulting logical
form and denotation here.  This keeps graph bookkeeping deterministic and makes
the same state usable during SFT, RL, and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import re

import torch


class HypothesisStatus(str, Enum):
    ACTIVE = "active"
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
    committed_id: Optional[str] = None
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


_EXPRESSION_RE = re.compile(r"\bexpression\d+\b")


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
        int(name[len("expression") :])
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
        if not re.fullmatch(r"expression\d+", lhs):
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

    def __init__(self, max_active: int = 6, max_nodes: int = 24):
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
        operation: str,
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
        if parent_id is not None and parent_id not in graph.nodes:
            raise KeyError(f"unknown parent hypothesis: {parent_id}")

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
            parent_id=parent_id,
            operation=str(operation),
            relation_id=relation_id,
            relation_prompt=relation_prompt,
            resolver_score=float(resolver_score or 0.0),
            depth=parent_depth + 1,
            provenance=list(provenance or ()),
        )
        graph.nodes[node_id] = node
        graph.execution_calls += 1

        if parent_id is not None:
            graph.edges.append(
                HypothesisEdge(
                    source=parent_id,
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
                    label="same_denotation",
                )
            )

        self._enforce_active_budget(graph)
        return node

    def _find_equivalent(
        self, graph: HypothesisGraphState, candidate: HypothesisNode
    ) -> Optional[HypothesisNode]:
        if not candidate.denotation:
            return None
        for node in graph.nodes.values():
            if node.node_id == candidate.node_id or not node.is_active:
                continue
            if node.denotation == candidate.denotation:
                return node
        return None

    def _enforce_active_budget(self, graph: HypothesisGraphState) -> None:
        active = [node for node in graph.nodes.values() if node.is_active]
        if len(active) <= self.max_active:
            return
        # The environment applies only the hard memory bound.  The language
        # policy is expected to prune deliberately before reaching it.
        active.sort(key=lambda node: (node.resolver_score, -node.depth, node.node_id))
        for node in active[: len(active) - self.max_active]:
            node.status = HypothesisStatus.PRUNED
            node.provenance.append("environment_budget")

    def prune(self, sample_id: int, node_id: str, reason: str = "policy") -> None:
        node = self.require_active(sample_id, node_id)
        node.status = HypothesisStatus.PRUNED
        node.provenance.append(reason)

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
        graph.committed_id = node_id
        return node

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
        lines = [
            "<hypothesis_graph>",
            f"active={len(self.active_nodes(sample_id))} "
            f"nodes={len(graph.nodes)} executions={graph.execution_calls}",
        ]
        for node in graph.nodes.values():
            if node.status not in {HypothesisStatus.ACTIVE, HypothesisStatus.COMMITTED}:
                continue
            answers = ", ".join(node.denotation[:max_answers]) or "empty"
            if len(node.denotation) > max_answers:
                answers += f", ... (+{len(node.denotation) - max_answers})"
            parent = node.parent_id or "ROOT"
            rel = node.relation_id or node.operation
            source = (
                "policy"
                if "policy_choice" in node.provenance
                else "sibling"
                if "hard_sibling" in node.provenance
                else "derived"
            )
            lines.append(
                f"{node.node_id} [{node.status.value}] parent={parent} "
                f"via={rel} source={source} depth={node.depth} "
                f"answers={len(node.denotation)}: {answers}"
            )
        lines.append(
            "Actions: Explore/Expand, Combine, Prune, or Commit one active hypothesis."
        )
        lines.append("</hypothesis_graph>")
        return "\n".join(lines)

    def to_dict(self, sample_id: int) -> dict:
        graph = self.state(sample_id)
        return {
            "sample_id": sample_id,
            "committed_id": graph.committed_id,
            "execution_calls": graph.execution_calls,
            "nodes": [
                {
                    **node.__dict__,
                    "function_state": list(node.function_state),
                    "denotation": list(node.denotation),
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


def concentrate_graph_credit(
    advantages: torch.Tensor,
    action_mask: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    """Increase outcome-credit magnitude only on committed graph decisions."""
    if advantages.shape != action_mask.shape:
        raise ValueError("advantages and action_mask must have the same shape")
    return advantages * (1.0 + float(weight) * action_mask.to(advantages.dtype))


def charge_execution_budget(
    token_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    execution_counts: torch.Tensor,
    max_nodes: int,
    cost: float,
) -> torch.Tensor:
    """Charge one normalized search cost on the last generated token."""
    if token_rewards.shape != response_mask.shape:
        raise ValueError("token_rewards and response_mask must have the same shape")
    if execution_counts.ndim != 1 or execution_counts.shape[0] != token_rewards.shape[0]:
        raise ValueError("execution_counts must be one value per rollout")
    if max_nodes <= 0:
        raise ValueError("max_nodes must be positive")
    result = token_rewards.clone()
    lengths = response_mask.long().sum(dim=-1).clamp_min(1) - 1
    penalties = float(cost) * execution_counts.to(result.dtype) / float(max_nodes)
    rows = torch.arange(result.shape[0], device=result.device)
    result[rows, lengths] -= penalties
    return result
