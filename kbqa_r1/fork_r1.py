"""Counterfactual graph-decision credit for Fork-R1.

Fork-R1 changes one executable graph action at a time, continues both branches
with the same policy and budget, and assigns their terminal reward difference
only to the tokens that produced the factual action.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch


@dataclass(frozen=True)
class RelationCandidate:
    relation: str
    score: float


@dataclass(frozen=True)
class ForkDecision:
    sample_id: int
    turn: int
    action_index: int
    step_number: int
    entity_argument: str
    relation_prompt: str
    chosen_relation: str
    alternative_relation: str
    resolver_margin: float
    state_before: tuple[str, ...]
    expression_counter: int
    entities: tuple[tuple[str, str], ...]
    prompt: str
    raw_action: str

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["state_before"] = list(self.state_before)
        result["entities"] = [list(entity) for entity in self.entities]
        return result


def relation_name(selected: Any) -> str:
    """Return a relation id/name from KBQA-R1's resolver result formats."""
    if selected is None:
        return ""
    if isinstance(selected, str):
        return selected
    if isinstance(selected, Mapping):
        return str(selected.get("relation_id") or selected.get("relation_name") or "")
    return str(
        getattr(selected, "relation_id", None)
        or getattr(selected, "relation_name", None)
        or ""
    )


def ranked_candidates(
    candidates: Sequence[Any], scores: Optional[Sequence[float]] = None
) -> List[RelationCandidate]:
    """Normalize resolver candidates and keep their semantic ranking."""
    normalized: List[RelationCandidate] = []
    for index, candidate in enumerate(candidates):
        if isinstance(candidate, (tuple, list)):
            # KBQA-R1 stores (display_name, relation_id).
            relation = str(candidate[1] if len(candidate) > 1 else candidate[0])
        else:
            relation = relation_name(candidate)
        if not relation:
            continue
        score = float(scores[index]) if scores is not None and index < len(scores) else 0.0
        normalized.append(RelationCandidate(relation=relation, score=score))
    normalized.sort(key=lambda item: item.score, reverse=True)
    return normalized


def choose_hard_sibling(
    chosen_relation: str,
    candidates: Sequence[RelationCandidate],
) -> Optional[RelationCandidate]:
    """Choose the highest-scoring executable sibling distinct from the choice."""
    chosen = chosen_relation.strip()
    for candidate in candidates:
        if candidate.relation.strip() != chosen:
            return candidate
    return None


def build_fork_decision(
    *,
    sample_id: int,
    turn: int,
    action_index: int,
    step_number: int,
    entity_argument: str,
    relation_prompt: str,
    selected_relation: Any,
    candidates: Sequence[Any],
    scores: Optional[Sequence[float]],
    state_before: Iterable[str],
    expression_counter: int = 0,
    entities: Iterable[Sequence[str]] = (),
    prompt: str = "",
    raw_action: str,
) -> Optional[ForkDecision]:
    chosen = relation_name(selected_relation)
    ranked = ranked_candidates(candidates, scores)
    sibling = choose_hard_sibling(chosen, ranked)
    if not chosen or sibling is None:
        return None
    chosen_score = next((item.score for item in ranked if item.relation == chosen), 0.0)
    return ForkDecision(
        sample_id=sample_id,
        turn=turn,
        action_index=action_index,
        step_number=step_number,
        entity_argument=entity_argument,
        relation_prompt=relation_prompt,
        chosen_relation=chosen,
        alternative_relation=sibling.relation,
        resolver_margin=chosen_score - sibling.score,
        state_before=tuple(state_before),
        expression_counter=int(expression_counter),
        entities=tuple((str(entity[0]), str(entity[1])) for entity in entities),
        prompt=str(prompt),
        raw_action=raw_action,
    )


def select_intervention(decisions: Sequence[ForkDecision]) -> Optional[ForkDecision]:
    """Intervene at the most uncertain real graph decision in a trajectory."""
    if not decisions:
        return None
    return min(decisions, key=lambda decision: (abs(decision.resolver_margin), decision.turn))


def append_intervened_join(decision: ForkDecision) -> List[str]:
    """Build the alternative executable state from the exact pre-action state."""
    state = list(decision.state_before)
    entity = decision.entity_argument.strip()
    if entity.startswith("expression"):
        expression = entity
    else:
        ids = []
        for statement in state:
            head = statement.split("=", 1)[0].strip()
            if head.startswith("expression") and head[10:].isdigit():
                ids.append(int(head[10:]))
        expression = f"expression{max(ids, default=0) + 1}"
        state.append(f"{expression} = START('{entity}')")
    state.append(
        f"{expression} = JOIN('{decision.alternative_relation}', {expression})"
    )
    return state


def find_token_subsequence(sequence: Sequence[int], needle: Sequence[int]) -> Optional[tuple[int, int]]:
    if not needle or len(needle) > len(sequence):
        return None
    last = len(sequence) - len(needle) + 1
    for start in range(last):
        if list(sequence[start : start + len(needle)]) == list(needle):
            return start, start + len(needle)
    return None


def build_action_credit_mask(
    responses: torch.Tensor,
    raw_actions: Sequence[str],
    tokenizer: Any,
) -> torch.Tensor:
    """Locate each intervened action in the generated response tensor."""
    mask = torch.zeros_like(responses, dtype=torch.float32)
    for row, raw_action in enumerate(raw_actions):
        if not raw_action:
            continue
        action_ids = tokenizer(raw_action, add_special_tokens=False)["input_ids"]
        response_ids = responses[row].tolist()
        span = find_token_subsequence(response_ids, action_ids)
        if span is not None:
            mask[row, span[0] : span[1]] = 1.0
    return mask


def apply_counterfactual_credit(
    base_advantages: torch.Tensor,
    action_mask: torch.Tensor,
    factual_reward: torch.Tensor,
    counterfactual_reward: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    """Add paired terminal reward differences only on intervened action tokens."""
    if base_advantages.shape != action_mask.shape:
        raise ValueError("base_advantages and action_mask must have the same shape")
    if factual_reward.ndim != 1 or counterfactual_reward.ndim != 1:
        raise ValueError("factual and counterfactual rewards must be vectors")
    if factual_reward.shape != counterfactual_reward.shape:
        raise ValueError("factual and counterfactual rewards must have the same shape")
    if factual_reward.shape[0] != base_advantages.shape[0]:
        raise ValueError("reward vectors must match the batch size")

    delta = (factual_reward - counterfactual_reward).clamp(-1.0, 1.0)
    return base_advantages + float(weight) * delta.unsqueeze(-1) * action_mask
