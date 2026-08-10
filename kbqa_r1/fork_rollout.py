"""Native branch evaluation used by Fork-R1 rollouts.

The engine is deliberately independent of gold logical forms. Its only
supervision is the ordinary terminal answer reward already used by KBQA-R1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from .fork_r1 import ForkDecision, select_intervention


@dataclass(frozen=True)
class ForkOutcome:
    decision: ForkDecision
    factual_reward: float
    counterfactual_reward: float
    counterfactual_trajectory: Any

    @property
    def credit(self) -> float:
        return max(-1.0, min(1.0, self.factual_reward - self.counterfactual_reward))


class CounterfactualRolloutEngine:
    """Evaluate one uncertain action by continuing its hard sibling branch.

    ``continue_alternative`` must restore ``decision.state_before``, execute
    ``decision.alternative_relation``, and continue with the frozen rollout
    policy under the same remaining turn budget. This interface keeps graph
    execution in KBQA-R1 while making the scientific invariant testable.
    """

    def __init__(
        self,
        continue_alternative: Callable[[ForkDecision], Any],
        score_trajectory: Callable[[Any], float],
    ):
        self.continue_alternative = continue_alternative
        self.score_trajectory = score_trajectory

    def evaluate(
        self,
        decisions: Sequence[ForkDecision],
        factual_reward: float,
    ) -> Optional[ForkOutcome]:
        decision = select_intervention(decisions)
        if decision is None:
            return None
        alternative = self.continue_alternative(decision)
        reward = float(self.score_trajectory(alternative))
        return ForkOutcome(
            decision=decision,
            factual_reward=float(factual_reward),
            counterfactual_reward=reward,
            counterfactual_trajectory=alternative,
        )
