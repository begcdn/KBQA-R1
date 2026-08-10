import unittest

import torch

from kbqa_r1.fork_r1 import (
    ForkDecision,
    RelationCandidate,
    append_intervened_join,
    apply_counterfactual_credit,
    choose_hard_sibling,
    find_token_subsequence,
    select_intervention,
)
from kbqa_r1.fork_rollout import CounterfactualRolloutEngine


def decision(**overrides):
    values = {
        "sample_id": 0,
        "turn": 1,
        "action_index": 0,
        "step_number": 2,
        "entity_argument": "expression1",
        "relation_prompt": "place where the person died",
        "chosen_relation": "people.deceased_person.place_of_death",
        "alternative_relation": "people.person.place_of_birth",
        "resolver_margin": 0.1,
        "state_before": (
            "expression1 = START('m.01')",
            "expression1 = JOIN('people.person.parents', expression1)",
        ),
        "expression_counter": 1,
        "entities": (("m.01", "Person"),),
        "prompt": "Who was the parent and where did they die?",
        "raw_action": "Find_relation [expression1 | place where the person died]",
    }
    values.update(overrides)
    return ForkDecision(**values)


class ForkR1Test(unittest.TestCase):
    def test_hard_sibling_is_best_non_chosen_candidate(self):
        candidates = [
            RelationCandidate("chosen", 0.9),
            RelationCandidate("hard", 0.85),
            RelationCandidate("easy", 0.2),
        ]
        self.assertEqual(choose_hard_sibling("chosen", candidates).relation, "hard")

    def test_intervention_targets_smallest_resolver_margin(self):
        selected = select_intervention(
            [decision(resolver_margin=0.3), decision(turn=2, resolver_margin=0.02)]
        )
        self.assertEqual(selected.turn, 2)

    def test_intervened_join_preserves_prefix_and_changes_one_relation(self):
        original = decision()
        state = append_intervened_join(original)
        self.assertEqual(state[:-1], list(original.state_before))
        self.assertEqual(
            state[-1],
            "expression1 = JOIN('people.person.place_of_birth', expression1)",
        )

    def test_entity_fork_adds_start_before_join(self):
        original = decision(entity_argument="m.02")
        state = append_intervened_join(original)
        self.assertEqual(state[-2], "expression2 = START('m.02')")
        self.assertEqual(
            state[-1],
            "expression2 = JOIN('people.person.place_of_birth', expression2)",
        )

    def test_credit_changes_only_action_tokens(self):
        base = torch.zeros((2, 4))
        mask = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.float32)
        result = apply_counterfactual_credit(
            base,
            mask,
            factual_reward=torch.tensor([1.0, 0.0]),
            counterfactual_reward=torch.tensor([0.25, 1.0]),
            weight=2.0,
        )
        expected = torch.tensor([[0, 1.5, 1.5, 0], [-2.0, 0, 0, 0]])
        torch.testing.assert_close(result, expected)

    def test_token_subsequence(self):
        self.assertEqual(find_token_subsequence([3, 4, 5, 6], [4, 5]), (1, 3))
        self.assertIsNone(find_token_subsequence([3, 4], [5]))

    def test_rollout_engine_scores_the_most_uncertain_sibling(self):
        decisions = [decision(resolver_margin=0.4), decision(turn=3, resolver_margin=0.01)]
        engine = CounterfactualRolloutEngine(
            continue_alternative=lambda selected: {"turn": selected.turn, "reward": 0.2},
            score_trajectory=lambda trajectory: trajectory["reward"],
        )
        outcome = engine.evaluate(decisions, factual_reward=0.9)
        self.assertEqual(outcome.decision.turn, 3)
        self.assertAlmostEqual(outcome.credit, 0.7)


if __name__ == "__main__":
    unittest.main()
