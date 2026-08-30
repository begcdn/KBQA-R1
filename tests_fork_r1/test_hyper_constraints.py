import importlib.util
from pathlib import Path

import pytest
import torch

from kbqa_r1.action_constraints import HyPERActionConstraintSpec


_MODULE_SPEC = importlib.util.spec_from_file_location(
    "hyper_constraints_under_test",
    Path(__file__).parents[1] / "verl" / "utils" / "hyper_constraints.py",
)
_MODULE = importlib.util.module_from_spec(_MODULE_SPEC)
_MODULE_SPEC.loader.exec_module(_MODULE)
HyPERXGrammarMasker = _MODULE.HyPERXGrammarMasker
masked_entropy_from_logits = _MODULE.masked_entropy_from_logits


class _Matcher:
    def __init__(self, compiled, *, terminate_without_stop_token=False):
        self.expected = list(compiled)
        self.position = 0
        self.terminate_without_stop_token = terminate_without_stop_token

    def fill_next_token_bitmask(self, bitmask):
        bitmask.zero_()
        bitmask[0, self.expected[self.position]] = True

    def accept_token(self, token_id):
        if token_id != self.expected[self.position]:
            return False
        self.position += 1
        return True

    def is_terminated(self):
        return self.terminate_without_stop_token and self.position == len(self.expected)


class _FakeXGrammar:
    GrammarMatcher = _Matcher

    @staticmethod
    def allocate_token_bitmask(batch_size, vocab_size):
        return torch.zeros((batch_size, vocab_size), dtype=torch.bool)

    @staticmethod
    def apply_token_bitmask_inplace(logits, bitmask):
        logits.masked_fill_(~bitmask, float("-inf"))


def _payload(turn=0):
    spec = HyPERActionConstraintSpec(
        state_key=f"state-{turn}",
        turn=turn,
        exact_actions=("Select [ H0 ]",),
        allow_open_operators=False,
    )
    return spec.to_dict()


def _masker(expected_by_digest):
    value = HyPERXGrammarMasker.__new__(HyPERXGrammarMasker)
    value.xgr = _FakeXGrammar
    value.vocab_size = 5
    value._compiled_grammar = lambda spec: expected_by_digest[spec.digest]
    return value


def test_dense_logits_are_masked_turn_by_turn():
    payload = _payload()
    masker = _masker({payload["digest"]: [1, 2]})
    logits = torch.zeros((1, 2, 5))

    masker.apply(
        logits,
        responses=torch.tensor([[1, 2]]),
        turn_ids=torch.tensor([[0, 0]]),
        constraint_rows=[[payload]],
    )

    assert torch.isfinite(logits[0, 0]).tolist() == [False, True, False, False, False]
    assert torch.isfinite(logits[0, 1]).tolist() == [False, False, True, False, False]


def test_remove_padding_rows_use_the_same_mask():
    payload = _payload()
    masker = _masker({payload["digest"]: [1, 2]})
    logits = torch.zeros((4, 5))

    masker.apply(
        logits,
        responses=torch.tensor([[1, 2]]),
        turn_ids=torch.tensor([[0, 0]]),
        constraint_rows=[[payload]],
        row_indices=torch.tensor([[3, 1]]),
    )

    assert torch.isfinite(logits[3]).tolist() == [False, True, False, False, False]
    assert torch.isfinite(logits[1]).tolist() == [False, False, True, False, False]
    assert torch.isfinite(logits[0]).all()


def test_missing_turn_metadata_fails_closed():
    payload = _payload(turn=1)
    masker = _masker({payload["digest"]: [1]})

    with pytest.raises(RuntimeError, match="does not cover generated turns"):
        masker.apply(
            torch.zeros((1, 1, 5)),
            responses=torch.tensor([[1]]),
            turn_ids=torch.tensor([[0]]),
            constraint_rows=[[payload]],
        )


def test_sampled_token_outside_mask_fails_closed():
    payload = _payload()
    masker = _masker({payload["digest"]: [1]})

    with pytest.raises(RuntimeError, match="sampled token violates"):
        masker.apply(
            torch.zeros((1, 1, 5)),
            responses=torch.tensor([[2]]),
            turn_ids=torch.tensor([[0]]),
            constraint_rows=[[payload]],
        )


def test_incomplete_action_prefix_fails_closed():
    payload = _payload()
    masker = _masker({payload["digest"]: [1, 2]})

    with pytest.raises(RuntimeError, match="ended before completing"):
        masker.apply(
            torch.zeros((1, 1, 5)),
            responses=torch.tensor([[1]]),
            turn_ids=torch.tensor([[0]]),
            constraint_rows=[[payload]],
        )


def test_masked_entropy_is_finite_and_ignores_impossible_tokens():
    logits = torch.tensor([[0.0, float("-inf"), 0.0]], requires_grad=True)
    entropy = masked_entropy_from_logits(logits)

    assert torch.isfinite(entropy).all()
    assert entropy.item() == pytest.approx(0.693147, rel=1e-5)
    entropy.sum().backward()
    assert torch.isfinite(logits.grad[[0], [0, 2]]).all()
