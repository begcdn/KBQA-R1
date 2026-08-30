"""Apply HyPER-R1 grammar masks to actor/reference logits."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from kbqa_r1.action_constraints import HyPERActionConstraintSpec


def masked_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Categorical entropy that treats masked zero-probability terms as zero."""
    floor = torch.finfo(logits.dtype).min
    finite_logits = torch.where(torch.isneginf(logits), floor, logits)
    probabilities = torch.softmax(finite_logits, dim=-1)
    return torch.logsumexp(finite_logits, dim=-1) - torch.sum(
        probabilities * finite_logits, dim=-1
    )


class HyPERXGrammarMasker:
    """Compile public action grammars and replay their token masks exactly."""

    def __init__(self, tokenizer: Any, vocab_size: int):
        try:
            import xgrammar as xgr
        except ImportError as exc:
            raise RuntimeError(
                "HyPER structural constraints require xgrammar in actor workers"
            ) from exc
        self.xgr = xgr
        self.vocab_size = int(vocab_size)
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            tokenizer, vocab_size=self.vocab_size
        )
        try:
            self.compiler = xgr.GrammarCompiler(
                tokenizer_info,
                cache_enabled=True,
                cache_limit_bytes=128 * 1024 * 1024,
            )
        except TypeError:
            # Older XGrammar builds have no bounded cache. Disabling it is
            # slower but prevents state-conditioned grammars accumulating.
            self.compiler = xgr.GrammarCompiler(
                tokenizer_info, cache_enabled=False
            )

    def _compiled_grammar(self, spec: HyPERActionConstraintSpec):
        return self.compiler.compile_regex(spec.response_pattern)

    def apply(
        self,
        logits: torch.Tensor,
        responses: torch.Tensor,
        turn_ids: torch.Tensor,
        constraint_rows: Sequence[Sequence[Mapping[str, Any]]],
        *,
        row_indices: torch.Tensor | None = None,
    ) -> None:
        """Mask logits in place and reject metadata/token disagreement.

        ``logits`` is either ``[batch, response, vocab]`` or remove-padding
        ``[tokens, vocab]``. In the latter case, ``row_indices`` maps each
        response token to the logit row that predicted it.
        """
        if logits.size(-1) != self.vocab_size:
            raise RuntimeError("HyPER constraint vocabulary does not match actor logits")
        if turn_ids.shape != responses.shape:
            raise RuntimeError("HyPER constraint turn IDs do not align with responses")

        for batch_index in range(responses.shape[0]):
            specs = {
                int(payload["turn"]): HyPERActionConstraintSpec.from_dict(payload)
                for payload in constraint_rows[batch_index]
            }
            constrained_turns = sorted(
                {int(value) for value in turn_ids[batch_index].tolist() if int(value) >= 0}
            )
            if constrained_turns != sorted(specs):
                raise RuntimeError(
                    "HyPER constraint metadata does not cover generated turns: "
                    f"tokens={constrained_turns} specs={sorted(specs)}"
                )
            for turn in constrained_turns:
                matcher = self.xgr.GrammarMatcher(
                    self._compiled_grammar(specs[turn]),
                    terminate_without_stop_token=True,
                )
                positions = torch.nonzero(
                    turn_ids[batch_index] == turn, as_tuple=False
                ).flatten().tolist()
                for position in positions:
                    try:
                        bitmask = self.xgr.allocate_token_bitmask(
                            1, self.vocab_size
                        )
                    except TypeError:
                        bitmask = self.xgr.allocate_token_bitmask(
                            batch_size=1, vocab_size=self.vocab_size
                        )
                    matcher.fill_next_token_bitmask(bitmask)
                    if row_indices is None:
                        target = logits[batch_index, position]
                    else:
                        row = int(row_indices[batch_index, position].item())
                        if row < 0:
                            raise RuntimeError(
                                "HyPER generated token has no remove-padding logit row"
                            )
                        target = logits[row]
                    self.xgr.apply_token_bitmask_inplace(
                        target.unsqueeze(0), bitmask.to(target.device)
                    )
                    token_id = int(responses[batch_index, position].item())
                    if not matcher.accept_token(token_id):
                        raise RuntimeError(
                            "HyPER sampled token violates its stored action grammar: "
                            f"turn={turn} position={position} token={token_id}"
                        )
                if not matcher.is_terminated():
                    raise RuntimeError(
                        "HyPER generated turn ended before completing its action grammar: "
                        f"turn={turn}"
                    )
