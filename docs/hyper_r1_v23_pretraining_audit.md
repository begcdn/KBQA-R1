# HyPER-R1 v23 pre-training audit

## Decision

The v23 corpus is ready for a short SFT and executable rollout screen. This is
not approval for full SFT or GRPO. Promotion still depends on live answer F1,
valid-action behavior, and recovery behavior after the short run.

## Repaired learning contracts

- Training and runtime now share one chat-template observation renderer.
- Deadline examples preserve the live 32-turn horizon instead of redefining a
  short trajectory as its own deadline.
- Commit comparisons delay a real, independent executed decoy. The original
  answer-and-intent-certified hypothesis remains the target, so the examples
  break recency without inventing denotations or semantic labels.
- Invalid-action recovery uses failures derivable from the visible runtime
  state. Fabricated `H999999` and `P999999` identifiers are absent.
- Comparison and deadline variants train only their new terminal decisions;
  their shared prefixes are not duplicated in SFT.

## Full-corpus results

The corpus contains 284,969 training decisions and 14,499 question-disjoint
validation decisions. All 59,219 generated demonstrations replay through the
live graph state machine with no failures and no contradictory decision states.

- 27,211 delayed-decoy Commit comparisons were generated. In every comparison,
  the certified target is older than the fresh decoy.
- Across 56,085 multi-candidate Commit states, the newest candidate is selected
  49.67% of the time, removing the deterministic recency shortcut.
- 2,150 deadline decisions occur with at most three turns remaining under the
  original 32-turn contract; every generated deadline decision meets this test.
- 2,198 recovery decisions cover five real failure modes: repeated Select,
  Recall of an active node, Commit of an empty node, Select of a parked node,
  and Widen of an exhausted catalog.

The exact Llama 3.1 chat template and tokenizer were applied to all 299,468
decision rows. Runtime and SFT contexts matched exactly. The longest row was
3,076 tokens under the configured 32,768-token limit, with zero overlength rows
and zero missing supervised targets.

The machine-readable measurements and all regeneration gates are in
`docs/hyper_r1_v23_corpus_report.json`.

## Next gate

Run a short SFT only. Select its checkpoint using executable rollout answer F1
and protocol behavior, not validation imitation accuracy alone. Do not start
full SFT or GRPO unless the live screen improves over the v15 initializer and
does not reproduce invalid-action loops, stale IDs, or deadline-only Commit
bias.
