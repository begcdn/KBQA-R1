# HyPER-R1

HyPER-R1 is a KG reasoning policy that delays irreversible commitment. A normal
relation request opens a bounded frontier of naturally ranked, executable
hypotheses. The policy can investigate one hypothesis while the others remain
available, prune hypotheses after execution supplies contrary evidence,
combine two necessary branches, and commit only an executable hypothesis that
covers the complete question.

## Fixed protocol

1. `Find_relation [ source | relation intent ]` executes the normal top-3
   relation proposals from one exact graph state. Gold relations are never
   inserted into this frontier.
2. `Select [ Hn ]` chooses the next hypothesis to investigate without deleting
   its siblings.
3. A subsequent `Find_relation` replaces that selected leaf with its executable
   children. Historical nodes remain in the graph.
4. `Prune [ Hn ]` removes an unsupported active hypothesis. The environment
   never silently prunes a low-scoring hypothesis for the policy.
5. `Combine [ Hn | Hm ]` executes the intersection of two retained branches.
6. `Commit [ Hn ]` is valid only for a nonempty active executable hypothesis.

The default frontier has three proposals, at most six active hypotheses, and
at most twenty-four executed nodes.

## Training data

The behavior-cloning corpus is built only from training-set gold programs.
Candidate sets come from the same relation retriever used at inference, and all
hypotheses and final answers are replayed through Freebase. Examples are
rejected when the natural frontier misses a required relation; this miss is
reported rather than repaired with gold injection.

The retained trajectory families teach distinct policy behavior:

- `frontier_commit`: open alternatives, select the complete executable branch,
  then commit;
- `delayed_frontier_recovery`: investigate a plausible wrong branch, execute
  its continuation, prune it after negative evidence, return to a preserved
  alternative, and finish correctly;
- `conjunction`: retrieve two necessary branches in one natural frontier,
  remove unrelated candidates, combine the branches, and commit the executed
  intersection.

The builder round-robins these families before filling the remaining corpus so
easy one-hop commits cannot erase the method-specific behavior.

```bash
export PROCESSED_GRAILQA=/path/to/processed/GrailQA_train.json
bash scripts/data_process/collect_hyper_r1_sft.sh

export BASE_MODEL=/path/to/base/model
bash scripts/train/train_hyper_r1_sft.sh
```

## Reinforcement learning

After SFT, the existing executable GRPO environment optimizes answer outcome.
Outcome advantage is concentrated on valid graph-control actions for every
advantage estimator, including GRPO. Credit includes useful exploration away
from the final lineage because such exploration is part of the policy. A
normalized execution penalty discourages indiscriminate search.

```bash
bash scripts/train/train_hyper_r1_grpo.sh
```

The scientific comparison is the complete method against the released
single-state KBQA-R1 policy under the same model, data, graph backend, and
execution budget. Required diagnostics are candidate recall, answer F1/EM,
successful recovery after a non-gold first choice, conjunction accuracy,
execution calls, frontier size, and invalid/premature commits.
