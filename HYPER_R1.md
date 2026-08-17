# HyPER-R1

HyPER-R1 is a KG reasoning policy that delays irreversible commitment. A normal
relation request opens a bounded frontier of naturally ranked, executable
hypotheses. The policy can investigate one hypothesis while the others remain
available, prune hypotheses after execution supplies contrary evidence,
combine two necessary branches, and commit only an executable hypothesis that
covers the complete question.

## Executable protocol

1. `Find_relation [ source ]` structurally enumerates every legal local relation,
   ranks that complete list from the immutable question and exact executable
   state, and executes its first stable page of six. Gold relations are never
   inserted and similarity scores never delete candidates.
2. `Widen [ source ]` repeatedly exposes the next six relations from the same
   cached ordering. It is a policy decision, not automatic top-k inflation, and
   it incurs the same execution cost as other explored hypotheses. The only
   practical cutoff is the shared rollout node budget.
3. `Select [ Hn ]` chooses the next hypothesis to investigate without deleting
   its siblings.
4. A subsequent `Find_relation` replaces that selected leaf with its executable
   children. Historical nodes remain in the graph.
5. `Prune [ Hn ]` removes an unsupported active hypothesis only after visible
   path or execution evidence contradicts it. The environment never silently
   prunes a low-scoring hypothesis for the policy.
6. `Combine [ Hn | Hm ]` executes the intersection of two retained branches.
7. The released executable logical actions remain part of the policy grammar:
   `Order`, `Compare`, `Time_constraint`, and `Count`. They transform a selected
   branch; comparison and ontology-order constraints may open an independent
   branch for later combination.
8. `Commit [ Hn ]` is valid only for a nonempty active executable hypothesis.

The default page contains six proposals. At most twenty-four hypotheses may be
active or executed during a rollout, and rollouts allow sixteen model turns.
Low rank and lack of capacity are not valid reasons to prune a plausible branch;
a trace that cannot fit this uniform budget is rejected from supervision.

## Training data

The behavior-cloning corpus is built only from training-set gold programs.
Candidate sets come from the same relation retriever used at inference, and all
hypotheses and final answers are replayed through Freebase. The complete
structurally legal relation list is ranked identically during data construction
and inference. Teachers expose stable pages only through the page containing a
required relation. Examples are rejected when a required relation is
structurally absent or cannot be reached within the same 24-node and 16-turn
budget. Misses are reported rather than repaired with gold injection.

The proposal query contains only the immutable question and the visible
executable state. Gold programs choose and verify teacher actions but never add
a missed relation to a frontier. Entity evidence is serialized as readable
`label [MID]` pairs: labels support semantic decisions while MIDs preserve exact
Freebase identity. When Freebase has no English name, the serializer uses an
explicit type descriptor such as `unnamed cheese [MID]` rather than inventing a
name; reports count names and type fallbacks separately. Candidate roots are
placed in a stable question-hash order, so their list position cannot reveal
which gold branch should be opened first.

The retained trajectory families teach distinct policy behavior:

- `frontier_commit`: open alternatives and commit the complete executable branch;
- `direct_frontier_progress`: select and continue a supported multi-hop branch
  while preserving its siblings;
- `delayed_frontier_recovery`: investigate a plausible wrong branch, execute
  its naturally retrieved continuation frontier, prune an empty child after
  visible negative evidence, return to a preserved
  alternative, and finish correctly;
- `semantic_frontier_recovery`: recover when a plausible branch remains
  nonempty but its visible relation path conflicts with the question;
- `adaptive_frontier_widen`: request later stable relation pages when the first
  page omits a required relation, then finish from the recovered hypothesis;
- `conjunction`: retrieve two necessary branches in one natural frontier,
  combine the branches, and commit the executed intersection while unrelated
  plausible candidates remain available.
- `operator_program`: replay a complete operator-bearing program through the
  same public actions used at inference, retaining relation alternatives,
  applying `Order`, `Compare`, `Time_constraint`, or `Count`, combining branches
  where required, and committing only after exact answer replay succeeds.

Each complete student-visible observation has one teacher action. Recovery
examples are not paired with direct-progress twins from the identical frontier;
that would train contradictory choices at the decision where the method is
supposed to learn whether to investigate or proceed. Likewise, if a required
conjunction frontier cannot be built, none of its individual branches is
relabelled as a complete question-answering trajectory.

The builder round-robins these families before filling the remaining corpus so
easy one-hop commits cannot erase the method-specific behavior. Exported
reports also block expensive SFT unless readable evidence covers at least 95%
of displayed MIDs, no identical observation has conflicting teacher actions,
and the verified corpus contains at least 500 recovery trajectories (or 10% of
a still larger multi-hop corpus). If the input contains logical operators, the
export also requires every observed runtime operator kind to survive into the
training split. Answer-equivalent alternatives are measured
at every executed frontier and at Commit; equality of denotation alone never
turns a semantically different path into a positive teacher action. Questions
with more than 100 answers are omitted from SFT to prevent answer copying from
dominating policy learning. A deterministic question-level 95/5 split writes
`train.parquet` and `validation.parquet` without trajectory leakage. The default
SFT input is instead `train_decision.parquet`: one exact conversation prefix per
graph action. This retains the full observable state while excluding the final
answer-copy turn from the behavior-cloning objective.

```bash
export PROCESSED_GRAILQA=/path/to/processed/GrailQA_train.json
bash scripts/data_process/collect_hyper_r1_sft.sh

export BASE_MODEL=/path/to/base/model
bash scripts/train/train_hyper_r1_sft.sh
```

## Reinforcement learning

After SFT, the executable GRPO environment optimizes answer outcome. Reward is
valid only when the final answer exactly matches an explicit executable Commit.
For sibling rollouts that take different actions from the same semantic
frontier state, their terminal-reward difference is assigned to those action
tokens. Actions without a matched alternative receive no invented local
credit. A normalized execution penalty discourages indiscriminate search.

```bash
export HYPER_SFT_MODEL=/path/to/hyper-r1-sft/checkpoint
bash scripts/train/train_hyper_r1_grpo.sh
```

The scientific comparison is the complete method against the released
single-state KBQA-R1 policy and a fixed beam control under the same model, data,
graph backend, relation ordering, and scored-expansion budget. Required
diagnostics are structural candidate recall, answer F1/EM, successful recovery
after a non-gold first choice, conjunction/operator accuracy, execution calls,
frontier size, and invalid or premature commits. HyPER-R1 earns its claim only
if learned retention and delayed commitment improve answers beyond merely
executing a wider fixed set of relations.
