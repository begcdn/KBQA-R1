# HyPER-R1

HyPER-R1 is a KG reasoning policy that delays irreversible commitment. A normal
relation request opens a bounded frontier of naturally ranked, executable
hypotheses. The policy can investigate one hypothesis while the others remain
available, prune hypotheses after execution supplies contrary evidence,
combine two necessary branches, and commit only an executable hypothesis that
covers the complete question.

## Executable protocol

1. `Find_relation [ source ]` asks the environment to execute the normal top-3
   question-conditioned relation proposals from one exact graph state. Gold relations are never
   inserted into this frontier. Unlike the released single-path resolver,
   HyPER-R1 does not reject low-similarity top candidates at a fixed threshold;
   uncertainty is represented by the bounded executable frontier itself.
2. `Widen [ source ]` exposes ranks 4--6 from that same cached natural ranking
   when the first batch does not cover the question. It is a policy decision,
   not automatic top-k inflation, and it incurs the same execution cost as other
   explored hypotheses.
3. `Select [ Hn ]` chooses the next hypothesis to investigate without deleting
   its siblings.
4. A subsequent `Find_relation` replaces that selected leaf with its executable
   children. Historical nodes remain in the graph.
5. `Prune [ Hn ]` removes an unsupported active hypothesis only after visible
   path or execution evidence contradicts it. The environment never silently
   prunes a low-scoring hypothesis for the policy.
6. `Combine [ Hn | Hm ]` executes the intersection of two retained branches.
7. `Commit [ Hn ]` is valid only for a nonempty active executable hypothesis.

The default frontier starts with three proposals and can widen to six, with at
most six active hypotheses and
at most twenty-four executed nodes. Rollouts allow ten model turns so the
longest retained recovery demonstration can still Commit and answer.

## Training data

The behavior-cloning corpus is built only from training-set gold programs.
Candidate sets come from the same relation retriever used at inference, and all
hypotheses and final answers are replayed through Freebase. Examples are
rejected when the natural top-6 misses a required relation; top-3 misses that are
recovered by natural ranks 4--6 become `Widen` demonstrations. Misses are
reported rather than repaired with gold injection.

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
- `adaptive_frontier_widen`: request a second ranked batch only when the first
  three proposals omit a required relation, then manage the bounded active set
  and finish from the recovered hypothesis;
- `conjunction`: retrieve two necessary branches in one natural frontier,
  combine the branches, and commit the executed intersection while unrelated
  plausible candidates remain available.

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
a still larger multi-hop corpus). Answer-equivalent alternatives are measured
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
single-state KBQA-R1 policy under the same model, data, graph backend, and
execution budget. Required diagnostics are candidate recall, answer F1/EM,
successful recovery after a non-gold first choice, conjunction accuracy,
execution calls, frontier size, and invalid/premature commits.
An always-top-6 control is required: HyPER-R1 only earns its adaptive-search
claim if it matches or improves answer quality with fewer executions than that
fixed wider frontier.
