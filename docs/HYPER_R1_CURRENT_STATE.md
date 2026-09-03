# HyPER-R1 Current State

Updated: 2026-08-31

This file is the authoritative handoff for the current experiment. Read it
before proposing another training run.

## Research Objective

HyPER-R1 is a KGQA search policy that preserves and revisits competing graph
hypotheses instead of irreversibly committing to one relation path. The final
goal is improved answer F1 on strong KGQA benchmarks, not action imitation by
itself.

## What Has Been Established

The v23 corpus repaired the old teacher/runtime prompt mismatch: it uses dense
public H/P IDs, the live observation and turn clock, executable action
affordances, and answer-F1-aligned Commit targets.

The first 500-step LoRA probe is invalid because its adapters contained zero
tensors. The repaired probe saved 448 tensors and showed a consistent learning
signal, but it was never evidence of autonomous policy quality.

A full non-LoRA v23 SFT run subsequently completed. Held-out action screens
were strong:

| Checkpoint | Parsable | Action type | Exact action | Early Commit |
|---|---:|---:|---:|---:|
| step 2500 | 1.000 | 0.947 | 0.883 | 0.000 |
| step 5000 | 1.000 | 0.939 | 0.898 | 0.000 |

These screens prove that the policy learned teacher-state action behavior. They
do not prove that it can recover from its own autonomous states.

## Autonomous Evaluation Finding

The executable SFT-only evaluation was paused at 1,288 questions. Preliminary
records showed approximately:

- fallback-assisted answer F1: 0.781;
- trajectories with an invalid action: 14.8%;
- repeated-action trajectories: 10.6%;
- forced terminal/exhaustion: 17.1% under the old instrumentation.

These figures identify a real autonomous-control problem, but the old run is
not a clean policy-quality estimate. Evaluation used 16 turns while SFT and
GRPO used 32, forced fallback answers were mixed with explicit model Commit,
and forced trajectories were removed from actor learning. Exact historical
turn exhaustion cannot be recovered because the old dumps did not preserve a
terminal reason.

The dominant stale/unknown-ID behavior is exposure bias plus unrestricted text
generation, not a return of the old prompt mismatch.

## Runtime Foundation Now Repaired

The current branch contains these correctness repairs:

- one canonical source for graph-control affordances and SFT affordance gates;
- one strict HyPER response parser accepting exactly one complete action;
- a 32-turn HyPER default in data collection, evaluation, and GRPO;
- explicit terminal provenance for model Commit, forced candidate, forced
  empty, turn exhaustion, empty generation, and execution timeout;
- forced-terminal reward `fallback F1 - 0.25` instead of censoring the rollout;
- forced rollouts excluded only from same-state sibling credit, not from their
  base trajectory advantage;
- separate fallback-assisted, policy-only, explicit-Commit, and forced-terminal
  metrics.

Do not use the old 0.781 figure as explicit-policy F1.

## Structural Decoding Gate

The current branch implements a typed state-conditioned grammar. Finite graph
actions use the exact public affordance lists, while logical operators constrain
their structural fields and retain only the semantic spans that the public
state cannot enumerate. The same stored turn grammar is replayed for rollout,
old-policy, actor, and reference logits.

A live two-question gate on the step-7500 checkpoint passed with vLLM 0.10.2,
XGrammar, the production tokenizer, the relation ranker, and Virtuoso:

- 16 generated actions, all accepted by the runtime validator;
- zero unknown/stale IDs, malformed actions, replay disagreements, or loops;
- two explicit model commits and no forced terminal or turn exhaustion;
- exact committed answer sets and commit-answer F1 of 1.0 on both questions.

This is an implementation and feasibility gate, not an answer-quality result.
Two questions cannot estimate benchmark F1 or prove that the policy has learned
good long-horizon exploration.

## Corrective Evidence Collected

The step-7500 policy completed 995 unmasked training-split rollouts. It
produced 756 explicit exact-answer trajectories. Exact-state executable
certification retained 174 unique semantic recovery decisions: in every row,
the alternative action is legal in the same visible state and reaches the
global semantic-utility upper bound under the same remaining budget.

Protocol-invalid actions are now removed by structural decoding, and the
available rollout corpus does not contain enough independently certified
protocol corrections to justify the earlier 15% allocation. The compact
corrective mixture is therefore:

- 65% stratified ordinary v23 decisions;
- 25% successful autonomous decisions;
- 10% legal-semantic recovery states certified by executable regret;
- at most one recovery state per trajectory;
- question-disjoint correction train/dev splits;
- no test failures or test questions in training.

## Next Experiment

Build a 1,500-row train and 140-row question-disjoint dev corpus from the
unique evidence above. Starting from the full step-7500 checkpoint, run a
200-step full-parameter corrective SFT at a low learning rate, saving every 50
steps. This is about four passes over the compact corpus; the earlier
1,500-2,000-step estimate would excessively repeat the 174 recovery labels.

Evaluate these pre-GRPO arms on correction-dev and held-out executable
rollouts:

- A0: current SFT, unmasked;
- A1: current SFT, masked;
- A2: corrective SFT, unmasked;
- A3: corrective SFT, masked.

Only then run a 5-10% masked GRPO pilot. Promote it only if explicit Commit,
loops, exhaustion, execution cost, and paired answer F1 all pass their gates.
