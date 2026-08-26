# HyPER-R1 Current State

Updated: 2026-08-26

This file is the authoritative handoff for the current experiment. Read it
before proposing another training run.

## Research Objective

HyPER-R1 is a KGQA search policy that preserves and revisits competing graph
hypotheses instead of irreversibly committing to one relation path. The final
goal is improved answer F1 on strong KGQA benchmarks, not action imitation by
itself.

## Why v23 Exists

Earlier supervised policies looked competent on teacher states but behaved
incorrectly during autonomous execution. Observed failures included repeated
invalid actions, stale or unknown proposal IDs, repeated state-action loops,
committing the wrong stored hypothesis, and exhausting the turn budget.

The v23 corpus was rebuilt to align demonstrations with the live runtime:
dense public IDs, the same observations and turn clock, executable action
affordances, runtime-comparison labels, and answer-F1-aligned commit behavior.

## Invalid Results

- The first v23 500-step LoRA run saved 40-byte adapters containing zero
  tensors. Its behavioral screen evaluated the untouched base model and must
  never be cited.
- The exporter filtered LoRA weights twice. Commit `449eae9` fixes this and
  makes both saving and evaluation reject empty adapters.

## Valid LoRA Signal Probe

Run: `grailqa-hyper-r1-v23-lora500-fixed`

- Model: Llama-3.1-8B-Instruct with rank-32 LoRA
- Updates: 500 of 8,905 batches, about 5.6% of one epoch
- Checkpoints: steps 100, 200, 300, 400, and 500
- Every checkpoint contains 448 tensors and 83,886,080 parameters

Held-out step-500 behavior:

- Parsable action: 100.0%
- Action type: 64.5%
- Exact action: 55.1%
- Deep-state type: 59.6%
- Deep-state exact: 55.1%
- Premature commit: 0.8%

The progression from step 100 to 500 is consistently positive. This establishes
that v23 supervision produces learnable behavior. It is not approval for full
SFT or GRPO. Rare central actions remain underlearned: `Widen` and `Combine`
were never predicted at step 500.

## The Two-Part Probe Gate

The short LoRA probe has two purposes:

1. Show that the intended policy behavior is learnable.
2. Show that learned behavior remains coherent under autonomous live runtime
   transitions.

Part 1 passed. Part 2 has not been run.

Single-state imitation cannot reveal temporal failures such as loops or stale
IDs because every example begins from a clean teacher state. Do not authorize
full non-LoRA SFT from the held-out action screen alone.

## Immediate Next Experiment

Run step 500 on 20-50 questions through the exact executable inference runtime.
Record:

- repeated invalid actions;
- stale or unknown IDs;
- repeated state-action loops;
- turn or execution-budget exhaustion;
- valid commits and agreement between the committed hypothesis and answer;
- answer F1;
- use of `Widen`, `Combine`, branch switching, and alternative preservation.

Interpret failures carefully. Failure to use rare actions after only 500 LoRA
updates is limited training. Malformed, impossible, stale, or looping behavior
is a train/runtime mismatch and blocks full SFT.

## Decision Sequence

1. Complete the small live-runtime gate.
2. If old mismatch failures are absent, run the planned full non-LoRA v23 SFT.
3. Select checkpoints using held-out behavior and live answer F1, not training
   loss alone.
4. Run benchmark-scale executable evaluation.
5. Start GRPO only after the SFT policy is operationally coherent and provides
   a sound initializer.

Do not extend the LoRA probe merely to satisfy the old action-screen threshold.
Its purpose is risk detection before the actual full SFT.
