# HyPER-R1: Hypothesis-Preserving Executable Reasoning

## Problem

Agentic KGQA systems usually commit to one relation at each turn. Exact KG
execution can reveal that a choice is weak, but by then the competing relation
has been discarded. Backtracking methods recover an old state after failure;
beam retrievers keep several paths outside the language policy. Neither gives a
trainable agent a persistent, executable representation of its alternatives.

## Method

HyPER-R1 replaces the agent's linear reasoning state with an **executable
hypothesis graph**.

- A hypothesis node contains a partial S-expression, its exact denotation, its
  parent operation, and its status.
- A contrast edge joins hypotheses produced by different relations from the
  same pre-action state.
- A composition edge records a logical operation such as intersection, count,
  comparison, or ordering.
- An equivalence edge merges hypotheses with the same executable denotation
  without erasing their different relation histories.

The language policy receives a compact serialization of the active graph and
uses the existing relation operation plus four graph actions:

1. `Find_relation [source | relation description]` executes the resolved
   relation and one hard sibling from the same pre-action state.
2. `Select [hypothesis]` restores the exact state of one active hypothesis so
   subsequent reasoning expands it.
3. `Combine [left | right]` constructs and executes an intersection.
4. `Prune [hypothesis]` rejects a hypothesis while retaining its provenance.
5. `Commit [hypothesis]` returns one executable hypothesis as the answer.

The environment, not the LLM, owns node identities, execution, deduplication,
and the active-set budget. This prevents malformed graph edits and keeps every
reported answer executable.

## Training

Training has two stages.

### Structured warm start

The released strong KBQA-R1 policy performs referenced rejection sampling in
the HyPER-R1 environment. Each successful trajectory therefore contains real
executions of the policy branch and a hard sibling from the same state. A
deterministic converter inserts `Select` and `Commit` demonstrations using only
the hypothesis identifiers returned by the environment. SFT on these traces
teaches the graph protocol without fabricating relations or denotations. It
does not teach that the policy branch is always best: GRPO remains responsible
for learning when a retained sibling should replace it.

### Reinforcement learning

The policy is optimized with the existing group-relative objective and exact
answer F1. Two graph-native signals are added:

- **decision credit:** group-relative terminal advantage is concentrated on
  actions in the committed hypothesis lineage;
- **budget reward:** a small cost is charged per executed expansion, so keeping
  alternatives is useful only when it improves the final answer.

No gold logical form, relation, or answer is exposed at inference time.

## Relation to Prior Methods

- Graph-R1 and KBQA-R1 use one mutable trajectory. HyPER-R1 exposes several
  executable alternatives to the policy at once.
- BoG reverts a linear trajectory after a dead end. HyPER-R1 preserves
  alternatives before failure and can compose them.
- CPR maintains a calibrated retrieval active set. Its LLM does not manipulate
  a persistent executable reasoning graph and it is not trained end to end as
  the graph controller.
- Ordinary beam search expands and prunes with an external scalar score.
  HyPER-R1's policy chooses which hypothesis to inspect, expand, combine, or
  commit using the full question and graph feedback.

## Primary Experiment

Train the same backbone and data under an identical execution budget:

1. released KBQA-R1;
2. KBQA-R1 plus ordinary relation beam search;
3. BoG-style retrospective recovery;
4. HyPER-R1.

The primary benchmark is full GrailQA, reporting overall F1 and
i.i.d./compositional/zero-shot slices. WebQSP is the simpler control. Report
answer F1, exact match, execution calls, tokens, latency, and recovery from an
initial wrong top-1 relation.

The method is falsified if it does not beat both the linear agent and the
matched-budget beam baseline on full GrailQA, or if its gain disappears when
execution calls are matched.
