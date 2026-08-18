# Prompt for Independent Pro Review

You are independently reviewing whether the current HyPER-R1 corpus and method
are ready for expensive supervised fine-tuning. Do not assume that a passed or
failed heuristic gate is correct. Trace the implementation and judge whether
the data teaches the intended inference-time policy.

Repository: `https://github.com/begcdn/KBQA-R1`

Branch: `codex/hyper-r1`

Review the current branch head. The compact analysis bundle was introduced at
commit `c1c6cf4`.

Start with:

- `artifacts/hyper_r1_v8_method_preserving/README.md`
- `artifacts/hyper_r1_v8_method_preserving/report.json`
- `artifacts/hyper_r1_v8_method_preserving/stratified_samples.jsonl`
- `kbqa_r1/hyper_data.py`
- `kbqa_r1/hyper_r1.py`
- `kbqa_r1/hyper_prompt.py`
- `kbqa_r1/relation_paging.py`
- `scripts/data_process/build_hyper_demonstrations.py`
- `scripts/train/train_hyper_r1_sft.sh`
- relevant tests under `tests_fork_r1/`

## Intended method

HyPER-R1 should learn executable knowledge-graph exploration without committing
to one interpretation too early. It keeps several hypotheses active, exposes
ranked relation pages, selects one branch for continuation without deleting the
others, prunes only after visible execution contradiction, can return to a
lower-ranked branch, combines branches for conjunction, handles logical
operators, and commits only when a hypothesis covers the complete question.

The demonstrations are derived from GrailQA gold programs, but relation
proposals come from the same question-and-visible-state ranker intended for
inference. Gold is used to identify the verified teacher action and evaluate
proposal recall; it must not be injected into the student-visible proposal set.
Every saved trajectory must replay under the runtime contract.

## Result requiring a decision

The full run processed 43,851 questions and produced 28,312 training
demonstrations. All replay, consistency, public-source, duplication, conjunction,
operator, entity-label, and deep-progress checks pass. The sole failed readiness
check is recovery quantity:

- semantic recovery: 1,240
- delayed recovery: 415
- total verified recovery: 1,655
- multi-hop training trajectories: 26,354
- configured minimum: `max(500, 10% of multi-hop)` = 2,636

The 10% requirement is a hand-set curriculum heuristic. It is not evidence by
itself that 1,655 examples are insufficient.

The observed builder funnel is:

- detected recovery opportunities: 1,957
- verified recovery demonstrations built: 1,737 (88.8% of detected)
- recovery demonstrations retained in training: 1,655 (95.3% of built)

This shows that ordinary verification, deduplication, and splitting do not
explain most of the shortfall. It does **not** show that the opportunity detector
finds every place where recovery ought to be taught.

## Your tasks

1. **Give a critical verdict:** choose exactly one of `train now`, `rebalance
   existing verified data`, `generate targeted additional data`, or `redesign
   before training`. State the strongest reason and what evidence would reverse
   your verdict.
2. **Audit inference compatibility:** verify that every demonstrated action is
   available from the public inference state, including entity/literal roots,
   relation paging, Select/Widen/Prune/Combine/Merge/operator/Commit semantics,
   and fixed budgets. Identify any teacher-only information visible to the
   student or any behavior taught under conditions absent at inference.
3. **Audit whether the corpus teaches the actual novelty:** determine whether
   examples genuinely teach preservation and recovery among plausible nonempty
   alternatives, or mostly teach gold-guided next-action imitation. Inspect the
   recovery samples and relevant builder logic, not only aggregate counts.
4. **Explain why recovery is low before prescribing a fix:** reconstruct a
   measured recovery funnel from source questions to final training examples.
   Inspect (a) the prevalence and shapes of recoverable gold programs, (b) the
   exact definition of a recovery opportunity, (c) proposal ranking and page
   visibility, (d) execution and visible-failure requirements, (e) node/turn
   budgets, (f) replay and public-source validation, and (g) deduplication,
   balancing, and the train/validation split. Determine whether the low count is
   mainly because GrailQA rarely requires recovery, the detector undercounts
   plausible nonempty wrong branches, teacher construction makes recovery
   unnecessary, or proposal/budget filters suppress otherwise useful examples.
   Distinguish genuine no-recovery questions from missed teaching opportunities.
5. **Judge the failed recovery threshold:** assess whether 1,655 high-quality
   recovery trajectories are likely enough for SFT, whether the 10% rule is
   justified, and whether oversampling/reweighting existing verified recovery
   examples is safer than synthesizing more. Do not recommend oversampling or
   synthesis until task 4 identifies the cause, and do not recommend changing
   the threshold merely to make the report green.
6. **Inspect distribution risks:** direct progress dominates; three-hop
   acceptance is 37.36% and four-hop acceptance is 5%. Determine whether the
   current curriculum would make a model commit early, ignore Widen/Prune, or
   fail on longer exploration despite excellent replay validity.
7. **Assess proposal quality and training implications:** relation top-1 is
   49.88%, first-page recall 91.62%, and within-budget recall 95.25%. Explain
   whether these numbers create useful policy-learning situations or impose a
   hard ceiling that must be fixed before SFT.
8. **Propose the smallest sound next step:** if data transformation is enough,
   specify exact family weights or sampling rules and preserve the
   question-disjoint split. If new data is needed, specify how it can be
   generated and verified without giving the student gold relations. If redesign
   is needed, identify the broken mechanism precisely.
9. **Pre-register SFT success criteria:** specify metrics that would show the
   policy learned HyPER-R1 rather than memorized valid syntax. Include action
   validity, relation/frontier decisions, recovery from a plausible wrong
   branch, Widen use, premature Commit, conjunction completion, end-answer
   quality, and at least one baseline/ablation.

## Required response format

Use these headings:

1. Verdict
2. What the corpus genuinely guarantees
3. Critical risks or contract violations
4. Why recovery is low
5. Is 1,655 recovery trajectories enough?
6. Exact next data action
7. SFT and evaluation plan
8. Stop conditions

Be decisive and implementation-aware. Distinguish proven corpus properties,
reasonable hypotheses, and unknowns. Do not propose a broad collection of small
experiments; recommend one coherent path toward the strongest final KGQA method.
