from dataclasses import replace

from kbqa_r1.hyper_data import (
    DemonstrationBuilder,
    DemonstrationValidator,
    IneligibleProgram,
    ProgramExecutionError,
    RelationOption,
    compile_gold_plan,
    decision_sft_records,
    step_sft_records,
    trajectory_sft_record,
)


def fake_executor(functions, target):
    text = "\n".join(functions)
    if "AND(" in text:
        return ["m.shared"]
    if "r.gold2" in text:
        return ["m.answer"] if "r.gold1" in text else []
    if "r.alt2" in text:
        return ["m.alt_terminal"]
    if target == "expression1":
        if "r.alt1" in text:
            return ["m.plausible"]
        if "r.other1" in text:
            return ["m.other"]
        if "r.left" in text:
            return ["m.a", "m.shared"]
        if "r.right" in text:
            return ["m.b", "m.shared"]
        if "r.unrelated" in text:
            return ["m.unrelated"]
        return ["m.gold_prefix"]
    if target == "expression2":
        return ["m.b", "m.shared"]
    if target == "expression3":
        return ["m.shared"]
    return []


def candidates(query, state, join):
    if "r.alt1" in "\n".join(state):
        return [
            RelationOption("r.gold2", 0.90, 1),
            RelationOption("r.alt2", 0.82, 2),
        ]
    if join.relation == "r.left":
        return [
            RelationOption("r.left", 0.93, 1),
            RelationOption("r.unrelated", 0.88, 2),
            RelationOption("r.right", 0.84, 3),
        ]
    if join.relation == "r.gold1":
        return [
            RelationOption("r.alt1", 0.92, 1),
            RelationOption("r.gold1", 0.81, 2),
            RelationOption("r.other1", 0.70, 3),
        ]
    return [
        RelationOption("r.alt2", 0.90, 1),
        RelationOption("r.gold2", 0.82, 2),
    ]


def one_hop_candidates(query, state, join):
    return [
        RelationOption(join.relation, 0.92, 1),
        RelationOption("r.alt1", 0.81, 2),
    ]


def test_compile_rejects_unsupported_operator():
    try:
        compile_gold_plan(["expression1 = COUNT(expression0)"])
    except IneligibleProgram as exc:
        assert "unsupported operator" in str(exc)
    else:
        raise AssertionError("COUNT should not silently enter the first corpus")


def test_compile_accepts_unnumbered_grailqa_expression_variable():
    plan = compile_gold_plan(
        [
            "expression = START('m.0hqs1x_')",
            "expression = JOIN('medicine.routed_drug.marketed_formulations', expression)",
            "expression1 = START('medicine.routed_drug')",
            "expression = AND(expression1, expression)",
            "expression = STOP(expression)",
        ]
    )

    assert plan.target_expression == "expression"
    assert [statement.kind for statement in plan.statements] == [
        "start",
        "join",
        "start",
        "and",
        "stop",
    ]


def test_compile_requires_stop_to_be_terminal():
    functions = [
        "expression1 = START('m.topic')",
        "expression2 = STOP(expression1)",
        "expression1 = JOIN('r.gold1', expression1)",
    ]
    try:
        compile_gold_plan(functions)
    except IneligibleProgram as exc:
        assert "terminal" in str(exc)
    else:
        raise AssertionError("statements after STOP must be rejected")


def test_builds_replayable_delayed_frontier_recovery():
    row = {
        "ID": "q1",
        "question": "Which answer follows both intended relations?",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.gold1', expression1)",
            "expression1 = JOIN('r.gold2', expression1)",
        ],
        "answer": ["m.answer"],
    }
    builder = DemonstrationBuilder(fake_executor, candidates)
    demos = builder.build(row)
    demo = demos[0]
    assert demo.family == "delayed_frontier_recovery"
    actions = [step.action for step in demo.steps]
    assert actions[:3] == ["Find_relation", "Select", "Find_relation"]
    assert "Prune" in actions
    assert actions[-3:] == ["Select", "Find_relation", "Commit"]
    assert demo.private_metadata["proposal_relations"] == [
        "r.alt1", "r.gold1", "r.other1"
    ]
    assert all(
        len(step.arguments) == 1
        for step in demo.steps
        if step.action == "Find_relation"
    )
    assert DemonstrationValidator(fake_executor, max_active=6).validate(demo) == []

    public = step_sft_records(demo)
    assert all("private_metadata" not in record for record in public)
    assert all("role" not in str(record) for record in public)
    assert all("gold_relation" not in str(record) for record in public)

    sft = trajectory_sft_record(demo)
    assistant = [
        message["content"] for message in sft["messages"]
        if message["role"] == "assistant"
    ]
    assert any("Find_relation" in content for content in assistant)
    assert any("Select" in content for content in assistant)
    assert any("Prune" in content for content in assistant)
    assert any("Commit" in content for content in assistant)
    assert sft["extra_info"]["gold_injected_into_proposals"] is False
    graph_observations = [
        message["content"]
        for message in sft["messages"]
        if message["role"] == "user" and "<hypothesis_graph>" in message["content"]
    ]
    assert "nodes=3 executions=3" in graph_observations[0]
    assert "H3" not in graph_observations[0]

    assert len(demos) == 1
    assert builder.stats["recovery_probe_natural_frontier"] == 1
    assert builder.stats["recovery_probe_visible_empty"] == 1
    assert builder.stats["continuation_proposal_hit"] == 1


def test_recovery_does_not_emit_a_contradictory_direct_twin():
    row = {
        "ID": "no-twin",
        "question": "Which answer follows both intended relations?",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.gold1', expression1)",
            "expression1 = JOIN('r.gold2', expression1)",
        ],
        "answer": ["m.answer"],
    }

    demos = DemonstrationBuilder(fake_executor, candidates).build(row)

    assert [demo.family for demo in demos] == ["delayed_frontier_recovery"]


def test_recovery_probe_uses_natural_frontier_without_hidden_gold_injection():
    row = {
        "ID": "natural-recovery",
        "question": "Which answer follows both intended relations?",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.gold1', expression1)",
            "expression1 = JOIN('r.gold2', expression1)",
        ],
        "answer": ["m.answer"],
    }

    def executor(functions, target):
        if "r.dead" in "\n".join(functions):
            return []
        return fake_executor(functions, target)

    def provider(query, state, join):
        if "r.alt1" in "\n".join(state):
            return [
                RelationOption("r.dead", 0.91, 1),
                RelationOption("r.alt2", 0.84, 2),
            ]
        return candidates(query, state, join)

    builder = DemonstrationBuilder(executor, provider)
    recovery = builder.build(row)[0]

    assert recovery.family == "delayed_frontier_recovery"
    wrong_children = [
        node for node in recovery.hypotheses.values()
        if node.parent_id == "H0"
    ]
    assert {node.relation for node in wrong_children} == {"r.alt2", "r.dead"}
    assert all(node.relation != "r.gold2" for node in wrong_children)
    assert any(step.action == "Prune" for step in recovery.steps)
    assert DemonstrationValidator(executor, max_active=6).validate(recovery) == []


def test_one_hop_frontier_commits_without_ceremonial_select():
    row = {
        "ID": "direct",
        "question": "Which answer follows the intended relation?",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.gold1', expression1)",
        ],
        "answer": ["m.gold_prefix"],
    }
    demo = DemonstrationBuilder(fake_executor, one_hop_candidates).build(row)[0]
    assert demo.family == "frontier_commit"
    assert [step.action for step in demo.steps] == ["Find_relation", "Commit"]
    assert DemonstrationValidator(fake_executor, max_active=6).validate(demo) == []
    decisions = decision_sft_records(demo)
    assert len(decisions) == 2
    assert all("<action>" in row["messages"][-1]["content"] for row in decisions)
    assert all("<answer>" not in row["messages"][-1]["content"] for row in decisions)
    assert decisions[-1]["messages"][-1]["loss_mask"] == 1
    assert all(
        message.get("loss_mask") == 0
        for message in decisions[-1]["messages"][:-1]
        if message["role"] == "assistant"
    )


def test_recovery_is_not_manufactured_when_gold_is_already_top1():
    row = {
        "ID": "gold-first",
        "question": "Which answer follows both intended relations?",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.gold1', expression1)",
            "expression1 = JOIN('r.gold2', expression1)",
        ],
        "answer": ["m.answer"],
    }

    def gold_first(query, state, join):
        if join.relation == "r.gold1":
            return [
                RelationOption("r.gold1", 0.95, 1),
                RelationOption("r.alt1", 0.90, 2),
            ]
        return [
            RelationOption("r.gold2", 0.95, 1),
            RelationOption("r.alt2", 0.90, 2),
        ]

    demos = DemonstrationBuilder(fake_executor, gold_first).build(row)
    assert [demo.family for demo in demos] == ["direct_frontier_progress"]
    assert [step.action for step in demos[0].steps] == [
        "Find_relation", "Select", "Find_relation", "Commit"
    ]
    assert all(step.action != "Prune" for step in demos[0].steps)
    assert DemonstrationValidator(fake_executor, max_active=6).validate(demos[0]) == []


def test_two_hop_progress_survives_routine_answer_type_tail():
    row = {
        "ID": "typed-two-hop",
        "question": "Which answer follows both intended relations?",
        "function_list": [
            "expression = START('m.topic')",
            "expression = JOIN('r.gold1', expression)",
            "expression = JOIN('r.gold2', expression)",
            "expression1 = START('answer.type')",
            "expression = AND(expression1, expression)",
            "expression = STOP(expression)",
        ],
        "answer": ["m.answer"],
    }

    def typed_executor(functions, target):
        text = "\n".join(functions)
        if "r.gold2" in text:
            return ["m.answer"] if "r.gold1" in text else []
        return fake_executor(functions, target)

    demos = DemonstrationBuilder(typed_executor, candidates).build(row)

    assert [demo.family for demo in demos] == ["direct_frontier_progress"]


def test_recovery_uses_visible_path_mismatch_for_nonempty_wrong_terminal_answer():
    row = {
        "ID": "wrong-answer",
        "question": "Which answer follows both intended relations?",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.gold1', expression1)",
            "expression1 = JOIN('r.gold2', expression1)",
        ],
        "answer": ["m.answer"],
    }

    def nonempty_wrong_executor(functions, target):
        text = "\n".join(functions)
        if "r.gold2" in text:
            return ["m.answer"] if "r.gold1" in text else ["m.wrong"]
        return fake_executor(functions, target)

    demos = DemonstrationBuilder(nonempty_wrong_executor, candidates).build(row)
    assert [demo.family for demo in demos] == ["semantic_frontier_recovery"]
    semantic_prunes = [
        step for step in demos[0].steps
        if step.action == "Prune"
        and any(
            fact.startswith("question_path_mismatch:")
            for fact in step.rationale_facts
        )
    ]
    assert semantic_prunes
    rendered = str(trajectory_sft_record(demos[0]))
    assert "visible relation path conflicts" in rendered
    assert DemonstrationValidator(
        nonempty_wrong_executor, max_active=6
    ).validate(demos[0]) == []


def test_widen_recovers_a_naturally_ranked_relation_outside_initial_frontier():
    row = {
        "ID": "widen",
        "question": "Which answer follows the intended relation?",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.gold1', expression1)",
        ],
        "answer": ["m.gold_prefix"],
    }

    def wider_candidates(query, state, join):
        return [
            RelationOption("r.alt1", 0.95, 1),
            RelationOption("r.other1", 0.90, 2),
            RelationOption("r.unrelated", 0.85, 3),
            RelationOption("r.gold1", 0.80, 4),
            RelationOption("r.extra1", 0.75, 5),
            RelationOption("r.extra2", 0.70, 6),
        ]

    builder = DemonstrationBuilder(fake_executor, wider_candidates)
    demo = builder.build(row)[0]

    assert demo.family == "adaptive_frontier_widen"
    assert [step.action for step in demo.steps] == [
        "Find_relation", "Widen", "Commit"
    ]
    assert demo.private_metadata["gold_rank"] == 4
    assert demo.private_metadata["proposal_recall_at_frontier"] is False
    assert demo.private_metadata["proposal_recall_at_max_frontier"] is True
    assert demo.private_metadata["candidate_future_values"]["r.gold1"]["answer_exact"]
    assert DemonstrationValidator(fake_executor, max_active=6).validate(demo) == []
    rendered = trajectory_sft_record(demo)
    assert any(
        "Widen [ m.topic ]" in message["content"]
        for message in rendered["messages"]
        if message["role"] == "assistant"
    )


def test_widen_also_applies_to_a_selected_hypothesis_continuation():
    row = {
        "ID": "continuation-widen",
        "question": "Which answer follows both intended relations?",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.gold1', expression1)",
            "expression1 = JOIN('r.gold2', expression1)",
        ],
        "answer": ["m.answer"],
    }

    def continuation_candidates(query, state, join):
        if join.relation == "r.gold1":
            return [
                RelationOption("r.gold1", 0.95, 1),
                RelationOption("r.alt1", 0.90, 2),
                RelationOption("r.other1", 0.85, 3),
            ]
        return [
            RelationOption("r.alt2", 0.95, 1),
            RelationOption("r.other2", 0.90, 2),
            RelationOption("r.third2", 0.85, 3),
            RelationOption("r.gold2", 0.80, 4),
            RelationOption("r.extra2", 0.75, 5),
            RelationOption("r.last2", 0.70, 6),
        ]

    demo = DemonstrationBuilder(fake_executor, continuation_candidates).build(row)[0]

    assert demo.family == "adaptive_frontier_widen"
    assert [step.action for step in demo.steps].count("Widen") == 1
    widen = next(step for step in demo.steps if step.action == "Widen")
    assert widen.arguments == ("expression1",)
    assert len(widen.created) == 3
    assert DemonstrationValidator(fake_executor, max_active=6).validate(demo) == []


def test_builder_rejects_trajectories_that_cannot_finish_within_rollout_budget():
    row = {
        "ID": "too-short",
        "question": "Which answer follows both intended relations?",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.gold1', expression1)",
            "expression1 = JOIN('r.gold2', expression1)",
        ],
        "answer": ["m.answer"],
    }
    builder = DemonstrationBuilder(fake_executor, candidates, max_turns=4)

    assert builder.build(row) == []
    assert builder.stats["trajectory_turn_budget_miss"] == 1


def test_builder_requires_room_for_two_complete_relation_frontiers():
    try:
        DemonstrationBuilder(
            fake_executor,
            candidates,
            frontier_width=3,
            max_active=5,
        )
    except ValueError as exc:
        assert "two complete relation frontiers" in str(exc)
    else:
        raise AssertionError("multi-root curriculum must fit both full frontiers")


def test_gold_is_never_injected_when_proposal_frontier_misses_it():
    row = {
        "ID": "miss",
        "question": "Question",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.gold1', expression1)",
        ],
        "answer": ["m.gold_prefix"],
    }

    def misses_gold(*args):
        return [RelationOption("r.alt1", 0.9, 1), RelationOption("r.other1", 0.8, 2)]

    assert DemonstrationBuilder(fake_executor, misses_gold).build(row) == []


def test_candidate_queries_use_only_question_visible_intent():
    row = {
        "ID": "no-hint",
        "question": "Who is connected to the topic?",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.secret_gold_relation', expression1)",
        ],
        "answer": ["m.gold_prefix"],
    }
    seen = []

    def provider(query, state, join):
        seen.append(query)
        return [
            RelationOption("r.secret_gold_relation", 0.9, 1),
            RelationOption("r.alt1", 0.8, 2),
        ]

    demo = DemonstrationBuilder(fake_executor, provider).build(row)[0]
    assert seen == [row["question"]]
    assert demo.steps[0].arguments == ("m.topic",)
    assert "secret gold relation" not in str(trajectory_sft_record(demo))

    leaked = replace(
        demo,
        steps=[
            replace(demo.steps[0], arguments=("m.topic", "secret gold relation")),
            *demo.steps[1:],
        ],
    )
    assert any(
        "environment-owned" in error
        for error in DemonstrationValidator(fake_executor).validate(leaked)
    )


def test_builder_resolves_entity_labels_and_uses_stable_nonpositional_order():
    row = {
        "ID": "readable-roots",
        "question": "Which answer follows the intended relation?",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.gold1', expression1)",
        ],
        "answer": ["m.gold_prefix"],
        "extra_info": {
            "candidate_entities": [
                ["m.topic", "m.topic"],
                ["m.other", "m.other"],
            ]
        },
        "prompt": (
            "Candidate Entities: ['m.topic' (m.topic), 'm.other' (m.other)]\n"
            "Question: Which answer follows the intended relation?"
        ),
    }
    labels = {
        "m.topic": "Topic Entity",
        "m.other": "Other Entity",
        "m.gold_prefix": "Answer Entity",
    }
    builder = DemonstrationBuilder(
        fake_executor,
        one_hop_candidates,
        entity_display_provider=lambda values: {
            value: labels[value] for value in values if value in labels
        },
    )

    first = builder.build(row)[0]
    second = builder.build(row)[0]
    assert first.private_metadata["candidate_entities"] == second.private_metadata[
        "candidate_entities"
    ]
    assert set(first.private_metadata["candidate_entities"]) == {
        ("Topic Entity", "m.topic"),
        ("Other Entity", "m.other"),
    }
    rendered = str(trajectory_sft_record(first))
    assert "Topic Entity" in rendered
    assert "Answer Entity [m.gold_prefix]" in rendered
    assert "'m.topic' (m.topic)" not in rendered


def test_failed_candidate_is_not_misrepresented_as_empty_evidence():
    row = {
        "ID": "execution-failure",
        "question": "Which answer follows the intended relation?",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.gold1', expression1)",
        ],
        "answer": ["m.gold_prefix"],
    }

    def executor(functions, target):
        if "r.failed" in "\n".join(functions):
            raise ProgramExecutionError("invalid candidate")
        return fake_executor(functions, target)

    def options(*args):
        return [
            RelationOption("r.gold1", 0.9, 1),
            RelationOption("r.failed", 0.8, 2),
            RelationOption("r.alt1", 0.7, 3),
        ]

    demo = DemonstrationBuilder(executor, options).build(row)[0]
    assert {node.relation for node in demo.hypotheses.values()} == {
        "r.gold1",
        "r.alt1",
    }


def test_rejects_rows_without_annotated_answers():
    row = {
        "ID": "missing-answer",
        "question": "Question",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.gold1', expression1)",
        ],
    }
    assert DemonstrationBuilder(fake_executor, candidates).build(row) == []


def test_large_answer_rows_do_not_turn_sft_into_answer_copying():
    answers = [f"m.answer_{index}" for index in range(101)]
    row = {
        "ID": "large-answer",
        "question": "List every answer.",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.gold1', expression1)",
        ],
        "answer": answers,
    }
    builder = DemonstrationBuilder(
        lambda functions, target: answers,
        one_hop_candidates,
    )

    assert builder.build(row) == []
    assert builder.stats["large_answer_row_skipped"] == 1


def test_conjunction_requires_both_branches():
    row = {
        "ID": "q3",
        "question": "Which entity satisfies both conditions?",
        "function_list": [
            "expression0 = START('m.topic')",
            "expression1 = JOIN('r.left', expression0)",
            "expression2 = JOIN('r.right', expression0)",
            "expression3 = AND(expression1, expression2)",
        ],
        "answer": ["m.shared"],
    }
    demos = DemonstrationBuilder(fake_executor, candidates).build(row)
    conjunction = next(demo for demo in demos if demo.family == "conjunction")
    assert [step.action for step in conjunction.steps] == [
        "Find_relation", "Combine", "Commit"
    ]
    assert DemonstrationValidator(fake_executor, max_active=6).validate(conjunction) == []

    bad_steps = list(conjunction.steps)
    bad_steps[1] = replace(bad_steps[1], arguments=("H0", "H1"))
    malformed = replace(conjunction, steps=bad_steps)
    assert any(
        "do not match required branches" in error
        for error in DemonstrationValidator(fake_executor, max_active=6).validate(malformed)
    )


def test_failed_conjunction_is_not_relabelled_as_complete_linear_branch():
    row = {
        "ID": "missing-conjunction-frontier",
        "question": "Which entity satisfies both conditions?",
        "function_list": [
            "expression0 = START('m.topic')",
            "expression1 = JOIN('r.left', expression0)",
            "expression2 = JOIN('r.right', expression0)",
            "expression3 = AND(expression1, expression2)",
        ],
        "answer": ["m.shared"],
    }

    def misses_right_relation(query, state, join):
        return [
            RelationOption("r.left", 0.93, 1),
            RelationOption("r.unrelated", 0.88, 2),
        ]

    demos = DemonstrationBuilder(fake_executor, misses_right_relation).build(row)

    assert demos == []


def test_failed_multihop_conjunction_does_not_export_either_partial_branch():
    row = {
        "ID": "missing-multihop-conjunction",
        "question": "Which answer follows both the direct and described conditions?",
        "function_list": [
            "expression = START('m.direct')",
            "expression = JOIN('r.right', expression)",
            "expression1 = START('m.description')",
            "expression1 = JOIN('r.clue', expression1)",
            "expression1 = JOIN('r.left', expression1)",
            "expression = AND(expression1, expression)",
        ],
        "answer": ["m.shared"],
    }

    def executor(functions, target):
        text = "\n".join(functions)
        if "AND(" in text:
            return ["m.shared"]
        if "r.right" in text:
            return ["m.shared"]
        if "r.left" in text and "r.clue" in text:
            return ["m.shared"]
        if "r.clue" in text:
            return ["m.prefix"]
        return ["m.other"]

    def misses_full_conjunction(query, state, join):
        if join.relation == "r.clue":
            return [
                RelationOption("r.clue", 0.9, 1),
                RelationOption("r.other", 0.8, 2),
            ]
        return [
            RelationOption("r.right", 0.9, 1),
            RelationOption("r.other", 0.8, 2),
        ]

    demos = DemonstrationBuilder(executor, misses_full_conjunction).build(row)

    assert demos == []


def test_answer_type_and_is_not_used_as_a_conjunction_lesson():
    row = {
        "ID": "answer-type",
        "question": "Which routed drug contains the formulation?",
        "function_list": [
            "expression = START('m.formulation')",
            "expression = JOIN('r.gold1', expression)",
            "expression1 = START('medicine.routed_drug')",
            "expression = AND(expression1, expression)",
            "expression = STOP(expression)",
        ],
        "answer": ["m.gold_prefix"],
    }

    demos = DemonstrationBuilder(fake_executor, one_hop_candidates).build(row)

    assert all(demo.family != "conjunction" for demo in demos)


def test_builds_two_root_conjunction_followed_by_answer_type_constraint():
    row = {
        "ID": "two-root",
        "question": "What category includes both School A and School B?",
        "function_list": [
            "expression = START('m.school_a')",
            "expression = JOIN('r.left', expression)",
            "expression1 = START('m.school_b')",
            "expression1 = JOIN('r.right', expression1)",
            "expression = AND(expression, expression1)",
            "expression1 = START('education.school_category')",
            "expression = AND(expression1, expression)",
            "expression = STOP(expression)",
        ],
        "answer": ["m.shared"],
    }

    def two_root_candidates(query, state, join):
        if join.relation == "r.left":
            return [
                RelationOption("r.left", 0.9, 1),
                RelationOption("r.alt1", 0.8, 2),
            ]
        return [
            RelationOption("r.right", 0.9, 1),
            RelationOption("r.alt2", 0.8, 2),
        ]

    def two_root_executor(functions, target):
        text = "\n".join(functions)
        if "AND(" in text:
            return ["m.shared"]
        if "r.left" in text:
            return ["m.left", "m.shared"]
        if "r.right" in text:
            return ["m.right", "m.shared"]
        return []

    builder = DemonstrationBuilder(two_root_executor, two_root_candidates)
    demos = builder.build(row)
    conjunctions = [demo for demo in demos if demo.family == "conjunction"]

    assert len(conjunctions) == 1
    conjunction = conjunctions[0]
    assert conjunction.private_metadata["conjunction_roots"] == 2
    assert [step.action for step in conjunction.steps] == [
        "Find_relation",
        "Find_relation",
        "Combine",
        "Commit",
    ]
    assert conjunction.steps[0].arguments == ("m.school_a",)
    assert conjunction.steps[1].arguments == ("m.school_b",)
    assert DemonstrationValidator(two_root_executor, max_active=6).validate(conjunction) == []


def test_builder_and_validator_share_execution_cache():
    row = {
        "ID": "cached",
        "question": "Which answer follows the intended relation?",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.gold1', expression1)",
        ],
        "answer": ["m.gold_prefix"],
    }
    calls = []

    def recording_executor(functions, target):
        calls.append((tuple(functions), target))
        return fake_executor(functions, target)

    builder = DemonstrationBuilder(recording_executor, one_hop_candidates)
    demo = builder.build(row)[0]
    calls_after_build = len(calls)
    validator = DemonstrationValidator(
        recording_executor,
        execution_cache=builder._execution_cache,
    )

    assert validator.validate(demo) == []
    assert len(calls) == calls_after_build


def test_reverse_relation_rewrite_is_preserved():
    from kbqa_r1.hyper_data import replace_join_relation

    raw = "expression1 = JOIN('(R people.person.parents)', expression1)"
    assert replace_join_relation(raw, "(R people.person.children)") == (
        "expression1 = JOIN('(R people.person.children)', expression1)"
    )
