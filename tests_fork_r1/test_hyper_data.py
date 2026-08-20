from dataclasses import replace

from kbqa_r1.hyper_data import (
    DemonstrationBuilder as _DemonstrationBuilder,
    DemonstrationValidator,
    IneligibleProgram,
    ProgramExecutionError,
    RelationOption,
    certify_program_commit,
    compile_gold_plan,
    decision_sft_records,
    programs_are_intent_equivalent,
    step_sft_records,
    trajectory_sft_record,
)


def DemonstrationBuilder(*args, **kwargs):
    """Use the production oracle-topic-entity protocol in fixtures."""
    return _DemonstrationBuilder(*args, **kwargs)


def semantic_actions(demo):
    """Hide storage/query mechanics when asserting the logical trajectory."""
    return [
        step.action
        for step in demo.steps
        if step.action not in {"Inspect", "Park", "Recall"}
    ]


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


def test_compile_accepts_runtime_logical_operators():
    plan = compile_gold_plan(
        [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.values', expression1)",
            "expression1 = ARG('ARGMAX', expression1, 'r.number')",
            "expression1 = TC(expression1, 'r.from', 'NOW')",
            "expression1 = COUNT(expression1)",
        ]
    )

    assert [statement.kind for statement in plan.statements] == [
        "start",
        "join",
        "order",
        "time_constraint",
        "count",
    ]
    assert [statement.arguments for statement in plan.operators] == [
        ("ARGMAX", "expression1", "r.number"),
        ("r.from", "NOW"),
        ("expression1",),
    ]


def test_compile_accepts_comparison_branch():
    plan = compile_gold_plan(
        [
            "expression1 = START('42^^http://www.w3.org/2001/XMLSchema#integer')",
            "expression1 = CMP('ge', 'r.number', expression1)",
        ]
    )

    assert plan.operators[0].kind == "compare"
    assert plan.operators[0].arguments == ("ge", "r.number", "expression1")


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

    assert plan.target_expression == "expression1"
    assert plan.executable_functions == (
        "expression1 = START('m.0hqs1x_')",
        "expression1 = JOIN('medicine.routed_drug.marketed_formulations', expression1)",
        "expression2 = START('medicine.routed_drug')",
        "expression1 = AND(expression2, expression1)",
    )
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


def test_operator_program_teaches_count_after_retained_relation_frontier():
    row = {
        "ID": "count-program",
        "question": "How many items are connected to the topic?",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.items', expression1)",
            "expression1 = COUNT(expression1)",
        ],
        "answer": ["2"],
    }

    def executor(functions, target):
        text = "\n".join(functions)
        if "COUNT(" in text:
            return ["2"]
        if "r.items" in text:
            return ["m.one", "m.two"]
        if "r.alternative" in text:
            return ["m.other"]
        return []

    def options(*_args):
        return [
            RelationOption("r.alternative", 0.9, 1),
            RelationOption("r.items", 0.8, 2),
        ]

    demo = DemonstrationBuilder(executor, options).build(row)[0]

    assert demo.family == "operator_program"
    assert semantic_actions(demo) == [
        "Find_relation",
        "Select",
        "Find_relation",
        "Select",
        "Count",
        "Commit",
    ]
    assert demo.private_metadata["recovery_stratum"] == "operator_adjacent"
    assert demo.private_metadata["probe_outcome"] == "unresolved_nonempty"
    assert DemonstrationValidator(executor).validate(demo) == []
    rendered = trajectory_sft_record(demo)
    assistant = [
        message["content"]
        for message in rendered["messages"]
        if message["role"] == "assistant"
    ]
    assert any("Count [ expression1 ]" in message for message in assistant)


def test_operator_recovery_uses_relation_siblings_below_type_constraint():
    row = {
        "ID": "typed-count-recovery",
        "question": "How many subjects are connected to the topic?",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.gold1', expression1)",
            "expression1 = JOIN('r.gold2', expression1)",
            "expression2 = START('domain.subject')",
            "expression1 = AND(expression2, expression1)",
            "expression1 = COUNT(expression1)",
        ],
        "answer": ["1"],
    }

    def executor(functions, target):
        del target
        text = "\n".join(functions)
        if "COUNT(" in text:
            return ["1"]
        if "AND(" in text:
            return ["m.answer"]
        if text.count("JOIN(") >= 3:
            return ["m.probe"]
        if "r.gold2" in text:
            return ["m.answer"]
        if "r.alt2" in text:
            return ["m.alternative"]
        if "r.gold1" in text:
            return ["m.prefix"]
        if "r.alt1" in text:
            return ["m.other_prefix"]
        return []

    demo = DemonstrationBuilder(executor, candidates).build(row)[0]

    assert demo.family == "operator_program"
    assert demo.private_metadata["recovery_stratum"] == "operator_adjacent"
    assert demo.private_metadata["probe_outcome"] == "unresolved_nonempty"
    assert "Merge" in semantic_actions(demo)
    assert "Count" in semantic_actions(demo)
    assert DemonstrationValidator(executor).validate(demo) == []


def test_count_teacher_rejects_private_operator_not_licensed_by_public_question():
    row = {
        "ID": "hidden-count-program",
        "question": "Which items are connected to the topic?",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.items', expression1)",
            "expression1 = COUNT(expression1)",
        ],
        "answer": ["2"],
    }

    def executor(functions, target):
        del target
        return (
            ["2"]
            if any("COUNT(" in item for item in functions)
            else ["m.one", "m.two"]
        )

    builder = DemonstrationBuilder(executor, candidates)

    assert builder.build(row) == []
    assert builder.stats["public_count_contract_false_negative"] == 1


def test_operator_program_opens_comparison_branch_and_combines_it():
    literal = "42^^http://www.w3.org/2001/XMLSchema#integer"
    row = {
        "ID": "comparison-program",
        "question": "Which topic items have a score of at least 42?",
        "function_list": [
            f"expression1 = START('{literal}')",
            "expression1 = CMP('ge', 'r.score', expression1)",
            "expression2 = START('m.topic')",
            "expression2 = JOIN('r.items', expression2)",
            "expression3 = AND(expression1, expression2)",
        ],
        "answer": ["m.shared"],
    }

    def executor(functions, target):
        text = "\n".join(functions)
        if "AND(" in text:
            return ["m.shared"]
        if "CMP(" in text:
            return ["m.high", "m.shared"]
        if "r.items" in text:
            return ["m.shared", "m.topic_item"]
        if "r.alternative" in text:
            return ["m.other"]
        return []

    def options(*_args):
        return [
            RelationOption("r.items", 0.9, 1),
            RelationOption("r.alternative", 0.8, 2),
        ]

    demo = DemonstrationBuilder(executor, options).build(row)[0]

    assert semantic_actions(demo) == [
        "Compare",
        "Find_relation",
        "Combine",
        "Commit",
    ]
    assert demo.steps[0].arguments == ("ge", "r.score", literal)
    assert DemonstrationValidator(executor).validate(demo) == []
    assert trajectory_sft_record(demo)["extra_info"]["replay_verified"] is True


def test_operator_program_executes_explicit_type_constraint_before_count():
    row = {
        "ID": "count-with-redundant-type",
        "question": "How many exhibition subjects are connected to the curator?",
        "function_list": [
            "expression = START('m.curator')",
            "expression = JOIN('r.subjects', expression)",
            "expression1 = START('exhibitions.exhibition_subject')",
            "expression = AND(expression1, expression)",
            "expression = COUNT(expression)",
        ],
        "answer": ["2"],
    }

    def executor(functions, target):
        text = "\n".join(functions)
        if "COUNT(" in text:
            return ["2"]
        if "AND(" in text or "r.subjects" in text:
            return ["m.one", "m.two"]
        if "r.alternative" in text:
            return ["m.other"]
        return []

    def options(*_args):
        return [
            RelationOption("r.alternative", 0.9, 1),
            RelationOption("r.subjects", 0.8, 2),
        ]

    builder = DemonstrationBuilder(executor, options)
    demo = builder.build(row)[0]

    assert semantic_actions(demo) == [
        "Find_relation",
        "Select",
        "Merge",
        "Select",
        "Find_relation",
        "Select",
        "Count",
        "Commit",
    ]
    assert demo.private_metadata["recovery_stratum"] == "operator_adjacent"
    merge = next(step for step in demo.steps if step.action == "Merge")
    assert merge.arguments == ("expression1", "exhibitions.exhibition_subject")
    assert builder.stats["operator_program_type_constraint"] == 1
    assert DemonstrationValidator(executor).validate(demo) == []


def test_operator_program_executes_explicit_type_constraint_after_compare():
    literal = "7.0^^http://www.w3.org/2001/XMLSchema#float"
    row = {
        "ID": "compare-with-redundant-type",
        "question": "Which wind forces have waves at least 7.0 high?",
        "function_list": [
            f"expression = START('{literal}')",
            "expression = CMP('ge', 'r.wave_height', expression)",
            "expression1 = START('meteorology.beaufort_wind_force')",
            "expression = AND(expression1, expression)",
        ],
        "answer": ["m.force10", "m.force11"],
    }

    def executor(functions, target):
        text = "\n".join(functions)
        if "CMP(" in text or "AND(" in text:
            return ["m.force10", "m.force11"]
        return []

    builder = DemonstrationBuilder(executor, lambda *_args: [])
    demo = builder.build(row)[0]

    assert semantic_actions(demo) == [
        "Compare",
        "Select",
        "Merge",
        "Commit",
    ]
    assert builder.stats["operator_program_type_constraint"] == 1
    assert DemonstrationValidator(executor).validate(demo) == []


def test_operator_program_keeps_nonredundant_bare_type_intersection():
    literal = "7.0^^http://www.w3.org/2001/XMLSchema#float"
    row = {
        "ID": "compare-with-required-type",
        "question": "Which wind force has waves at least 7.0 high?",
        "function_list": [
            f"expression = START('{literal}')",
            "expression = CMP('ge', 'r.wave_height', expression)",
            "expression1 = START('meteorology.beaufort_wind_force')",
            "expression = AND(expression1, expression)",
        ],
        "answer": ["m.force10"],
    }

    def executor(functions, target):
        text = "\n".join(functions)
        if "AND(" in text:
            return ["m.force10"]
        if "CMP(" in text:
            return ["m.force10", "m.other_type"]
        return []

    builder = DemonstrationBuilder(executor, lambda *_args: [])
    demo = builder.build(row)[0]

    assert semantic_actions(demo) == [
        "Compare",
        "Select",
        "Merge",
        "Commit",
    ]
    assert demo.hypotheses[demo.steps[2].created[0]].denotation == ("m.force10",)
    assert DemonstrationValidator(executor).validate(demo) == []


def test_operator_program_applies_order_and_time_to_the_selected_branch():
    row = {
        "ID": "ordered-time-program",
        "question": "Which connected item was latest among those active now?",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.items', expression1)",
            "expression1 = ARG('ARGMAX', expression1, 'r.date')",
            "expression1 = TC(expression1, 'r.from', 'NOW')",
        ],
        "answer": ["m.latest"],
    }

    def executor(functions, target):
        text = "\n".join(functions)
        if "TC(" in text:
            return ["m.latest"]
        if "ARG(" in text:
            return ["m.latest", "m.old"]
        if "r.items" in text:
            return ["m.latest", "m.old", "m.other"]
        if "r.alternative" in text:
            return ["m.unrelated"]
        return []

    def options(*_args):
        return [
            RelationOption("r.items", 0.9, 1),
            RelationOption("r.alternative", 0.8, 2),
        ]

    demo = DemonstrationBuilder(executor, options).build(row)[0]

    assert semantic_actions(demo) == [
        "Find_relation",
        "Select",
        "Order",
        "Select",
        "Time_constraint",
        "Commit",
    ]
    assert DemonstrationValidator(executor).validate(demo) == []


def test_operator_program_is_rejected_when_runtime_node_budget_cannot_fit_it():
    row = {
        "ID": "operator-node-budget",
        "question": "How many entities follow both relations?",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.first', expression1)",
            "expression1 = JOIN('r.second', expression1)",
            "expression1 = COUNT(expression1)",
        ],
        "answer": ["2"],
    }

    def executor(functions, target):
        text = "\n".join(functions)
        if "COUNT(" in text:
            return ["2"]
        return ["m.one", "m.two"]

    def six_options(query, state, join):
        return [
            RelationOption(
                join.relation if index == 0 else f"r.alt_{join.index}_{index}",
                1.0 - index / 10,
                index + 1,
            )
            for index in range(6)
        ]

    builder = DemonstrationBuilder(
        executor,
        six_options,
        max_active=12,
        max_nodes=12,
        frontier_width=6,
    )

    assert builder.build(row) == []
    assert builder.stats["operator_program_node_budget_miss"] == 1


def test_builds_replayable_certified_empty_recovery():
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
    assert demo.family == "certified_empty_recovery"
    actions = [step.action for step in demo.steps]
    assert semantic_actions(demo)[:3] == ["Find_relation", "Select", "Find_relation"]
    assert actions.count("Inspect") >= 3
    assert "Prune" in actions
    assert semantic_actions(demo)[-3:] == ["Select", "Find_relation", "Commit"]
    assert demo.private_metadata["proposal_relations"] == [
        "r.alt1", "r.gold1", "r.other1"
    ]
    assert all(
        len(step.arguments) == 1
        for step in demo.steps
        if step.action == "Find_relation"
    )
    assert DemonstrationValidator(fake_executor, max_active=24).validate(demo) == []
    forced_select = next(
        step for step in demo.steps if step.supervision == "intervention"
    )
    assert forced_select.action == "Select"
    prune = next(step for step in demo.steps if step.action == "Prune")
    assert prune.certificate_kind == "empty_monotone"

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
    assert "stored=0" in graph_observations[0]
    assert "execution_attempts=0/24" in graph_observations[0]
    assert "<proposal_catalog>" in graph_observations[0]
    assert "H3" not in graph_observations[0]

    assert len(demos) == 1
    assert builder.stats["recovery_probe_natural_frontier"] == 1
    assert builder.stats["recovery_probe_visible_empty"] == 1
    assert builder.stats["continuation_proposal_hit"] == 1


def test_builder_extracts_ids_from_official_grailqa_answer_records():
    row = {
        "ID": "q-official-answer",
        "question": "Which answer follows both intended relations?",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.gold1', expression1)",
            "expression1 = JOIN('r.gold2', expression1)",
        ],
        "answer": [
            {
                "answer_type": "Entity",
                "answer_argument": "m.answer",
                "entity_name": "The answer",
            }
        ],
    }

    demos = DemonstrationBuilder(fake_executor, candidates).build(row)

    assert len(demos) == 1
    assert demos[0].gold_answers == ("m.answer",)


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

    assert [demo.family for demo in demos] == ["certified_empty_recovery"]


def test_three_hop_program_teaches_repeated_public_frontier_progress():
    row = {
        "ID": "three-hop",
        "question": "Which answer follows all three intended relations?",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.gold1', expression1)",
            "expression1 = JOIN('r.gold2', expression1)",
            "expression1 = JOIN('r.gold3', expression1)",
        ],
        "answer": ["m.answer"],
    }

    def executor(functions, target):
        text = "\n".join(functions)
        if all(relation in text for relation in ("r.gold1", "r.gold2", "r.gold3")):
            return ["m.answer"]
        if "r.gold1" in text and "r.gold2" in text:
            return ["m.second"]
        if "r.gold1" in text:
            return ["m.first"]
        return ["m.alternative"]

    def provider(query, state, join):
        return [
            RelationOption(join.relation, 0.95, 1),
            RelationOption(f"r.alt_{join.index}_a", 0.85, 2),
            RelationOption(f"r.alt_{join.index}_b", 0.75, 3),
        ]

    demo = DemonstrationBuilder(
        executor, provider, max_turns=32
    ).build(row)[0]

    assert demo.family == "deep_frontier_progress"
    assert demo.private_metadata["path_hops"] == 3
    assert [step.action for step in demo.steps].count("Find_relation") == 3
    assert [step.action for step in demo.steps].count("Select") == 2
    assert demo.steps[-1].action == "Commit"
    assert max(node.depth for node in demo.hypotheses.values()) == 2
    assert DemonstrationValidator(executor, max_active=24).validate(demo) == []


def test_three_hop_program_applies_terminal_type_after_public_progress():
    row = {
        "ID": "typed-three-hop",
        "question": "Which typed answer follows all three intended relations?",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.gold1', expression1)",
            "expression1 = JOIN('r.gold2', expression1)",
            "expression1 = JOIN('r.gold3', expression1)",
            "expression2 = START('answer.type')",
            "expression1 = AND(expression1, expression2)",
        ],
        "answer": ["m.answer"],
    }

    def executor(functions, target):
        text = "\n".join(functions)
        if all(relation in text for relation in ("r.gold1", "r.gold2", "r.gold3")):
            return ["m.answer"] if "AND(" in text else ["m.answer", "m.other"]
        if "r.gold1" in text and "r.gold2" in text:
            return ["m.second"]
        if "r.gold1" in text:
            return ["m.first"]
        return ["m.alternative"]

    def provider(query, state, join):
        return [
            RelationOption(join.relation, 0.95, 1),
            RelationOption(f"r.alt_{join.index}_a", 0.85, 2),
            RelationOption(f"r.alt_{join.index}_b", 0.75, 3),
        ]

    demo = DemonstrationBuilder(executor, provider, max_turns=32).build(row)[0]

    assert demo.family == "deep_frontier_progress"
    assert [step.action for step in demo.steps].count("Find_relation") == 3
    assert semantic_actions(demo)[-3:] == [
        "Select",
        "Merge",
        "Commit",
    ]
    assert DemonstrationValidator(executor, max_active=24).validate(demo) == []

    over_budget = DemonstrationBuilder(executor, provider, max_turns=8)
    assert over_budget.build(row) == []
    assert over_budget.stats["trajectory_turn_budget_miss"] == 1


def test_deep_program_recovers_from_a_natural_nonempty_continuation():
    row = {
        "ID": "deep-recovery",
        "question": "Which answer follows all three intended relations?",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.gold1', expression1)",
            "expression1 = JOIN('r.gold2', expression1)",
            "expression1 = JOIN('r.gold3', expression1)",
        ],
        "answer": ["m.answer"],
    }

    def executor(functions, target):
        text = "\n".join(functions)
        if all(relation in text for relation in ("r.gold1", "r.gold2", "r.gold3")):
            return ["m.answer"]
        if "r.gold1" in text and "r.gold2" in text:
            return ["m.second"]
        if "r.gold1" in text:
            return ["m.plausible"]
        return ["m.other"]

    def provider(query, state, join):
        text = "\n".join(state)
        if join.relation == "r.gold2":
            return [
                RelationOption("r.detour", 0.95, 1),
                RelationOption("r.gold2", 0.90, 2),
            ]
        if "r.detour" in text:
            return [
                RelationOption("r.probe", 0.95, 1),
                RelationOption("r.probe_alt", 0.80, 2),
            ]
        return [
            RelationOption(join.relation, 0.95, 1),
            RelationOption(f"r.alt_{join.index}", 0.80, 2),
        ]

    demo = DemonstrationBuilder(executor, provider, max_turns=32).build(row)[0]

    assert demo.family == "deep_frontier_progress"
    assert demo.private_metadata["recovery_stratum"] == "deep"
    assert demo.private_metadata["probe_outcome"] == "unresolved_nonempty"
    assert [step.action for step in demo.steps].count("Find_relation") == 4
    assert any(step.action == "Park" for step in demo.steps)
    assert DemonstrationValidator(executor, max_active=24).validate(demo) == []


def test_four_hop_program_stays_within_the_training_turn_budget():
    row = {
        "ID": "four-hop",
        "question": "Which answer follows all four intended relations?",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.gold1', expression1)",
            "expression1 = JOIN('r.gold2', expression1)",
            "expression1 = JOIN('r.gold3', expression1)",
            "expression1 = JOIN('r.gold4', expression1)",
        ],
        "answer": ["m.answer"],
    }

    def executor(functions, target):
        text = "\n".join(functions)
        gold_count = sum(f"r.gold{index}" in text for index in range(1, 5))
        if gold_count == 4:
            return ["m.answer"]
        if gold_count:
            return [f"m.prefix{gold_count}"]
        return ["m.alternative"]

    def provider(query, state, join):
        return [
            RelationOption(join.relation, 0.95, 1),
            RelationOption(f"r.alt_{join.index}_a", 0.85, 2),
            RelationOption(f"r.alt_{join.index}_b", 0.75, 3),
        ]

    demo = DemonstrationBuilder(executor, provider, max_turns=32).build(row)[0]

    assert demo.private_metadata["path_hops"] == 4
    assert len(demo.steps) <= 32
    assert [step.action for step in demo.steps].count("Find_relation") == 4
    assert max(node.depth for node in demo.hypotheses.values()) == 3
    assert DemonstrationValidator(executor, max_active=24).validate(demo) == []


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

    assert recovery.family == "certified_empty_recovery"
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
    assert semantic_actions(demo) == ["Find_relation", "Commit"]
    assert sum(step.action == "Inspect" for step in demo.steps) == 2
    assert DemonstrationValidator(fake_executor, max_active=6).validate(demo) == []
    decisions = decision_sft_records(demo)
    assert len(decisions) == sum(
        step.supervision == "policy_target" for step in demo.steps
    )
    assert all("<action>" in row["messages"][-1]["content"] for row in decisions)
    assert all("<answer>" not in row["messages"][-1]["content"] for row in decisions)
    assert decisions[-1]["messages"][-1]["loss_mask"] == 1
    assert all(
        message.get("loss_mask") == 0
        for message in decisions[-1]["messages"][:-1]
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
    assert semantic_actions(demos[0]) == [
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

    assert [demo.family for demo in demos] == ["certified_empty_recovery"]
    assert any(step.action == "Prune" for step in demos[0].steps)
    continuation_sources = [
        step.arguments[0]
        for step in demos[0].steps
        if step.action == "Find_relation" and step.arguments[0].startswith("expression")
    ]
    assert continuation_sources
    assert set(continuation_sources) == {"expression1"}
    assert semantic_actions(demos[0])[-3:] == [
        "Select",
        "Merge",
        "Commit",
    ]
    assert DemonstrationValidator(typed_executor).validate(demos[0]) == []


def test_nonempty_probe_is_preserved_as_unresolved_and_intervention_is_masked():
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
    assert [demo.family for demo in demos] == [
        "non_destructive_nonempty_recovery"
    ]
    assert all(step.action != "Prune" for step in demos[0].steps)
    intervention = next(
        step for step in demos[0].steps if step.supervision == "intervention"
    )
    assert intervention.action == "Select"
    return_select = next(
        step
        for step in demos[0].steps
        if "switch_while_preserving_unresolved_probe" in step.rationale_facts
    )
    unresolved = {
        node.hypothesis_id
        for node in demos[0].hypotheses.values()
        if node.parent_id == intervention.arguments[0] and node.denotation
    }
    assert unresolved.intersection(return_select.visible_before)
    trajectory = trajectory_sft_record(demos[0])
    rendered = str(trajectory)
    assert "still unresolved" in rendered
    forced_text = f"Select [ {intervention.arguments[0]} ]"
    forced_messages = [
        message
        for message in trajectory["messages"]
        if forced_text in message.get("content", "")
    ]
    assert len(forced_messages) == 1
    assert forced_messages[0]["loss_mask"] == 0
    assert DemonstrationValidator(
        nonempty_wrong_executor, max_active=6
    ).validate(demos[0]) == []
    decisions = decision_sft_records(demos[0])
    targets = [row["messages"][-1]["content"] for row in decisions]
    assert all(forced_text not in target for target in targets)


def test_answer_exact_natural_alternative_is_not_pruned_or_called_complete():
    row = {
        "ID": "answer-exact-alternative",
        "question": "Which answer follows both intended relations?",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.gold1', expression1)",
            "expression1 = JOIN('r.gold2', expression1)",
        ],
        "answer": ["m.answer"],
    }

    def executor(functions, target):
        text = "\n".join(functions)
        if "r.alt1" in text and "r.natural" in text:
            return ["m.answer"]
        return fake_executor(functions, target)

    def provider(query, state, join):
        if "r.alt1" in "\n".join(state):
            return [
                RelationOption("r.natural", 0.95, 1),
                RelationOption("r.alt2", 0.80, 2),
            ]
        return candidates(query, state, join)

    builder = DemonstrationBuilder(executor, provider)
    demo = builder.build(row)[0]

    assert demo.family == "non_destructive_nonempty_recovery"
    natural = next(
        node
        for node in demo.hypotheses.values()
        if node.relation == "r.natural"
    )
    assert natural.denotation == demo.gold_answers
    assert all(
        not (step.action == "Prune" and step.arguments == (natural.hypothesis_id,))
        for step in demo.steps
    )
    committed = next(step.arguments[0] for step in demo.steps if step.action == "Commit")
    assert committed != natural.hypothesis_id
    assert builder.stats["recovery_answer_exact_without_intent_proof"] == 1


def test_answer_equality_does_not_prove_logical_program_equivalence():
    gold = [
        "expression1 = START('m.topic')",
        "expression1 = JOIN('r.gold', expression1)",
    ]
    spurious = [
        "expression1 = START('m.topic')",
        "expression1 = JOIN('r.detour', expression1)",
    ]

    assert not programs_are_intent_equivalent(
        spurious, "expression1", gold, "expression1"
    )
    certificate = certify_program_commit(
        spurious,
        "expression1",
        ["m.answer"],
        gold,
        "expression1",
        ["m.answer"],
    )
    assert certificate.answer_exact
    assert not certificate.intent_equivalent
    assert not certificate.valid


def test_conjunctive_equivalence_ignores_association_and_duplicate_atoms():
    left_associated = [
        "expression0 = START('m.topic')",
        "expression1 = JOIN('r.a', expression0)",
        "expression2 = JOIN('r.b', expression0)",
        "expression3 = AND(expression1, expression2)",
        "expression4 = JOIN('r.c', expression0)",
        "expression5 = AND(expression3, expression4)",
    ]
    right_associated = [
        "expression0 = START('m.topic')",
        "expression1 = JOIN('r.a', expression0)",
        "expression2 = JOIN('r.b', expression0)",
        "expression3 = JOIN('r.c', expression0)",
        "expression4 = AND(expression2, expression3)",
        "expression5 = AND(expression1, expression4)",
    ]
    one_branch = [
        "expression0 = START('m.topic')",
        "expression1 = JOIN('r.a', expression0)",
    ]
    duplicate_branch = [
        "expression0 = START('m.topic')",
        "expression1 = JOIN('r.a', expression0)",
        "expression2 = JOIN('r.a', expression0)",
        "expression3 = AND(expression1, expression2)",
    ]

    assert programs_are_intent_equivalent(
        left_associated, "expression5", right_associated, "expression5"
    )
    assert programs_are_intent_equivalent(
        one_branch, "expression1", duplicate_branch, "expression3"
    )


def test_conjunctive_equivalence_rejects_missing_or_reversed_constraints():
    conjunction = [
        "expression0 = START('m.topic')",
        "expression1 = JOIN('r.a', expression0)",
        "expression2 = JOIN('r.b', expression0)",
        "expression3 = AND(expression1, expression2)",
    ]
    missing = conjunction[:2]
    reversed_relation = [
        "expression0 = START('m.topic')",
        "expression1 = JOIN('r.a_reverse', expression0)",
    ]

    assert not programs_are_intent_equivalent(
        missing, "expression1", conjunction, "expression3"
    )
    assert not programs_are_intent_equivalent(
        reversed_relation, "expression1", missing, "expression1"
    )


def test_validator_rejects_same_answer_commit_with_wrong_logical_program():
    row = {
        "ID": "spurious-commit",
        "question": "Which answer follows the intended relation?",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.gold', expression1)",
        ],
        "answer": ["m.answer"],
    }

    def executor(functions, _target):
        text = "\n".join(functions)
        if "r.gold" in text or "r.detour" in text:
            return ["m.answer"]
        return []

    def options(*_args):
        return [
            RelationOption("r.gold", 0.9, 1),
            RelationOption("r.detour", 0.8, 2),
        ]

    demo = DemonstrationBuilder(executor, options).build(row)[0]
    committed_id = next(
        step.arguments[0] for step in demo.steps if step.action == "Commit"
    )
    committed = demo.hypotheses[committed_id]
    demo.hypotheses[committed_id] = replace(
        committed,
        function_state=(
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.detour', expression1)",
        ),
        relation="r.detour",
    )

    errors = DemonstrationValidator(executor).validate(demo)
    assert any("lacks exact answer-and-intent proof" in error for error in errors)


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
        alternatives = [
            RelationOption(f"r.alt_{rank}", 1.0 - rank / 20, rank)
            for rank in range(1, 13)
        ]
        return [
            *alternatives,
            RelationOption("r.gold1", 0.30, 13),
            RelationOption("r.extra14", 0.25, 14),
        ]

    builder = DemonstrationBuilder(fake_executor, wider_candidates)
    demo = builder.build(row)[0]

    assert demo.family == "adaptive_frontier_widen"
    assert semantic_actions(demo) == [
        "Find_relation", "Widen", "Widen", "Commit"
    ]
    assert demo.private_metadata["gold_rank"] == 13
    assert demo.private_metadata["proposal_recall_at_frontier"] is False
    assert demo.private_metadata["proposal_recall_within_budget"] is True
    assert demo.private_metadata["candidate_future_values"]["r.gold1"]["answer_exact"]
    assert DemonstrationValidator(fake_executor, max_active=24).validate(demo) == []
    rendered = trajectory_sft_record(demo)
    rendered_text = str(rendered["messages"])
    assert "source=m.topic exposed=6/14" in rendered_text
    assert "source=m.topic exposed=12/14" in rendered_text
    assert "source=m.topic exposed=14/14" in rendered_text
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
            RelationOption("r.extra2", 0.80, 4),
            RelationOption("r.last2", 0.75, 5),
            RelationOption("r.sixth2", 0.70, 6),
            RelationOption("r.gold2", 0.65, 7),
            RelationOption("r.eighth2", 0.60, 8),
        ]

    demo = DemonstrationBuilder(fake_executor, continuation_candidates).build(row)[0]

    assert demo.family == "adaptive_frontier_widen"
    assert [step.action for step in demo.steps].count("Widen") == 1
    widen = next(step for step in demo.steps if step.action == "Widen")
    assert widen.arguments == ("expression1",)
    assert widen.created == ()
    assert len(widen.exposed) == 2
    assert all(step.action != "Prune" for step in demo.steps)
    assert DemonstrationValidator(fake_executor, max_active=24).validate(demo) == []


def test_post_widen_state_can_probe_and_recover_without_rejecting_nonempty_evidence():
    row = {
        "ID": "post-widen-recovery",
        "question": "Which answer follows both intended relations?",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.gold1', expression1)",
            "expression1 = JOIN('r.gold2', expression1)",
        ],
        "answer": ["m.answer"],
    }

    def executor(functions, target):
        text = "\n".join(functions)
        if "r.gold1" in text and "r.gold2" in text:
            return ["m.answer"]
        if "r.gold1" in text:
            return ["m.gold_prefix"]
        return ["m.plausible"]

    def provider(query, state, join):
        text = "\n".join(state)
        if not any("JOIN(" in raw for raw in state):
            return [
                *[
                    RelationOption(f"r.alt_{rank}", 1.0 - rank / 20, rank)
                    for rank in range(1, 7)
                ],
                RelationOption("r.gold1", 0.60, 7),
            ]
        if "r.gold1" in text:
            return [
                RelationOption("r.gold2", 0.95, 1),
                RelationOption("r.second_alt", 0.80, 2),
            ]
        return [
            RelationOption("r.probe", 0.95, 1),
            RelationOption("r.probe_alt", 0.80, 2),
        ]

    demo = DemonstrationBuilder(executor, provider, max_turns=32).build(row)[0]

    assert demo.private_metadata["recovery_stratum"] == "post_widen"
    assert demo.private_metadata["probe_outcome"] == "unresolved_nonempty"
    assert [step.action for step in demo.steps].count("Widen") == 1
    assert any(step.action == "Park" for step in demo.steps)
    assert DemonstrationValidator(executor, max_active=24).validate(demo) == []


def test_plausible_frontier_is_not_pruned_for_capacity():
    row = {
        "ID": "full-capacity-eviction",
        "question": "Which answer follows both intended relations?",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.gold1', expression1)",
            "expression1 = JOIN('r.gold2', expression1)",
        ],
        "answer": ["m.answer"],
    }

    def full_frontier_candidates(query, state, join):
        if join.relation == "r.gold1":
            return [
                RelationOption("r.alt1", 0.95, 1),
                RelationOption("r.other1", 0.90, 2),
                RelationOption("r.extra1", 0.85, 3),
                RelationOption("r.extra2", 0.80, 4),
                RelationOption("r.extra3", 0.75, 5),
                RelationOption("r.gold1", 0.70, 6),
            ]
        return [
            RelationOption("r.alt2", 0.95, 1),
            RelationOption("r.gold2", 0.90, 2),
        ]

    demo = DemonstrationBuilder(fake_executor, full_frontier_candidates).build(row)[0]
    assert all(
        "frontier_capacity_eviction" not in step.rationale_facts
        and "frontier_capacity_reservation" not in step.rationale_facts
        for step in demo.steps
    )
    assert DemonstrationValidator(fake_executor, max_active=24).validate(demo) == []


def test_fake_semantic_prune_is_rejected_without_a_certificate():
    row = {
        "ID": "capacity-rationale-too-early",
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

    demo = DemonstrationBuilder(nonempty_wrong_executor, candidates).build(row)[0]
    bad_steps = list(demo.steps)
    return_index = next(
        index
        for index, step in enumerate(bad_steps)
        if "switch_while_preserving_unresolved_probe" in step.rationale_facts
    )
    return_step = bad_steps[return_index]
    nonempty_probe = next(
        node_id
        for node_id in return_step.visible_before
        if demo.hypotheses[node_id].parent_id is not None
        and demo.hypotheses[node_id].denotation
    )
    bad_steps.insert(
        return_index,
        replace(
            return_step,
            action="Prune",
            arguments=(nonempty_probe,),
            created=(),
            rationale_facts=("question_path_mismatch:r.alt1",),
            certificate_kind=None,
            certificate_evidence=(),
        ),
    )

    errors = DemonstrationValidator(
        nonempty_wrong_executor, max_active=6
    ).validate(replace(demo, steps=bad_steps))
    assert any("public contradiction certificate" in error for error in errors)


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


def test_action_budget_does_not_reserve_the_separate_answer_generation():
    row = {
        "ID": "exact-action-budget",
        "question": "Which answer follows the intended relation?",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.gold1', expression1)",
        ],
        "answer": ["m.answer"],
        "candidate_entity_map": {"m.topic": "Topic"},
    }

    def executor(functions, target):
        del target
        return ["m.answer"] if "r.gold1" in "\n".join(functions) else []

    def provider(query, state, join):
        del query, state
        return [RelationOption(join.relation, 0.9, 1)]

    # Find_relation, Inspect, Commit occupy the three action turns. Runtime
    # generates the answer afterwards, outside max_turns.
    builder = DemonstrationBuilder(
        executor,
        provider,
        max_turns=3,
    )
    audit = builder._audit_gold_program_proposals(
        row["question"], compile_gold_plan(row["function_list"])
    )

    assert audit["required_actions"] == 3
    assert audit["required_turns_with_answer"] == 4
    assert audit["budget_checks"]["turns"] is True
    assert audit["runtime_reachable"] is True


def test_symbolic_relation_pages_do_not_consume_active_hypothesis_slots():
    builder = DemonstrationBuilder(
        fake_executor,
        candidates,
        frontier_width=3,
        max_active=5,
    )
    assert builder.max_active == 5


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


def test_builder_resolves_oracle_root_labels_and_excludes_linker_distractors():
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
    assert first.private_metadata["candidate_entities"] == [
        ("Topic Entity", "m.topic")
    ]
    rendered = str(trajectory_sft_record(first))
    assert "Topic Entity" in rendered
    assert "Other Entity" not in rendered
    assert "Answer Entity [m.gold_prefix]" in rendered
    assert "'m.topic' (m.topic)" not in rendered
    assert first.private_metadata["root_entity_provenance"] == "oracle_gold_program"


def test_production_builder_uses_gold_program_as_oracle_entity_link():
    row = {
        "ID": "missing-linker-root",
        "question": "Which answer follows the intended relation?",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.gold1', expression1)",
        ],
        "answer": ["m.gold_prefix"],
    }

    builder = _DemonstrationBuilder(fake_executor, one_hop_candidates)
    demo = builder.build(row)[0]

    assert demo.private_metadata["candidate_entities"] == [("m.topic", "m.topic")]
    assert demo.private_metadata["root_entity_provenance"] == "oracle_gold_program"
    assert builder.stats["oracle_root_entities_used"] == 1


def test_full_program_proposal_audit_applies_the_complete_lazy_runtime_budget():
    functions = [
        "expression1 = START('m.topic')",
        "expression1 = JOIN('r.first', expression1)",
        "expression1 = JOIN('r.deep', expression1)",
    ]

    def provider(_question, _state, decision):
        if decision.relation == "r.first":
            return [
                RelationOption("r.first", 0.99, 1),
                RelationOption("r.other", 0.98, 2),
            ]
        options = [
            RelationOption(f"r.other{index}", 1.0 - index / 100, index)
            for index in range(1, 13)
        ]
        options.append(RelationOption("r.deep", 0.1, 13))
        return options

    builder = _DemonstrationBuilder(
        fake_executor,
        provider,
        max_active=12,
        max_nodes=12,
        max_turns=8,
    )
    audit = builder._audit_gold_program_proposals(
        "question", compile_gold_plan(functions)
    )

    assert audit["all_relations_present"] is True
    assert audit["all_relations_within_budget"] is False
    assert audit["runtime_reachable"] is False
    assert audit["budget_checks"]["turns"] is False
    assert audit["decisions"][1]["rank"] == 13
    assert audit["decisions"][1]["pages_required"] == 3
    assert builder.stats["gold_program_relation_present"] == 2
    assert builder.stats["gold_program_runtime_reachable"] == 0


def test_operator_only_program_is_reachable_without_relation_decisions():
    functions = [
        "expression1 = START('location.location')",
        "expression1 = ARG('ARGMAX', expression1, 'location.location.population')",
    ]
    builder = _DemonstrationBuilder(fake_executor, one_hop_candidates)

    audit = builder._audit_gold_program_proposals(
        "What location has the largest population?",
        compile_gold_plan(functions),
    )

    assert audit["decisions"] == []
    assert audit["all_relations_present"] is True
    assert audit["supported_program"] is True
    assert audit["runtime_reachable"] is True
    assert audit["required_actions"] == 2
    assert builder.stats["gold_program_runtime_reachable"] == 1


def test_builder_never_exposes_gold_types_as_candidate_entities():
    row = {
        "ID": "no-type-leak",
        "question": "Which connected people are chefs?",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.people', expression1)",
            "expression2 = START('dining.chef')",
            "expression1 = AND(expression1, expression2)",
        ],
        "answer": ["m.chef"],
        "extra_info": {
            "candidate_entities": [
                ["Topic", "m.topic"],
                ["Chef", "dining.chef"],
            ]
        },
    }

    def executor(functions, _target):
        text = "\n".join(functions)
        if "AND(" in text:
            return ["m.chef"]
        if "r.people" in text:
            return ["m.chef", "m.other"]
        if "r.alternative" in text:
            return ["m.other"]
        return []

    def options(*_args):
        return [
            RelationOption("r.people", 0.9, 1),
            RelationOption("r.alternative", 0.8, 2),
        ]

    demo = DemonstrationBuilder(executor, options).build(row)[0]

    assert demo.family == "frontier_commit"
    assert semantic_actions(demo) == [
        "Find_relation",
        "Select",
        "Merge",
        "Commit",
    ]
    assert demo.private_metadata["candidate_entities"] == [("Topic", "m.topic")]
    rendered = trajectory_sft_record(demo)
    user_prompt = rendered["messages"][0]["content"]
    assert "dining.chef" not in user_prompt.split("Question:", 1)[0]
    assert any(
        "Merge [ expression1 | dining.chef ]" in message["content"]
        for message in rendered["messages"]
        if message["role"] == "assistant"
    )
    assert DemonstrationValidator(executor).validate(demo) == []


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
    assert semantic_actions(conjunction) == [
        "Find_relation", "Combine", "Commit"
    ]
    assert DemonstrationValidator(fake_executor, max_active=6).validate(conjunction) == []

    bad_steps = list(conjunction.steps)
    combine_index = next(
        index for index, step in enumerate(bad_steps) if step.action == "Combine"
    )
    bad_steps[combine_index] = replace(
        bad_steps[combine_index], arguments=("H0", "H1")
    )
    malformed = replace(conjunction, steps=bad_steps)
    assert any(
        "wrong parents" in error
        for error in DemonstrationValidator(fake_executor, max_active=6).validate(malformed)
    )


def test_conjunction_recovers_before_irreversible_combine():
    row = {
        "ID": "conjunction-recovery",
        "question": "Which entity satisfies both conditions?",
        "function_list": [
            "expression0 = START('m.topic')",
            "expression1 = JOIN('r.left', expression0)",
            "expression2 = JOIN('r.right', expression0)",
            "expression3 = AND(expression1, expression2)",
        ],
        "answer": ["m.shared"],
    }

    def executor(functions, target):
        text = "\n".join(functions)
        if "AND(" in text:
            return ["m.shared"]
        if "r.left" in text:
            return ["m.left", "m.shared"]
        if "r.right" in text:
            return ["m.right", "m.shared"]
        return ["m.plausible"]

    def provider(query, state, join):
        if "r.detour" in "\n".join(state):
            return [
                RelationOption("r.probe", 0.95, 1),
                RelationOption("r.probe_alt", 0.80, 2),
            ]
        return [
            RelationOption("r.detour", 0.98, 1),
            RelationOption("r.left", 0.90, 2),
            RelationOption("r.right", 0.85, 3),
        ]

    conjunction = next(
        demo
        for demo in DemonstrationBuilder(executor, provider).build(row)
        if demo.family == "conjunction"
    )

    assert conjunction.private_metadata["recovery_stratum"] == "conjunction"
    assert conjunction.private_metadata["probe_outcome"] == "unresolved_nonempty"
    assert semantic_actions(conjunction)[-2:] == ["Combine", "Commit"]
    assert any(step.action == "Park" for step in conjunction.steps)
    assert DemonstrationValidator(executor, max_active=24).validate(conjunction) == []


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
    typed = [demo for demo in demos if demo.family == "conjunction"]

    assert len(typed) == 1
    conjunction = typed[0]
    assert semantic_actions(conjunction) == [
        "Find_relation",
        "Find_relation",
        "Combine",
        "Select",
        "Merge",
        "Commit",
    ]
    roots = [
        step.arguments for step in conjunction.steps if step.action == "Find_relation"
    ]
    assert roots[:2] == [("m.school_a",), ("m.school_b",)]
    assert DemonstrationValidator(two_root_executor, max_active=6).validate(conjunction) == []


def test_builds_typed_public_deep_conjunction_from_two_multihop_roots():
    row = {
        "ID": "deep-two-root",
        "question": "Which item follows both two-hop branches?",
        "function_list": [
            "expression = START('m.left_root')",
            "expression = JOIN('r.left1', expression)",
            "expression = JOIN('r.left2', expression)",
            "expression1 = START('m.right_root')",
            "expression1 = JOIN('r.right1', expression1)",
            "expression1 = JOIN('r.right2', expression1)",
            "expression = AND(expression, expression1)",
            "expression2 = START('answer.type')",
            "expression = AND(expression, expression2)",
            "expression = STOP(expression)",
        ],
        "answer": ["m.shared"],
    }

    def provider(query, state, join):
        return [
            RelationOption(join.relation, 0.9, 1),
            RelationOption("r.alt", 0.8, 2),
        ]

    def executor(functions, target):
        text = "\n".join(functions)
        if "r.alt" in text:
            return ["m.alt"]
        if text.count("AND(") >= 2:
            return ["m.shared"]
        if "AND(" in text:
            return ["m.shared", "m.other"]
        if "r.left2" in text:
            return ["m.left", "m.shared", "m.other"]
        if "r.right2" in text:
            return ["m.right", "m.shared", "m.other"]
        if "r.left1" in text:
            return ["m.left_prefix"]
        if "r.right1" in text:
            return ["m.right_prefix"]
        return []

    builder = DemonstrationBuilder(executor, provider, max_turns=32)
    demos = builder.build(row)
    deep = [demo for demo in demos if demo.family == "deep_conjunction_progress"]

    assert len(deep) == 1
    assert [step.action for step in deep[0].steps].count("Find_relation") == 4
    assert semantic_actions(deep[0])[-4:] == [
        "Combine",
        "Select",
        "Merge",
        "Commit",
    ]
    assert DemonstrationValidator(executor, max_active=6).validate(deep[0]) == []


def test_deep_conjunction_can_continue_after_combining_branches():
    row = {
        "ID": "deep-two-root-tail",
        "question": "Which answer follows the intersection and final relation?",
        "function_list": [
            "expression = START('m.left_root')",
            "expression = JOIN('r.left1', expression)",
            "expression = JOIN('r.left2', expression)",
            "expression1 = START('m.right_root')",
            "expression1 = JOIN('r.right1', expression1)",
            "expression = AND(expression, expression1)",
            "expression = JOIN('r.tail', expression)",
            "expression = STOP(expression)",
        ],
        "answer": ["m.answer"],
    }

    def provider(query, state, join):
        return [
            RelationOption(join.relation, 0.9, 1),
            RelationOption("r.alt", 0.8, 2),
        ]

    def executor(functions, target):
        text = "\n".join(functions)
        if "r.alt" in text:
            return ["m.alt"]
        if "r.tail" in text:
            return ["m.answer"]
        if "AND(" in text:
            return ["m.intermediate"]
        if "r.left2" in text:
            return ["m.left", "m.intermediate"]
        if "r.right1" in text:
            return ["m.right", "m.intermediate"]
        if "r.left1" in text:
            return ["m.left_prefix"]
        return []

    builder = DemonstrationBuilder(executor, provider, max_turns=32)
    demos = builder.build(row)
    deep = [demo for demo in demos if demo.family == "deep_conjunction_progress"]

    assert len(deep) == 1
    actions = semantic_actions(deep[0])
    combine_index = actions.index("Combine")
    continuation = [action for action in actions[combine_index + 1 :] if action != "Prune"]
    assert continuation[:2] == ["Select", "Find_relation"]
    assert actions[-1] == "Commit"
    assert DemonstrationValidator(executor, max_active=6).validate(deep[0]) == []


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
