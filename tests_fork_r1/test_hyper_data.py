from dataclasses import replace

from kbqa_r1.hyper_data import (
    DemonstrationBuilder,
    DemonstrationValidator,
    IneligibleProgram,
    RelationOption,
    compile_gold_plan,
    relation_hint,
    step_sft_records,
    trajectory_sft_record,
)


def fake_executor(functions, target):
    text = "\n".join(functions)
    if "AND(" in text:
        return ["m.shared"]
    if "r.gold2" in text:
        return ["m.answer"] if "r.gold1" in text else ["m.wrong_terminal"]
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
    if "left and right" in query:
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
    demo = DemonstrationBuilder(fake_executor, candidates).build(row)[0]
    assert demo.family == "delayed_frontier_recovery"
    actions = [step.action for step in demo.steps]
    assert actions[:3] == ["Find_relation", "Select", "Find_relation"]
    assert "Prune" in actions
    assert actions[-3:] == ["Find_relation", "Select", "Commit"]
    assert demo.private_metadata["proposal_relations"] == [
        "r.alt1", "r.gold1", "r.other1"
    ]
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


def test_one_hop_frontier_teaches_select_then_commit():
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
    assert [step.action for step in demo.steps] == ["Find_relation", "Select", "Commit"]
    assert DemonstrationValidator(fake_executor, max_active=6).validate(demo) == []


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
        "Find_relation", "Prune", "Select", "Select", "Combine", "Commit"
    ]
    assert DemonstrationValidator(fake_executor, max_active=6).validate(conjunction) == []

    bad_steps = list(conjunction.steps)
    bad_steps[4] = replace(bad_steps[4], arguments=("H0", "H1"))
    malformed = replace(conjunction, steps=bad_steps)
    assert any(
        "do not match required branches" in error
        for error in DemonstrationValidator(fake_executor, max_active=6).validate(malformed)
    )


def test_reverse_relation_hint_and_rewrite_are_preserved():
    from kbqa_r1.hyper_data import replace_join_relation

    raw = "expression1 = JOIN('(R people.person.parents)', expression1)"
    assert relation_hint("(R people.person.children)") == "reverse children"
    assert replace_join_relation(raw, "(R people.person.children)") == (
        "expression1 = JOIN('(R people.person.children)', expression1)"
    )
