from kbqa_r1.hyper_data import (
    DemonstrationBuilder,
    DemonstrationValidator,
    IneligibleProgram,
    RelationOption,
    compile_gold_plan,
    step_sft_records,
)
from dataclasses import replace


def fake_executor(functions, target):
    text = "\n".join(functions)
    if target == "expression1":
        if "r.wrong" in text:
            return ["m.wrong"]
        if "r.left" in text:
            return ["m.a", "m.shared"]
        return ["m.gold"]
    if target == "expression2":
        return ["m.b", "m.shared"]
    if target == "expression3":
        return ["m.shared"]
    return []


def candidates(question, state, join):
    return [
        RelationOption("r.wrong", 0.92, 1),
        RelationOption(join.relation, 0.81, 2),
    ]


def gold_first_candidates(question, state, join):
    return [
        RelationOption(join.relation, 0.92, 1),
        RelationOption("r.wrong", 0.81, 2),
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
        "expression1 = JOIN('r.gold', expression1)",
    ]
    try:
        compile_gold_plan(functions)
    except IneligibleProgram as exc:
        assert "terminal" in str(exc)
    else:
        raise AssertionError("statements after STOP must be rejected")


def test_builds_and_replays_wrong_sibling_recovery():
    row = {
        "ID": "q1",
        "question": "Which answer follows the intended relation?",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.gold', expression1)",
        ],
        "answer": ["m.gold"],
    }
    demo = DemonstrationBuilder(fake_executor, candidates).build(row)[0]
    assert demo.family == "wrong_sibling_recovery"
    assert demo.private_metadata["distractor_relation"] == "r.wrong"
    assert [step.action for step in demo.steps] == ["Prune", "Select", "Commit"]
    assert DemonstrationValidator(fake_executor).validate(demo) == []

    public = step_sft_records(demo)
    assert all("private_metadata" not in record for record in public)
    assert all("role" not in str(record) for record in public)
    assert all("gold_relation" not in str(record) for record in public)
    assert any("r.gold" in str(record) for record in public)


def test_skips_execution_equivalent_alternative():
    def equivalent_executor(functions, target):
        return ["m.gold"]

    row = {
        "ID": "q2",
        "question": "Question",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.gold', expression1)",
        ],
        "answer": ["m.gold"],
    }
    demos = DemonstrationBuilder(equivalent_executor, candidates).build(row)
    assert demos == []


def test_teaches_correct_direct_commit_without_needless_switching():
    row = {
        "ID": "direct",
        "question": "Which answer follows the intended relation?",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.gold', expression1)",
        ],
        "answer": ["m.gold"],
    }
    demo = DemonstrationBuilder(fake_executor, gold_first_candidates).build(row)[0]
    assert demo.family == "correct_top1_commit"
    assert demo.hypotheses["H0"].role == "gold"
    assert [step.action for step in demo.steps] == ["Prune", "Select", "Commit"]
    assert DemonstrationValidator(fake_executor).validate(demo) == []


def test_rejects_rows_without_annotated_answers():
    row = {
        "ID": "missing-answer",
        "question": "Question",
        "function_list": [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.gold', expression1)",
        ],
    }
    assert DemonstrationBuilder(fake_executor, candidates).build(row) == []


def test_conjunction_requires_both_branches():
    row = {
        "ID": "q3",
        "question": "Which entity satisfies both conditions?",
        "function_list": [
            "expression1 = START('m.left')",
            "expression1 = JOIN('r.left', expression1)",
            "expression2 = START('m.right')",
            "expression3 = AND(expression1, expression2)",
        ],
        "answer": ["m.shared"],
    }
    demos = DemonstrationBuilder(fake_executor, lambda *args: []).build(row)
    conjunction = next(demo for demo in demos if demo.family == "conjunction")
    assert [step.action for step in conjunction.steps] == ["Select", "Select", "Combine", "Commit"]
    assert DemonstrationValidator(fake_executor).validate(conjunction) == []

    bad_steps = list(conjunction.steps)
    bad_steps[2] = replace(bad_steps[2], arguments=("H0", "H2"))
    malformed = replace(conjunction, steps=bad_steps)
    assert any(
        "do not match required branches" in error
        for error in DemonstrationValidator(fake_executor).validate(malformed)
    )


def test_reverse_relation_rewrite_is_preserved():
    from kbqa_r1.hyper_data import replace_join_relation

    raw = "expression1 = JOIN('(R people.person.parents)', expression1)"
    assert replace_join_relation(raw, "(R people.person.children)") == (
        "expression1 = JOIN('(R people.person.children)', expression1)"
    )
