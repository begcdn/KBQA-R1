import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "kbqa_r1" / "sexpr" / "action_parser.py"
SPEC = importlib.util.spec_from_file_location("hyper_action_parser", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ActionParser = MODULE.ActionParser
ActionType = MODULE.ActionType


def test_hyper_graph_actions_parse():
    parser = ActionParser()
    text = """<action>
Action1: Select [ H2 ]
Action2: Widen [ expression2 ]
Action3: Inspect [ P7 ]
Action4: Park [ H3 ]
Action5: Recall [ H3 ]
Action6: Prune [ H3 ]
Action7: Combine [ H2 | H4 ]
Action8: Commit [ H5 ]
Action9: Abstain
</action>"""

    actions = parser.parse_actions_from_text(text)
    assert [action.action_type for action in actions] == [
        ActionType.SELECT_HYPOTHESIS,
        ActionType.WIDEN_FRONTIER,
        ActionType.INSPECT_PROPOSAL,
        ActionType.PARK_HYPOTHESIS,
        ActionType.RECALL_HYPOTHESIS,
        ActionType.PRUNE_HYPOTHESIS,
        ActionType.COMBINE_HYPOTHESES,
        ActionType.COMMIT_HYPOTHESIS,
        ActionType.ABSTAIN,
    ]
    assert actions[1].arguments == ["expression2"]
    assert actions[2].arguments == ["P7"]
    assert actions[6].arguments == ["H2", "H4"]
    assert actions[-1].arguments == []


def test_find_relation_supports_hyper_source_only_and_legacy_intent():
    parser = ActionParser()
    hyper = parser.parse_action("Find_relation [ expression2 ]")
    legacy = parser.parse_action(
        "Find_relation [ expression2 | place where the person died ]"
    )
    assert hyper.action_type == ActionType.FIND_RELATION
    assert hyper.arguments == ["expression2"]
    assert legacy.arguments == ["expression2", "place where the person died"]


def test_action_span_points_to_tagged_action_not_thought_copy():
    parser = ActionParser()
    text = (
        "<think>I may use Select [ H2 ] later.</think>\n"
        "<action>Action1: Select [ H2 ] trailing commentary</action>"
    )

    action = parser.parse_actions_from_text(text)[0]

    assert text[slice(*action.source_span)] == "Action1: Select [ H2 ]"


def test_actions_from_multiple_blocks_keep_distinct_offsets():
    parser = ActionParser()
    text = "<action>Select [ H1 ]</action> then <action>Commit [ H1 ]</action>"

    actions = parser.parse_actions_from_text(text)

    assert [text[slice(*action.source_span)] for action in actions] == [
        "Select [ H1 ]",
        "Commit [ H1 ]",
    ]


def test_single_action_response_requires_one_complete_payload():
    parser = ActionParser()
    valid = "<think>free reasoning</think><action> Select [ H2 ] </action>"

    action = parser.parse_single_action_response(valid)

    assert action is not None
    assert action.action_type == ActionType.SELECT_HYPOTHESIS
    assert valid[slice(*action.source_span)] == "Select [ H2 ]"


def test_single_action_response_rejects_protocol_extras():
    parser = ActionParser()

    assert parser.parse_single_action_response(
        "<action>Select [ H2 ] trailing commentary</action>"
    ) is None
    assert parser.parse_single_action_response(
        "<action>Select [ H2 ]\nCommit [ H2 ]</action>"
    ) is None
    assert parser.parse_single_action_response(
        "<action>Select [ H2 ]</action><action>Commit [ H2 ]</action>"
    ) is None
