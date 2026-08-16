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
Action3: Prune [ H3 ]
Action4: Combine [ H2 | H4 ]
Action5: Commit [ H5 ]
</action>"""

    actions = parser.parse_actions_from_text(text)
    assert [action.action_type for action in actions] == [
        ActionType.SELECT_HYPOTHESIS,
        ActionType.WIDEN_FRONTIER,
        ActionType.PRUNE_HYPOTHESIS,
        ActionType.COMBINE_HYPOTHESES,
        ActionType.COMMIT_HYPOTHESIS,
    ]
    assert actions[1].arguments == ["expression2"]
    assert actions[3].arguments == ["H2", "H4"]


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
