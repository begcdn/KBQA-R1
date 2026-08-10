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
Action2: Prune [ H3 ]
Action3: Combine [ H2 | H4 ]
Action4: Commit [ H5 ]
</action>"""

    actions = parser.parse_actions_from_text(text)
    assert [action.action_type for action in actions] == [
        ActionType.SELECT_HYPOTHESIS,
        ActionType.PRUNE_HYPOTHESIS,
        ActionType.COMBINE_HYPOTHESES,
        ActionType.COMMIT_HYPOTHESIS,
    ]
    assert actions[2].arguments == ["H2", "H4"]
