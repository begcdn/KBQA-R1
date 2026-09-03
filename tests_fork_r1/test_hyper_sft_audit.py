import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_hyper_sft_corpus.py"
SPEC = importlib.util.spec_from_file_location("audit_hyper_sft_corpus", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


OBSERVATION = """<information>
<hypothesis_graph>
Available targets: Select=[H1]; Park=[H1,H2]; Commit(nonempty active)=[H1]; Combine=[H1|H2]; Prune candidates=[H2]; Recall=[H3]; Find_relation sources=[m.topic].
</hypothesis_graph>
<proposal_catalog>
Available proposal targets: Inspect=[P4]; Widen=[m.topic].
</proposal_catalog>
</information>"""


@pytest.mark.parametrize(
    "target",
    (
        "<action>Select [ H1 ]</action>",
        "<action>Park [ H2 ]</action>",
        "<action>Commit [ H1 ]</action>",
        "<action>Combine [ H1 | H2 ]</action>",
        "<action>Prune [ H2 ]</action>",
        "<action>Recall [ H3 ]</action>",
        "<action>Find_relation [ m.topic ]</action>",
        "<action>Inspect [ P4 ]</action>",
        "<action>Widen [ m.topic ]</action>",
    ),
)
def test_accepts_only_current_rendered_graph_targets(target):
    assert MODULE.assert_supervised_action_afforded(
        OBSERVATION, target, row_index=7
    )


def test_rejects_a_stale_graph_identifier():
    with pytest.raises(RuntimeError, match="unavailable Select target H0"):
        MODULE.assert_supervised_action_afforded(
            OBSERVATION, "<action>Select [ H0 ]</action>", row_index=8
        )


def test_accepts_reverse_order_for_advertised_combine_pair():
    assert MODULE.assert_supervised_action_afforded(
        OBSERVATION, "<action>Combine [ H2 | H1 ]</action>", row_index=9
    )


def test_rejects_a_nonadvertised_combine_pair():
    with pytest.raises(RuntimeError, match="unavailable Combine target H2\\|H3"):
        MODULE.assert_supervised_action_afforded(
            OBSERVATION, "<action>Combine [ H2 | H3 ]</action>", row_index=9
        )


def test_accepts_initial_find_relation_candidate_sources():
    observation = """Candidate Entities: ['Thing (extended) across
two lines' (m.0123)]
Candidate Literals: ['5' (5^^http://www.w3.org/2001/XMLSchema#integer)]
Question: Which thing?"""

    assert MODULE.assert_supervised_action_afforded(
        observation,
        "<action>Find_relation [ m.0123 ]</action>",
        row_index=9,
    )
    assert MODULE.assert_supervised_action_afforded(
        observation,
        "<action>Find_relation [ 5^^http://www.w3.org/2001/XMLSchema#integer ]</action>",
        row_index=10,
    )


def test_operator_target_is_left_for_executable_replay_gate():
    assert not MODULE.assert_supervised_action_afforded(
        OBSERVATION,
        "<action>Count [ expression1 ]</action>",
        row_index=9,
    )


def test_abstain_is_rejected_under_f1_runtime_contract():
    with pytest.raises(RuntimeError, match="runtime F1 mode disables it"):
        MODULE.assert_supervised_action_afforded(
            OBSERVATION, "<action>Abstain</action>", row_index=10
        )
