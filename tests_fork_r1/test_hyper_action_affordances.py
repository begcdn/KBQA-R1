import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "data_process"
    / "add_hyper_action_affordances.py"
)
SPEC = importlib.util.spec_from_file_location("add_hyper_action_affordances", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_adds_only_current_graph_targets_and_nonempty_commit_targets():
    text = """<information>
<hypothesis_graph>
active=2 capacity=24 stored=4 parked=1 execution_attempts=4/24 selected=none committed=none
H1 [active] parents=H0 operation=expand via=r.child source=policy depth=1 path=r.root -> r.child answers=1: m.answer
H3 [active] parents=ROOT operation=expand via=r.empty source=alternative depth=0 path=r.empty answers=0: empty
Actions: Select, Find_relation [ source ], Widen [ source ], Inspect [ Pn ], Park, Recall, Combine, Prune, Commit, or Abstain.
</hypothesis_graph>
<parked_hypotheses>
H2 path=r.other answers=2
</parked_hypotheses>
</information>"""

    migrated, graphs, catalogs = MODULE.add_action_affordances(text)

    assert graphs == 1
    assert catalogs == 0
    assert "Select/Park=[H1,H3]" in migrated
    assert "Commit(nonempty active)=[H1]" in migrated
    assert "Combine/Prune candidates=[H1,H3]" in migrated
    assert "Recall=[H2]" in migrated
    assert "H0" not in migrated.split("Available targets:", 1)[1].splitlines()[0]


def test_adds_visible_proposals_and_only_available_widen_source():
    text = """<proposal_catalog>
source=expression2 exposed=2/3 page_size=2
P1 rank=2 relation=r.visible score=0.5000 status=visible
P0 rank=1 relation=r.failed score=0.6000 status=failed
Use Inspect [ Pn ] to execute one visible proposal; Widen only reveals the next page.
</proposal_catalog>
<proposal_catalog>
source=m.topic exposed=1/1 page_size=2
P2 rank=1 relation=r.done score=0.7000 status=visible
Use Inspect [ Pn ] to execute one visible proposal; Widen only reveals the next page.
</proposal_catalog>"""

    migrated, graphs, catalogs = MODULE.add_action_affordances(text)

    assert graphs == 0
    assert catalogs == 2
    assert "Inspect=[P1]; Widen=[expression2]" in migrated
    assert "Inspect=[P2]; Widen=[none]" in migrated
    assert "Inspect=[P0" not in migrated


def test_migration_is_idempotent_and_updates_terminal_commit_protocol():
    text = """- Use `Commit [ Hn ]` when one hypothesis expresses the complete question. After the environment confirms it, return its values inside <answer>.
- Hypothesis IDs and execution results are owned by the environment. Never invent or edit them.
Preserve plausible alternatives until later execution distinguishes them. Select is not Commit: selecting one hypothesis for expansion does not reject the others. After Commit, perform no more graph actions and copy the committed values exactly into <answer>.
<hypothesis_graph>
active=0 capacity=24 stored=0 parked=0 execution_attempts=0/24 selected=none committed=none
Actions: Select, Find_relation [ source ], Widen [ source ], Inspect [ Pn ], Park, Recall, Combine, Prune, Commit, or Abstain.
</hypothesis_graph>"""

    once, graphs, _ = MODULE.add_action_affordances(text)
    twice, second_graphs, second_catalogs = MODULE.add_action_affordances(once)

    assert graphs == 1
    assert second_graphs == 0
    assert second_catalogs == 0
    assert once == twice
    assert "Commit is terminal" in once
    assert "Commit ends the search" in once
    assert "Every observation lists current action targets" in once
    assert "return its values inside <answer>" not in once
