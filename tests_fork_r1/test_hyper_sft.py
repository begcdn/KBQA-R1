from kbqa_r1.hyper_sft import add_graph_decisions, latest_policy_node


GRAPH = """<information>
Executed.
<hypothesis_graph>
active=2 nodes=2 executions=2
H0 [active] parent=ROOT via=r.gold source=policy depth=0 answers=1: m.gold
H1 [active] parent=ROOT via=r.sibling source=sibling depth=0 answers=1: m.other
Actions: Explore/Expand, Combine, Prune, or Commit one active hypothesis.
</hypothesis_graph>
</information>"""


def test_latest_policy_node_uses_newest_factual_node():
    text = GRAPH + "\nH12 [active] parent=H0 via=r.next source=policy depth=1 answers=1: m.x"
    assert latest_policy_node(text) == "H12"


def test_add_graph_decisions_uses_observed_ids_and_commits_before_answer():
    messages = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "<action>Find_relation [ x | r ]</action>"},
        {"role": "tool", "content": GRAPH},
        {"role": "assistant", "content": "<answer>m.gold</answer>"},
    ]
    converted = add_graph_decisions(messages)
    contents = [message["content"] for message in converted]
    assert any("Select [ H0 ]" in content for content in contents)
    assert any("Commit [ H0 ]" in content for content in contents)
    assert contents[-1] == "<answer>m.gold</answer>"


def test_add_graph_decisions_does_not_duplicate_existing_control_action():
    messages = [
        {"role": "tool", "content": GRAPH},
        {"role": "assistant", "content": "<action>Select [ H0 ]</action>"},
    ]
    converted = add_graph_decisions(messages)
    assert len(converted) == len(messages)
