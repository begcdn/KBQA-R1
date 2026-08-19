import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "data_process"
    / "smoke_hyper_count_prune.py"
)
SPEC = importlib.util.spec_from_file_location("smoke_hyper_count_prune", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_count_prune_smoke_protocol_with_fake_executor():
    def executor(functions, target):
        del target
        return ["0"] if any("COUNT(" in statement for statement in functions) else []

    result = MODULE.run_smoke(executor)

    assert result["status"] == "passed"
    assert result["count_prune_rejected"] is True
    assert result["count_result"] == ["0"]
    assert result["count_commit"] is True
    assert result["noncount_prune_certificate"]["kind"] == "empty_monotone"
