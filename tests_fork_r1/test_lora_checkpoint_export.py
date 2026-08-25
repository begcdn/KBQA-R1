import importlib.util
import sys
import types
from pathlib import Path

import torch


_MODULE_PATH = Path(__file__).parents[1] / "verl" / "utils" / "checkpoint" / "peft_adapter.py"
_SPEC = importlib.util.spec_from_file_location("hyper_peft_adapter", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load PEFT adapter exporter from {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
save_peft_adapter = _MODULE.save_peft_adapter


class _FakeSafeTensorReader:
    def __init__(self, tensors):
        self.tensors = tensors

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def keys(self):
        return self.tensors.keys()

    def get_tensor(self, key):
        return self.tensors[key]


def test_save_peft_adapter_passes_full_state_to_peft_once(tmp_path, monkeypatch):
    full_state = {
        "base.weight": torch.ones(2, 2),
        "layer.lora_A.default.weight": torch.ones(2, 3),
        "layer.lora_B.default.weight": torch.ones(3, 2),
    }
    extracted = {
        "layer.lora_A.weight": full_state["layer.lora_A.default.weight"],
        "layer.lora_B.weight": full_state["layer.lora_B.default.weight"],
    }
    saved = {}

    def get_adapter_state(_model, state_dict):
        assert state_dict is full_state
        return extracted

    peft_save = types.ModuleType("peft.utils.save_and_load")
    peft_save.get_peft_model_state_dict = get_adapter_state
    monkeypatch.setitem(sys.modules, "peft.utils.save_and_load", peft_save)

    safetensors = types.ModuleType("safetensors")
    safetensors.safe_open = lambda *_args, **_kwargs: _FakeSafeTensorReader(saved)
    monkeypatch.setitem(sys.modules, "safetensors", safetensors)

    class FakeModel:
        def save_pretrained(self, output_dir, state_dict, safe_serialization):
            assert state_dict is full_state
            assert safe_serialization is True
            (output_dir / "adapter_model.safetensors").write_bytes(b"not-empty")
            saved.update(extracted)

    tensors, parameters = save_peft_adapter(FakeModel(), full_state, tmp_path)

    assert tensors == 2
    assert parameters == 12


def test_save_peft_adapter_rejects_empty_extraction(tmp_path, monkeypatch):
    peft_save = types.ModuleType("peft.utils.save_and_load")
    peft_save.get_peft_model_state_dict = lambda *_args, **_kwargs: {}
    monkeypatch.setitem(sys.modules, "peft.utils.save_and_load", peft_save)

    class FakeModel:
        def save_pretrained(self, *_args, **_kwargs):
            raise AssertionError("empty adapters must fail before writing")

    try:
        save_peft_adapter(FakeModel(), {"base.weight": torch.ones(1)}, tmp_path)
    except RuntimeError as exc:
        assert "no LoRA weights" in str(exc)
    else:
        raise AssertionError("empty adapter extraction was accepted")
