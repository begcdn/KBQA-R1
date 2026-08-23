import threading
import sys
import types

if "pyodbc" not in sys.modules:
    pyodbc = types.ModuleType("pyodbc")
    pyodbc.Connection = object
    pyodbc.Error = Exception
    pyodbc.SQL_CHAR = 1
    pyodbc.SQL_WCHAR = 2
    sys.modules["pyodbc"] = pyodbc

from kbqa_r1.sexpr.simcse_tool import SimCSE


def test_fork_index_shares_encoder_but_not_mutable_index():
    source = object.__new__(SimCSE)
    source.tokenizer = object()
    source.model = object()
    source.device = "cpu"
    source._device_lock = threading.RLock()
    source._current_device = "cpu"
    source.index = {"sentences": ["source"]}
    source.is_faiss_index = True
    source.num_cells = 100
    source.num_cells_in_search = 10
    source.pooler = "cls"

    fork = source.fork_index()

    assert fork.model is source.model
    assert fork.tokenizer is source.tokenizer
    assert fork._device_lock is source._device_lock
    assert fork.index is None
    assert not fork.is_faiss_index

    fork.index = {"sentences": ["fork"]}
    assert source.index == {"sentences": ["source"]}
