"""
S-Expression processing module for KBQA-R1
This module provides s-expression generation, validation, and execution capabilities
to replace direct SPARQL generation approach.
"""

from importlib import import_module

# Import constants from separate file to avoid circular imports
from .constants import COMPARISON_MODE_MAPPING, COMPARISON_MODE_READABLE_MAPPING

# Keep action parsing usable for CPU-only corpus audits.
from .action_parser import ActionParser


_LAZY_IMPORTS = {
    "FunctionBuilder": (".function_builder", "FunctionBuilder"),
    "SExprGenerator": (".sexpr_generator", "SExprGenerator"),
    "SExprValidator": (".sexpr_validator", "SExprValidator"),
    "SExprExecutor": (".sexpr_executor", "SExprExecutor"),
    "RelationRetrieval": (".relation_retrieval", "RelationRetrieval"),
    "ExecutionValidator": (".execution_validator", "ExecutionValidator"),
    "DynamicRelationRetrieval": (
        ".dynamic_relation_retrieval",
        "DynamicRelationRetrieval",
    ),
}


def __getattr__(name):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(name)
    module_name, attribute = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value

__all__ = [
    'ActionParser',
    'FunctionBuilder', 
    'SExprGenerator',
    'SExprValidator',
    'SExprExecutor',
    'RelationRetrieval',
    'ExecutionValidator',
    'DynamicRelationRetrieval',
    'COMPARISON_MODE_MAPPING',
    'COMPARISON_MODE_READABLE_MAPPING'
]
