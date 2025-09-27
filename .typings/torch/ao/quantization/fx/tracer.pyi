import torch
from _typeshed import Incomplete
from torch.fx._symbolic_trace import Tracer
from torch.fx.proxy import Scope
from typing import Callable

__all__ = ['QuantizationTracer']

class ScopeContextManager(torch.fx.proxy.ScopeContextManager):
    def __init__(self, scope: Scope, current_module: torch.nn.Module, current_module_path: str) -> None: ...

class QuantizationTracer(Tracer):
    skipped_module_names: Incomplete
    skipped_module_classes: Incomplete
    scope: Incomplete
    record_stack_traces: bool
    def __init__(self, skipped_module_names: list[str], skipped_module_classes: list[Callable]) -> None: ...
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool: ...
