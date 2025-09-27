import weakref
from . import config as config
from .utils import ExactWeakKeyDictionary as ExactWeakKeyDictionary, nn_module_has_global_hooks as nn_module_has_global_hooks
from _typeshed import Incomplete
from torch.nn import Module as Module
from typing import Any

unpatched_nn_module_init: Incomplete

class MutationTracker:
    db: ExactWeakKeyDictionary
    mutation_count: int
    watchers: list[weakref.ReferenceType[Any]]
    def __init__(self) -> None: ...
    def on_mutation(self, name: str) -> None: ...
    def track(self, guarded_code: Any) -> None: ...

def watch(obj: Any, guarded_code: Any) -> None:
    """invalidate guarded_code when obj is mutated"""
def ensure_patched(cls) -> None: ...

class GenerationTracker:
    generation: int
    dynamic_classes: ExactWeakKeyDictionary
    generation_values: ExactWeakKeyDictionary
    @classmethod
    def tag(cls, obj: Any) -> None: ...
    @staticmethod
    def mark_class_dynamic(cls) -> None: ...
    @classmethod
    def get_generation_value(cls, obj: Any) -> int: ...
    @classmethod
    def check(cls, obj: Any) -> bool: ...
    @classmethod
    def clear(cls) -> None: ...

def is_dynamic_nn_module(obj: Any, is_export: bool) -> bool:
    """Check for nn.Modules() created dynamically or mutated"""
def install_generation_tagging_init() -> None:
    """
    Monkey patch torch.nn.Module.__init__ and torch.nn.Module.__setstate__
    so we can detect nn.Module instances created dynamically inside forward methods.
    """
