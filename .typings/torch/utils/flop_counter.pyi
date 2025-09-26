import torch
import types
from _typeshed import Incomplete
from torch.utils._python_dispatch import TorchDispatchMode
from typing import Any, Callable, TypeVar
from typing_extensions import ParamSpec

__all__ = ['FlopCounterMode', 'register_flop_formula']

_T = TypeVar('_T')
_P = ParamSpec('_P')

def register_flop_formula(targets, get_raw: bool = False) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...

class FlopCounterMode:
    """
    ``FlopCounterMode`` is a context manager that counts the number of flops within its context.

    It does this using a ``TorchDispatchMode``.

    It also supports hierarchical output by passing a module (or list of
    modules) to FlopCounterMode on construction. If you do not need hierarchical
    output, you do not need to use it with a module.

    Example usage

    .. code-block:: python

        mod = ...
        with FlopCounterMode(mod) as flop_counter:
            mod.sum().backward()

    """
    flop_counts: dict[str, dict[Any, int]]
    depth: Incomplete
    display: Incomplete
    mode: _FlopCounterMode | None
    flop_registry: Incomplete
    mod_tracker: Incomplete
    def __init__(self, mods: torch.nn.Module | list[torch.nn.Module] | None = None, depth: int = 2, display: bool = True, custom_mapping: dict[Any, Any] | None = None) -> None: ...
    def get_total_flops(self) -> int: ...
    def get_flop_counts(self) -> dict[str, dict[Any, int]]:
        """Return the flop counts as a dictionary of dictionaries.

        The outer
        dictionary is keyed by module name, and the inner dictionary is keyed by
        operation name.

        Returns:
            Dict[str, Dict[Any, int]]: The flop counts as a dictionary.
        """
    def get_table(self, depth=None): ...
    def __enter__(self): ...
    def __exit__(self, *args): ...
    def _count_flops(self, func_packet, out, args, kwargs): ...

class _FlopCounterMode(TorchDispatchMode):
    counter: Incomplete
    def __init__(self, counter: FlopCounterMode) -> None: ...
    def __torch_dispatch__(self, func, types, args=(), kwargs=None): ...
