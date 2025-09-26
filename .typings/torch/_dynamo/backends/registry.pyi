import functools
import torch
from _typeshed import Incomplete
from collections.abc import Sequence
from importlib.metadata import EntryPoint
from torch import fx as fx
from typing import Callable, Protocol

log: Incomplete

class CompiledFn(Protocol):
    def __call__(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]: ...
CompilerFn = Callable[[fx.GraphModule, list[torch.Tensor]], CompiledFn]
_BACKENDS: dict[str, EntryPoint | None]
_COMPILER_FNS: dict[str, CompilerFn]

def register_backend(compiler_fn: CompilerFn | None = None, name: str | None = None, tags: Sequence[str] = ()):
    """
    Decorator to add a given compiler to the registry to allow calling
    `torch.compile` with string shorthand.  Note: for projects not
    imported by default, it might be easier to pass a function directly
    as a backend and not use a string.

    Args:
        compiler_fn: Callable taking a FX graph and fake tensor inputs
        name: Optional name, defaults to `compiler_fn.__name__`
        tags: Optional set of string tags to categorize backend with
    """

register_debug_backend: Incomplete
register_experimental_backend: Incomplete

def lookup_backend(compiler_fn):
    """Expand backend strings to functions"""
def list_backends(exclude_tags=('debug', 'experimental')) -> list[str]:
    '''
    Return valid strings that can be passed to:

        torch.compile(..., backend="name")
    '''
@functools.cache
def _lazy_import() -> None: ...
@functools.cache
def _discover_entrypoint_backends() -> None: ...
