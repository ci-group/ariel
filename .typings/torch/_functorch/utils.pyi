import contextlib
from collections.abc import Generator
from torch.utils._exposed_in import exposed_in as exposed_in
from typing import Any

__all__ = ['exposed_in', 'argnums_t', 'enable_single_level_autograd_function', 'unwrap_dead_wrappers']

@contextlib.contextmanager
def enable_single_level_autograd_function() -> Generator[None, None, None]: ...
def unwrap_dead_wrappers(args: tuple[Any, ...]) -> tuple[Any, ...]: ...
argnums_t = int | tuple[int, ...]
