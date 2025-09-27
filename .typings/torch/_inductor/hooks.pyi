import contextlib
import torch
from collections.abc import Generator
from typing import Callable

INTERMEDIATE_HOOKS: list[Callable[[str, torch.Tensor], None]]

@contextlib.contextmanager
def intermediate_hook(fn) -> Generator[None]: ...
def run_intermediate_hooks(name, val) -> None: ...
