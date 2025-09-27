import torch
from collections.abc import Sequence
from torch.nn.modules import Module
from typing import TypeVar

__all__ = ['replicate']

T = TypeVar('T', bound=Module)

def replicate(network: T, devices: Sequence[int | torch.device], detach: bool = False) -> list[T]: ...
