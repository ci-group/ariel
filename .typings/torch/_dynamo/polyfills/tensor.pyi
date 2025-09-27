import torch
from typing import Any

__all__ = ['make_subclass']

def make_subclass(cls, data: torch.Tensor, requires_grad: bool = False, **kwargs: Any) -> Any: ...
