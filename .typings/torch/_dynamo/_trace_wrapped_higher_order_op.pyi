import torch
from torch._ops import HigherOrderOperator, OpOverload
from torch.overrides import TorchFunctionMode
from typing import Any

__all__ = ['trace_wrapped']

Tensor = torch.Tensor

class ModIndex(torch.autograd.Function):
    generate_vmap_rule: bool
    @staticmethod
    def forward(x: Tensor, indices: list[Tensor]) -> Tensor: ...
    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None: ...
    @staticmethod
    def backward(ctx, gradOut): ...

class TransformGetItemToIndex(TorchFunctionMode):
    def __torch_function__(self, func: OpOverload, types: tuple[torch._C._TensorMeta, ...], args: tuple[object, ...] = (), kwargs: dict[str, object] | None = None) -> object: ...

def trace_wrapped(*args: Any, **kwargs: Any) -> Any: ...

class TraceWrapped(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
