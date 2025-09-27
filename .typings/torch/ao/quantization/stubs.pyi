import torch
from _typeshed import Incomplete
from torch import nn
from torch.ao.quantization import QConfig
from typing import Any

__all__ = ['QuantStub', 'DeQuantStub', 'QuantWrapper']

class QuantStub(nn.Module):
    """Quantize stub module, before calibration, this is same as an observer,
    it will be swapped as `nnq.Quantize` in `convert`.

    Args:
        qconfig: quantization configuration for the tensor,
            if qconfig is not provided, we will get qconfig from parent modules
    """
    qconfig: Incomplete
    def __init__(self, qconfig: QConfig | None = None) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class DeQuantStub(nn.Module):
    """Dequantize stub module, before calibration, this is same as identity,
    this will be swapped as `nnq.DeQuantize` in `convert`.

    Args:
        qconfig: quantization configuration for the tensor,
            if qconfig is not provided, we will get qconfig from parent modules
    """
    qconfig: Incomplete
    def __init__(self, qconfig: Any | None = None) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class QuantWrapper(nn.Module):
    """A wrapper class that wraps the input module, adds QuantStub and
    DeQuantStub and surround the call to module with call to quant and dequant
    modules.

    This is used by the `quantization` utility functions to add the quant and
    dequant modules, before `convert` function `QuantStub` will just be observer,
    it observes the input tensor, after `convert`, `QuantStub`
    will be swapped to `nnq.Quantize` which does actual quantization. Similarly
    for `DeQuantStub`.
    """
    quant: QuantStub
    dequant: DeQuantStub
    module: nn.Module
    def __init__(self, module: nn.Module) -> None: ...
    def forward(self, X: torch.Tensor) -> torch.Tensor: ...
