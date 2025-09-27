import torch
import torch.nn as nn
from .utils import ReferenceQuantizedModule
from typing import Any

__all__ = ['Linear']

class Linear(nn.Linear, ReferenceQuantizedModule):
    """A reference quantized linear module that fits into the FX
    Graph Mode Quantization workflow
    activation will be floating point Tensor, we will store floating
    point weight as well in the module, but in forward we'll quantize
    and dequantize the weight before running the floating point functional
    linear operator.
    """
    _IS_REFERENCE: bool
    def __init__(self, in_features: int, out_features: int, bias_: bool = True, device: torch.device | None = None, dtype: torch.dtype | None = None, weight_qparams: dict[str, Any] | None = None) -> None: ...
    def _get_name(self) -> str: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        we have:
        w(float) -- quant - dequant         x(float) ------------- F.linear ---

        In the full model, we will see
        w(float) -- quant - *dequant         x -- quant --- *dequant --  *F.linear --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized linear
        """
    @classmethod
    def from_float(cls, float_linear: nn.Linear, weight_qparams: dict[str, Any]) -> Linear: ...
