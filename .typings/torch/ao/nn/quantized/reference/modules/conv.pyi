import torch
import torch.nn as nn
from .utils import ReferenceQuantizedModule
from _typeshed import Incomplete
from torch.nn.common_types import _size_1_t
from typing import Any

__all__ = ['Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d']

class _ConvNd(torch.nn.modules.conv._ConvNd, ReferenceQuantizedModule):
    """A reference version of nn.quantized.Conv2d
    we will not pack the parameters in this module, since weight packing is an
    optimization for quantized backends supported in PyTorch (fbgemm/qnnpack),
    this is useful when user want to use this module in other backends like Glow.
    """
    __annotations__: Incomplete
    _IS_REFERENCE: bool
    @staticmethod
    def from_float(cls, float_conv, weight_qparams): ...

class Conv1d(_ConvNd, nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_1_t, stride: _size_1_t = 1, padding: _size_1_t = 0, dilation: _size_1_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None, weight_qparams: dict[str, Any] | None = None) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        we have:
        w(float) -- quant - dequant         x(float) ------------- F.conv1d ---

        In the full model, we will see
        w(float) -- quant - *dequant         x -- quant --- *dequant --  *F.conv1d --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized conv1d
        """
    def _get_name(self): ...
    @classmethod
    def from_float(cls, float_conv, weight_qparams): ...

class Conv2d(_ConvNd, nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None, weight_qparams: dict[str, Any] | None = None) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        we have:
        w(float) -- quant - dequant         x(float) ------------- F.conv2d ---

        In the full model, we will see
        w(float) -- quant - *dequant         x -- quant --- *dequant --  *F.conv2d --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized conv2d
        """
    def _get_name(self): ...
    @classmethod
    def from_float(cls, float_conv, weight_qparams): ...

class Conv3d(_ConvNd, nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None, weight_qparams: dict[str, Any] | None = None) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        we have:
        w(float) -- quant - dequant         x(float) ------------- F.conv3d ---

        In the full model, we will see
        w(float) -- quant - *dequant         x -- quant --- *dequant --  *F.conv3d --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized conv3d
        """
    def _get_name(self): ...
    @classmethod
    def from_float(cls, float_conv, weight_qparams): ...

class _ConvTransposeNd(_ConvNd, torch.nn.modules.conv._ConvTransposeNd):
    """A reference version of nn.quantized.ConvTranspose2d
    we will not pack the parameters in this module, since weight packing is an
    optimization for quantized backends supported in PyTorch (fbgemm/qnnpack),
    this is useful when user want to use this module in other backends like Glow.
    """
    @staticmethod
    def from_float(cls, float_conv, weight_qparams): ...

class ConvTranspose1d(_ConvTransposeNd, nn.ConvTranspose1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_1_t, stride: _size_1_t = 1, padding: _size_1_t = 0, output_padding: _size_1_t = 0, groups: int = 1, bias: bool = True, dilation: _size_1_t = 1, padding_mode: str = 'zeros', device=None, dtype=None, weight_qparams: dict[str, Any] | None = None) -> None: ...
    def forward(self, x: torch.Tensor, output_size: list[int] | None = None) -> torch.Tensor:
        """
        we have:
        w(float) -- quant - dequant         x(float) ------------- F.convTranspose1d ---
        In the full model, we will see
        w(float) -- quant - *dequant         x -- quant --- *dequant --  *F.convTranspose1d --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized conv1d
        """
    def _get_name(self): ...
    @classmethod
    def from_float(cls, float_conv, weight_qparams): ...

class ConvTranspose2d(_ConvTransposeNd, nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = True, dilation: int = 1, padding_mode: str = 'zeros', device=None, dtype=None, weight_qparams: dict[str, Any] | None = None) -> None: ...
    def forward(self, x: torch.Tensor, output_size: list[int] | None = None) -> torch.Tensor:
        """
        we have:
        w(float) -- quant - dequant         x(float) ------------- F.convTranspose2d ---
        In the full model, we will see
        w(float) -- quant - *dequant         x -- quant --- *dequant --  *F.convTranspose2d --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized conv2d
        """
    def _get_name(self): ...
    @classmethod
    def from_float(cls, float_conv, weight_qparams): ...

class ConvTranspose3d(_ConvTransposeNd, nn.ConvTranspose3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = True, dilation: int = 1, padding_mode: str = 'zeros', device=None, dtype=None, weight_qparams: dict[str, Any] | None = None) -> None: ...
    def forward(self, x: torch.Tensor, output_size: list[int] | None = None) -> torch.Tensor:
        """
        we have:
        w(float) -- quant - dequant         x(float) ------------- F.convTranspose3d ---
        In the full model, we will see
        w(float) -- quant - *dequant         x -- quant --- *dequant --  *F.convTranspose3d --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized conv3d
        """
    def _get_name(self): ...
    @classmethod
    def from_float(cls, float_conv, weight_qparams): ...
