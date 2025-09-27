import torch.nn as nn
from _typeshed import Incomplete
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import ClassVar

__all__ = ['Conv1d', 'Conv2d', 'Conv3d']

class _ConvNd(nn.modules.conv._ConvNd):
    _FLOAT_MODULE: ClassVar[type[nn.modules.conv._ConvNd]]
    qconfig: Incomplete
    weight_fake_quant: Incomplete
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, ...], stride: tuple[int, ...], padding: str | tuple[int, ...], dilation: tuple[int, ...], transposed: bool, output_padding: tuple[int, ...], groups: int, bias: bool, padding_mode: str, qconfig=None, device=None, dtype=None) -> None: ...
    def forward(self, input): ...
    @staticmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False):
        """Create a qat module from a float module

        Args:
           `mod`: a float module, either produced by torch.ao.quantization utilities
           or directly from user
        """
    def to_float(self):
        """This works for both single qat conv, and the qat conv - relu modules
        to convert the qat module to a floating point module
        """

class Conv1d(_ConvNd, nn.Conv1d):
    """
    A Conv1d module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as :class:`~torch.nn.Conv1d`

    Similar to :class:`~torch.nn.Conv2d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """
    _FLOAT_MODULE: ClassVar[type[nn.Conv1d]]
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv1d]]
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_1_t, stride: _size_1_t = 1, padding: str | _size_1_t = 0, dilation: _size_1_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', qconfig=None, device=None, dtype=None) -> None: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False): ...

class Conv2d(_ConvNd, nn.Conv2d):
    """
    A Conv2d module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.Conv2d`, please see
    https://pytorch.org/docs/stable/nn.html?highlight=conv2d#torch.nn.Conv2d
    for documentation.

    Similar to `torch.nn.Conv2d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """
    _FLOAT_MODULE: ClassVar[type[nn.Conv2d]]
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv2d]]
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: str | _size_2_t = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', qconfig=None, device=None, dtype=None) -> None: ...
    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False): ...

class Conv3d(_ConvNd, nn.Conv3d):
    """
    A Conv3d module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.Conv3d`, please see
    https://pytorch.org/docs/stable/nn.html?highlight=conv3d#torch.nn.Conv3d
    for documentation.

    Similar to `torch.nn.Conv3d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """
    _FLOAT_MODULE: ClassVar[type[nn.Conv3d]]
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv3d]]
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_3_t, stride: _size_3_t = 1, padding: str | _size_3_t = 0, dilation: _size_3_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', qconfig=None, device=None, dtype=None) -> None: ...
    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False): ...
