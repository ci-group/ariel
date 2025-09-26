import torch.ao.nn.intrinsic
import torch.ao.nn.quantized as nnq

__all__ = ['ConvReLU1d', 'ConvReLU2d', 'ConvReLU3d']

_reverse_repeat_padding = nnq.modules.conv._reverse_repeat_padding

class ConvReLU1d(nnq.Conv1d):
    """
    A ConvReLU1d module is a fused module of Conv1d and ReLU

    We adopt the same interface as :class:`torch.ao.nn.quantized.Conv1d`.

    Attributes:
        Same as torch.ao.nn.quantized.Conv1d

    """
    _FLOAT_MODULE = torch.ao.nn.intrinsic.ConvReLU1d
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None) -> None: ...
    def forward(self, input): ...
    def _get_name(self): ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False): ...
    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point): ...

class ConvReLU2d(nnq.Conv2d):
    """
    A ConvReLU2d module is a fused module of Conv2d and ReLU

    We adopt the same interface as :class:`torch.ao.nn.quantized.Conv2d`.

    Attributes:
        Same as torch.ao.nn.quantized.Conv2d

    """
    _FLOAT_MODULE = torch.ao.nn.intrinsic.ConvReLU2d
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None) -> None: ...
    def forward(self, input): ...
    def _get_name(self): ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False): ...
    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point): ...

class ConvReLU3d(nnq.Conv3d):
    """
    A ConvReLU3d module is a fused module of Conv3d and ReLU

    We adopt the same interface as :class:`torch.ao.nn.quantized.Conv3d`.

    Attributes: Same as torch.ao.nn.quantized.Conv3d

    """
    _FLOAT_MODULE = torch.ao.nn.intrinsic.ConvReLU3d
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None) -> None: ...
    def forward(self, input): ...
    def _get_name(self): ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False): ...
    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point): ...
