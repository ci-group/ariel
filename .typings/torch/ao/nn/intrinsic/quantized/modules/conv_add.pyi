import torch.ao.nn.intrinsic
import torch.ao.nn.quantized as nnq

_reverse_repeat_padding = nnq.modules.conv._reverse_repeat_padding

class ConvAdd2d(nnq.Conv2d):
    """
    A ConvAdd2d module is a fused module of Conv2d and Add

    We adopt the same interface as :class:`torch.ao.nn.quantized.Conv2d`.

    Attributes:
        Same as torch.ao.nn.quantized.Conv2d

    """
    _FLOAT_MODULE = torch.ao.nn.intrinsic.ConvAdd2d
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None) -> None: ...
    def forward(self, input, extra_input): ...
    def _get_name(self): ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False): ...
    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point): ...

class ConvAddReLU2d(nnq.Conv2d):
    """
    A ConvAddReLU2d module is a fused module of Conv2d, Add and Relu

    We adopt the same interface as :class:`torch.ao.nn.quantized.Conv2d`.

    Attributes:
        Same as torch.ao.nn.quantized.Conv2d

    """
    _FLOAT_MODULE = torch.ao.nn.intrinsic.ConvAddReLU2d
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None) -> None: ...
    def forward(self, input, extra_input): ...
    def _get_name(self): ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False): ...
    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point): ...
