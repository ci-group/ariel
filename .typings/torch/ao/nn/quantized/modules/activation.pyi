import torch
from _typeshed import Incomplete

__all__ = ['ReLU6', 'Hardswish', 'ELU', 'LeakyReLU', 'Sigmoid', 'Softmax', 'MultiheadAttention', 'PReLU']

class ReLU6(torch.nn.ReLU):
    """Applies the element-wise function:

    :math:`\\text{ReLU6}(x) = \\min(\\max(x_0, x), q(6))`, where :math:`x_0` is the
    zero_point, and :math:`q(6)` is the quantized representation of number 6.

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: ../scripts/activation_images/ReLU6.png

    Examples::

        >>> m = nn.quantized.ReLU6()
        >>> input = torch.randn(2)
        >>> # xdoctest: +SKIP
        >>> input = torch.quantize_per_tensor(input, 1.0, 0, dtype=torch.qint32)
        >>> output = m(input)
    """
    inplace: Incomplete
    def __init__(self, inplace: bool = False) -> None: ...
    def forward(self, input): ...
    def _get_name(self): ...
    @staticmethod
    def from_float(mod, use_precomputed_fake_quant: bool = False): ...

class Hardswish(torch.nn.Hardswish):
    """This is the quantized version of :class:`~torch.nn.Hardswish`.

    Args:
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
    """
    def __init__(self, scale, zero_point, device=None, dtype=None) -> None: ...
    def forward(self, input): ...
    def _get_name(self): ...
    @staticmethod
    def from_float(mod, use_precomputed_fake_quant: bool = False): ...
    @classmethod
    def from_reference(cls, mod, scale, zero_point): ...

class ELU(torch.nn.ELU):
    """This is the quantized equivalent of :class:`~torch.nn.ELU`.

    Args:
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
        alpha: the alpha constant
    """
    scale: Incomplete
    zero_point: Incomplete
    def __init__(self, scale, zero_point, alpha: float = 1.0) -> None: ...
    def forward(self, input): ...
    def _get_name(self): ...
    @staticmethod
    def from_float(mod, use_precomputed_fake_quant: bool = False): ...
    @classmethod
    def from_reference(cls, mod, scale, zero_point): ...

class LeakyReLU(torch.nn.LeakyReLU):
    """This is the quantized equivalent of :class:`~torch.nn.LeakyReLU`.

    Args:
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
        negative_slope: Controls the angle of the negative slope. Default: 1e-2
    """
    def __init__(self, scale: float, zero_point: int, negative_slope: float = 0.01, inplace: bool = False, device=None, dtype=None) -> None: ...
    def forward(self, input): ...
    def _get_name(self): ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False): ...
    @classmethod
    def from_reference(cls, mod, scale, zero_point): ...

class Sigmoid(torch.nn.Sigmoid):
    """This is the quantized equivalent of :class:`~torch.nn.Sigmoid`.

    Args:
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
    """
    output_scale: Incomplete
    output_zero_point: Incomplete
    def __init__(self, output_scale: float, output_zero_point: int) -> None: ...
    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False): ...

class Softmax(torch.nn.Softmax):
    """This is the quantized version of :class:`~torch.nn.Softmax`.

    Args:
        dim: A dimension along which Softmax will be computed (so every slice along dim will sum to 1).
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
    """
    dim: Incomplete
    scale: Incomplete
    zero_point: Incomplete
    def __init__(self, dim=None, scale: float = 1.0, zero_point: int = 0) -> None: ...
    def forward(self, input): ...
    def _get_name(self): ...
    @staticmethod
    def from_float(mod, use_precomputed_fake_quant: bool = False): ...
    @classmethod
    def from_reference(cls, mod, scale, zero_point): ...

class MultiheadAttention(torch.ao.nn.quantizable.MultiheadAttention):
    _FLOAT_MODULE = torch.ao.nn.quantizable.MultiheadAttention
    def _get_name(self): ...
    @classmethod
    def from_float(cls, other) -> None: ...
    @classmethod
    def from_observed(cls, other): ...

class PReLU(torch.nn.Module):
    """This is the quantized equivalent of :class:`~torch.nn.PReLU`.

    Args:
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
        num_parameters: number of parameters: 1, or the number of channels at input. Default: 1
    """
    num_parameters: Incomplete
    scale: Incomplete
    zero_point: Incomplete
    def __init__(self, output_scale: float, output_zero_point: int, num_parameters: int = 1) -> None: ...
    weight: Incomplete
    def set_weight(self, w: torch.Tensor) -> None: ...
    def forward(self, input: torch.Tensor) -> torch.Tensor: ...
    def _get_name(self): ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False): ...
    @classmethod
    def from_reference(cls, mod, scale, zero_point): ...
