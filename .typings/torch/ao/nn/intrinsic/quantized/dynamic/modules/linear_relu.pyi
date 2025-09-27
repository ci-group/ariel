import torch
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.quantized.dynamic as nnqd

__all__ = ['LinearReLU']

class LinearReLU(nnqd.Linear):
    """
    A LinearReLU module fused from Linear and ReLU modules that can be used
    for dynamic quantization.
    Supports both, FP16 and INT8 quantization.

    We adopt the same interface as :class:`torch.ao.nn.quantized.dynamic.Linear`.

    Attributes:
        Same as torch.ao.nn.quantized.dynamic.Linear

    Examples::

        >>> # xdoctest: +SKIP
        >>> m = nn.intrinsic.quantized.dynamic.LinearReLU(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    _FLOAT_MODULE = nni.LinearReLU
    def __init__(self, in_features, out_features, bias: bool = True, dtype=...) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def _get_name(self): ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False): ...
    @classmethod
    def from_reference(cls, ref_qlinear_relu): ...
