import torch.ao.nn.intrinsic as nni
import torch.ao.nn.qat as nnqat

class LinearReLU(nnqat.Linear, nni._FusedModule):
    """
    A LinearReLU module fused from Linear and ReLU modules, attached with
    FakeQuantize modules for weight, used in
    quantization aware training.

    We adopt the same interface as :class:`torch.nn.Linear`.

    Similar to `torch.ao.nn.intrinsic.LinearReLU`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight: fake quant module for weight

    Examples::

        >>> # xdoctest: +SKIP
        >>> m = nn.qat.LinearReLU(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    _FLOAT_MODULE = nni.LinearReLU
    def __init__(self, in_features, out_features, bias: bool = True, qconfig=None) -> None: ...
    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False): ...
    def to_float(self): ...
