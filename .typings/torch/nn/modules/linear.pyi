from .lazy import LazyModuleMixin
from .module import Module
from _typeshed import Incomplete
from torch import Tensor
from torch.nn.parameter import UninitializedParameter
from typing import Any

__all__ = ['Bilinear', 'Identity', 'LazyLinear', 'Linear']

class Identity(Module):
    """A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 20])

    """
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class Linear(Module):
    """Applies an affine linear transformation to the incoming data: :math:`y = xA^T + b`.

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_\\text{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_\\text{in} = \\text{in\\_features}`.
        - Output: :math:`(*, H_\\text{out})` where all but the last dimension
          are the same shape as the input and :math:`H_\\text{out} = \\text{out\\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\\text{out\\_features}, \\text{in\\_features})`. The values are
            initialized from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`, where
            :math:`k = \\frac{1}{\\text{in\\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\\text{out\\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` where
                :math:`k = \\frac{1}{\\text{in\\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__: Incomplete
    in_features: int
    out_features: int
    weight: Tensor
    bias: Incomplete
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None: ...
    def reset_parameters(self) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class NonDynamicallyQuantizableLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None: ...

class Bilinear(Module):
    """Applies a bilinear transformation to the incoming data: :math:`y = x_1^T A x_2 + b`.

    Args:
        in1_features: size of each first input sample, must be > 0
        in2_features: size of each second input sample, must be > 0
        out_features: size of each output sample, must be > 0
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input1: :math:`(*, H_\\text{in1})` where :math:`H_\\text{in1}=\\text{in1\\_features}` and
          :math:`*` means any number of additional dimensions including none. All but the last dimension
          of the inputs should be the same.
        - Input2: :math:`(*, H_\\text{in2})` where :math:`H_\\text{in2}=\\text{in2\\_features}`.
        - Output: :math:`(*, H_\\text{out})` where :math:`H_\\text{out}=\\text{out\\_features}`
          and all but the last dimension are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\\text{out\\_features}, \\text{in1\\_features}, \\text{in2\\_features})`.
            The values are initialized from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`, where
            :math:`k = \\frac{1}{\\text{in1\\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\\text{out\\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`, where
                :math:`k = \\frac{1}{\\text{in1\\_features}}`

    Examples::

        >>> m = nn.Bilinear(20, 30, 40)
        >>> input1 = torch.randn(128, 20)
        >>> input2 = torch.randn(128, 30)
        >>> output = m(input1, input2)
        >>> print(output.size())
        torch.Size([128, 40])
    """
    __constants__: Incomplete
    in1_features: int
    in2_features: int
    out_features: int
    weight: Tensor
    bias: Incomplete
    def __init__(self, in1_features: int, in2_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None: ...
    def reset_parameters(self) -> None: ...
    def forward(self, input1: Tensor, input2: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class LazyLinear(LazyModuleMixin, Linear):
    """A :class:`torch.nn.Linear` module where `in_features` is inferred.

    In this module, the `weight` and `bias` are of :class:`torch.nn.UninitializedParameter`
    class. They will be initialized after the first call to ``forward`` is done and the
    module will become a regular :class:`torch.nn.Linear` module. The ``in_features`` argument
    of the :class:`Linear` is inferred from the ``input.shape[-1]``.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\\text{out\\_features}, \\text{in\\_features})`. The values are
            initialized from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`, where
            :math:`k = \\frac{1}{\\text{in\\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\\text{out\\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` where
                :math:`k = \\frac{1}{\\text{in\\_features}}`


    """
    cls_to_become = Linear
    weight: UninitializedParameter
    bias: UninitializedParameter
    out_features: Incomplete
    def __init__(self, out_features: int, bias: bool = True, device=None, dtype=None) -> None: ...
    def reset_parameters(self) -> None: ...
    in_features: Incomplete
    def initialize_parameters(self, input) -> None: ...
