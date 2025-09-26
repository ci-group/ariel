import torch
from .module import Module
from _typeshed import Incomplete
from torch import Tensor

__all__ = ['Threshold', 'ReLU', 'RReLU', 'Hardtanh', 'ReLU6', 'Sigmoid', 'Hardsigmoid', 'Tanh', 'SiLU', 'Mish', 'Hardswish', 'ELU', 'CELU', 'SELU', 'GLU', 'GELU', 'Hardshrink', 'LeakyReLU', 'LogSigmoid', 'Softplus', 'Softshrink', 'MultiheadAttention', 'PReLU', 'Softsign', 'Tanhshrink', 'Softmin', 'Softmax', 'Softmax2d', 'LogSoftmax']

class Threshold(Module):
    """Thresholds each element of the input Tensor.

    Threshold is defined as:

    .. math::
        y =
        \\begin{cases}
        x, &\\text{ if } x > \\text{threshold} \\\\\n        \\text{value}, &\\text{ otherwise }
        \\end{cases}

    Args:
        threshold: The value to threshold at
        value: The value to replace with
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Threshold.png

    Examples::

        >>> m = nn.Threshold(0, 0.5)
        >>> input = torch.arange(-3, 3)
        >>> output = m(input)
    """
    __constants__: Incomplete
    threshold: float
    value: float
    inplace: bool
    def __init__(self, threshold: float, value: float, inplace: bool = False) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self): ...

class ReLU(Module):
    """Applies the rectified linear unit function element-wise.

    :math:`\\text{ReLU}(x) = (x)^+ = \\max(0, x)`

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/ReLU.png

    Examples::

        >>> m = nn.ReLU()
        >>> input = torch.randn(2)
        >>> output = m(input)


      An implementation of CReLU - https://arxiv.org/abs/1603.05201

        >>> m = nn.ReLU()
        >>> input = torch.randn(2).unsqueeze(0)
        >>> output = torch.cat((m(input), m(-input)))
    """
    __constants__: Incomplete
    inplace: bool
    def __init__(self, inplace: bool = False) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class RReLU(Module):
    """Applies the randomized leaky rectified linear unit function, element-wise.

    Method described in the paper:
    `Empirical Evaluation of Rectified Activations in Convolutional Network <https://arxiv.org/abs/1505.00853>`_.

    The function is defined as:

    .. math::
        \\text{RReLU}(x) =
        \\begin{cases}
            x & \\text{if } x \\geq 0 \\\\\n            ax & \\text{ otherwise }
        \\end{cases}

    where :math:`a` is randomly sampled from uniform distribution
    :math:`\\mathcal{U}(\\text{lower}, \\text{upper})` during training while during
    evaluation :math:`a` is fixed with :math:`a = \\frac{\\text{lower} + \\text{upper}}{2}`.

    Args:
        lower: lower bound of the uniform distribution. Default: :math:`\\frac{1}{8}`
        upper: upper bound of the uniform distribution. Default: :math:`\\frac{1}{3}`
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/RReLU.png

    Examples::

        >>> m = nn.RReLU(0.1, 0.3)
        >>> input = torch.randn(2)
        >>> output = m(input)

    """
    __constants__: Incomplete
    lower: float
    upper: float
    inplace: bool
    def __init__(self, lower: float = ..., upper: float = ..., inplace: bool = False) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self): ...

class Hardtanh(Module):
    """Applies the HardTanh function element-wise.

    HardTanh is defined as:

    .. math::
        \\text{HardTanh}(x) = \\begin{cases}
            \\text{max\\_val} & \\text{ if } x > \\text{ max\\_val } \\\\\n            \\text{min\\_val} & \\text{ if } x < \\text{ min\\_val } \\\\\n            x & \\text{ otherwise } \\\\\n        \\end{cases}

    Args:
        min_val: minimum value of the linear region range. Default: -1
        max_val: maximum value of the linear region range. Default: 1
        inplace: can optionally do the operation in-place. Default: ``False``

    Keyword arguments :attr:`min_value` and :attr:`max_value`
    have been deprecated in favor of :attr:`min_val` and :attr:`max_val`.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Hardtanh.png

    Examples::

        >>> m = nn.Hardtanh(-2, 2)
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__: Incomplete
    min_val: float
    max_val: float
    inplace: bool
    def __init__(self, min_val: float = -1.0, max_val: float = 1.0, inplace: bool = False, min_value: float | None = None, max_value: float | None = None) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class ReLU6(Hardtanh):
    """Applies the ReLU6 function element-wise.

    .. math::
        \\text{ReLU6}(x) = \\min(\\max(0,x), 6)

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/ReLU6.png

    Examples::

        >>> m = nn.ReLU6()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    def __init__(self, inplace: bool = False) -> None: ...
    def extra_repr(self) -> str: ...

class Sigmoid(Module):
    """Applies the Sigmoid function element-wise.

    .. math::
        \\text{Sigmoid}(x) = \\sigma(x) = \\frac{1}{1 + \\exp(-x)}


    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Sigmoid.png

    Examples::

        >>> m = nn.Sigmoid()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    def forward(self, input: Tensor) -> Tensor: ...

class Hardsigmoid(Module):
    """Applies the Hardsigmoid function element-wise.

    Hardsigmoid is defined as:

    .. math::
        \\text{Hardsigmoid}(x) = \\begin{cases}
            0 & \\text{if~} x \\le -3, \\\\\n            1 & \\text{if~} x \\ge +3, \\\\\n            x / 6 + 1 / 2 & \\text{otherwise}
        \\end{cases}

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Hardsigmoid.png

    Examples::

        >>> m = nn.Hardsigmoid()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__: Incomplete
    inplace: bool
    def __init__(self, inplace: bool = False) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class Tanh(Module):
    """Applies the Hyperbolic Tangent (Tanh) function element-wise.

    Tanh is defined as:

    .. math::
        \\text{Tanh}(x) = \\tanh(x) = \\frac{\\exp(x) - \\exp(-x)} {\\exp(x) + \\exp(-x)}

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Tanh.png

    Examples::

        >>> m = nn.Tanh()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    def forward(self, input: Tensor) -> Tensor: ...

class SiLU(Module):
    """Applies the Sigmoid Linear Unit (SiLU) function, element-wise.

    The SiLU function is also known as the swish function.

    .. math::
        \\text{silu}(x) = x * \\sigma(x), \\text{where } \\sigma(x) \\text{ is the logistic sigmoid.}

    .. note::
        See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_
        where the SiLU (Sigmoid Linear Unit) was originally coined, and see
        `Sigmoid-Weighted Linear Units for Neural Network Function Approximation
        in Reinforcement Learning <https://arxiv.org/abs/1702.03118>`_ and `Swish:
        a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941v1>`_
        where the SiLU was experimented with later.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/SiLU.png

    Examples::

        >>> m = nn.SiLU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__: Incomplete
    inplace: bool
    def __init__(self, inplace: bool = False) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class Mish(Module):
    """Applies the Mish function, element-wise.

    Mish: A Self Regularized Non-Monotonic Neural Activation Function.

    .. math::
        \\text{Mish}(x) = x * \\text{Tanh}(\\text{Softplus}(x))

    .. note::
        See `Mish: A Self Regularized Non-Monotonic Neural Activation Function <https://arxiv.org/abs/1908.08681>`_

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Mish.png

    Examples::

        >>> m = nn.Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__: Incomplete
    inplace: bool
    def __init__(self, inplace: bool = False) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class Hardswish(Module):
    """Applies the Hardswish function, element-wise.

    Method described in the paper: `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`_.

    Hardswish is defined as:

    .. math::
        \\text{Hardswish}(x) = \\begin{cases}
            0 & \\text{if~} x \\le -3, \\\\\n            x & \\text{if~} x \\ge +3, \\\\\n            x \\cdot (x + 3) /6 & \\text{otherwise}
        \\end{cases}

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Hardswish.png

    Examples::

        >>> m = nn.Hardswish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__: Incomplete
    inplace: bool
    def __init__(self, inplace: bool = False) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class ELU(Module):
    """Applies the Exponential Linear Unit (ELU) function, element-wise.

    Method described in the paper: `Fast and Accurate Deep Network Learning by Exponential Linear
    Units (ELUs) <https://arxiv.org/abs/1511.07289>`__.

    ELU is defined as:

    .. math::
        \\text{ELU}(x) = \\begin{cases}
        x, & \\text{ if } x > 0\\\\\n        \\alpha * (\\exp(x) - 1), & \\text{ if } x \\leq 0
        \\end{cases}

    Args:
        alpha: the :math:`\\alpha` value for the ELU formulation. Default: 1.0
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/ELU.png

    Examples::

        >>> m = nn.ELU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__: Incomplete
    alpha: float
    inplace: bool
    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class CELU(Module):
    """Applies the CELU function element-wise.

    .. math::
        \\text{CELU}(x) = \\max(0,x) + \\min(0, \\alpha * (\\exp(x/\\alpha) - 1))

    More details can be found in the paper `Continuously Differentiable Exponential Linear Units`_ .

    Args:
        alpha: the :math:`\\alpha` value for the CELU formulation. Default: 1.0
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/CELU.png

    Examples::

        >>> m = nn.CELU()
        >>> input = torch.randn(2)
        >>> output = m(input)

    .. _`Continuously Differentiable Exponential Linear Units`:
        https://arxiv.org/abs/1704.07483
    """
    __constants__: Incomplete
    alpha: float
    inplace: bool
    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class SELU(Module):
    """Applies the SELU function element-wise.

    .. math::
        \\text{SELU}(x) = \\text{scale} * (\\max(0,x) + \\min(0, \\alpha * (\\exp(x) - 1)))

    with :math:`\\alpha = 1.6732632423543772848170429916717` and
    :math:`\\text{scale} = 1.0507009873554804934193349852946`.

    .. warning::
        When using ``kaiming_normal`` or ``kaiming_normal_`` for initialisation,
        ``nonlinearity='linear'`` should be used instead of ``nonlinearity='selu'``
        in order to get `Self-Normalizing Neural Networks`_.
        See :func:`torch.nn.init.calculate_gain` for more information.

    More details can be found in the paper `Self-Normalizing Neural Networks`_ .

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/SELU.png

    Examples::

        >>> m = nn.SELU()
        >>> input = torch.randn(2)
        >>> output = m(input)

    .. _Self-Normalizing Neural Networks: https://arxiv.org/abs/1706.02515
    """
    __constants__: Incomplete
    inplace: bool
    def __init__(self, inplace: bool = False) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class GLU(Module):
    """Applies the gated linear unit function.

    :math:`{GLU}(a, b)= a \\otimes \\sigma(b)` where :math:`a` is the first half
    of the input matrices and :math:`b` is the second half.

    Args:
        dim (int): the dimension on which to split the input. Default: -1

    Shape:
        - Input: :math:`(\\ast_1, N, \\ast_2)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(\\ast_1, M, \\ast_2)` where :math:`M=N/2`

    .. image:: ../scripts/activation_images/GLU.png

    Examples::

        >>> m = nn.GLU()
        >>> input = torch.randn(4, 2)
        >>> output = m(input)
    """
    __constants__: Incomplete
    dim: int
    def __init__(self, dim: int = -1) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class GELU(Module):
    """Applies the Gaussian Error Linear Units function.

    .. math:: \\text{GELU}(x) = x * \\Phi(x)

    where :math:`\\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.

    When the approximate argument is 'tanh', Gelu is estimated with:

    .. math:: \\text{GELU}(x) = 0.5 * x * (1 + \\text{Tanh}(\\sqrt{2 / \\pi} * (x + 0.044715 * x^3)))

    Args:
        approximate (str, optional): the gelu approximation algorithm to use:
            ``'none'`` | ``'tanh'``. Default: ``'none'``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/GELU.png

    Examples::

        >>> m = nn.GELU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__: Incomplete
    approximate: str
    def __init__(self, approximate: str = 'none') -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class Hardshrink(Module):
    """Applies the Hard Shrinkage (Hardshrink) function element-wise.

    Hardshrink is defined as:

    .. math::
        \\text{HardShrink}(x) =
        \\begin{cases}
        x, & \\text{ if } x > \\lambda \\\\\n        x, & \\text{ if } x < -\\lambda \\\\\n        0, & \\text{ otherwise }
        \\end{cases}

    Args:
        lambd: the :math:`\\lambda` value for the Hardshrink formulation. Default: 0.5

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Hardshrink.png

    Examples::

        >>> m = nn.Hardshrink()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__: Incomplete
    lambd: float
    def __init__(self, lambd: float = 0.5) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class LeakyReLU(Module):
    """Applies the LeakyReLU function element-wise.

    .. math::
        \\text{LeakyReLU}(x) = \\max(0, x) + \\text{negative\\_slope} * \\min(0, x)


    or

    .. math::
        \\text{LeakyReLU}(x) =
        \\begin{cases}
        x, & \\text{ if } x \\geq 0 \\\\\n        \\text{negative\\_slope} \\times x, & \\text{ otherwise }
        \\end{cases}

    Args:
        negative_slope: Controls the angle of the negative slope (which is used for
          negative input values). Default: 1e-2
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input

    .. image:: ../scripts/activation_images/LeakyReLU.png

    Examples::

        >>> m = nn.LeakyReLU(0.1)
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__: Incomplete
    inplace: bool
    negative_slope: float
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class LogSigmoid(Module):
    """Applies the Logsigmoid function element-wise.

    .. math::
        \\text{LogSigmoid}(x) = \\log\\left(\\frac{ 1 }{ 1 + \\exp(-x)}\\right)

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/LogSigmoid.png

    Examples::

        >>> m = nn.LogSigmoid()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    def forward(self, input: Tensor) -> Tensor: ...

class Softplus(Module):
    """Applies the Softplus function element-wise.

    .. math::
        \\text{Softplus}(x) = \\frac{1}{\\beta} * \\log(1 + \\exp(\\beta * x))

    SoftPlus is a smooth approximation to the ReLU function and can be used
    to constrain the output of a machine to always be positive.

    For numerical stability the implementation reverts to the linear function
    when :math:`input \\times \\beta > threshold`.

    Args:
        beta: the :math:`\\beta` value for the Softplus formulation. Default: 1
        threshold: values above this revert to a linear function. Default: 20

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Softplus.png

    Examples::

        >>> m = nn.Softplus()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__: Incomplete
    beta: float
    threshold: float
    def __init__(self, beta: float = 1.0, threshold: float = 20.0) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class Softshrink(Module):
    """Applies the soft shrinkage function element-wise.

    .. math::
        \\text{SoftShrinkage}(x) =
        \\begin{cases}
        x - \\lambda, & \\text{ if } x > \\lambda \\\\\n        x + \\lambda, & \\text{ if } x < -\\lambda \\\\\n        0, & \\text{ otherwise }
        \\end{cases}

    Args:
        lambd: the :math:`\\lambda` (must be no less than zero) value for the Softshrink formulation. Default: 0.5

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Softshrink.png

    Examples::

        >>> m = nn.Softshrink()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__: Incomplete
    lambd: float
    def __init__(self, lambd: float = 0.5) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class MultiheadAttention(Module):
    """Allows the model to jointly attend to information from different representation subspaces.

    This MultiheadAttention layer implements the original architecture described
    in the `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_ paper. The
    intent of this layer is as a reference implementation for foundational understanding
    and thus it contains only limited features relative to newer architectures.
    Given the fast pace of innovation in transformer-like architectures, we recommend
    exploring this `tutorial <https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html>`_
    to build efficient layers from building blocks in core or using higher
    level libraries from the `PyTorch Ecosystem <https://landscape.pytorch.org/>`_.

    Multi-Head Attention is defined as:

    .. math::
        \\text{MultiHead}(Q, K, V) = \\text{Concat}(\\text{head}_1,\\dots,\\text{head}_h)W^O

    where :math:`\\text{head}_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    ``nn.MultiheadAttention`` will use the optimized implementations of
    ``scaled_dot_product_attention()`` when possible.

    In addition to support for the new ``scaled_dot_product_attention()``
    function, for speeding up Inference, MHA will use
    fastpath inference with support for Nested Tensors, iff:

    - self attention is being computed (i.e., ``query``, ``key``, and ``value`` are the same tensor).
    - inputs are batched (3D) with ``batch_first==True``
    - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor argument ``requires_grad``
    - training is disabled (using ``.eval()``)
    - ``add_bias_kv`` is ``False``
    - ``add_zero_attn`` is ``False``
    - ``kdim`` and ``vdim`` are equal to ``embed_dim``
    - if a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ is passed, neither ``key_padding_mask``
      nor ``attn_mask`` is passed
    - autocast is disabled

    If the optimized inference fastpath implementation is in use, a
    `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be passed for
    ``query``/``key``/``value`` to represent padding more efficiently than using a
    padding mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_
    will be returned, and an additional speedup proportional to the fraction of the input
    that is padding can be expected.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Examples::

        >>> # xdoctest: +SKIP
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)

    .. _`FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`:
         https://arxiv.org/abs/2205.14135

    """
    __constants__: Incomplete
    bias_k: torch.Tensor | None
    bias_v: torch.Tensor | None
    embed_dim: Incomplete
    kdim: Incomplete
    vdim: Incomplete
    _qkv_same_embed_dim: Incomplete
    num_heads: Incomplete
    dropout: Incomplete
    batch_first: Incomplete
    head_dim: Incomplete
    q_proj_weight: Incomplete
    k_proj_weight: Incomplete
    v_proj_weight: Incomplete
    in_proj_weight: Incomplete
    in_proj_bias: Incomplete
    out_proj: Incomplete
    add_zero_attn: Incomplete
    def __init__(self, embed_dim, num_heads, dropout: float = 0.0, bias: bool = True, add_bias_kv: bool = False, add_zero_attn: bool = False, kdim=None, vdim=None, batch_first: bool = False, device=None, dtype=None) -> None: ...
    def _reset_parameters(self) -> None: ...
    def __setstate__(self, state) -> None: ...
    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Tensor | None = None, need_weights: bool = True, attn_mask: Tensor | None = None, average_attn_weights: bool = True, is_causal: bool = False) -> tuple[Tensor, Tensor | None]:
        '''Compute attention outputs using query, key, and value embeddings.

            Supports optional parameters for padding, masks and attention weights.

        Args:
            query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
                or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
                :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
                Queries are compared against key-value pairs to produce the output.
                See "Attention Is All You Need" for more details.
            key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
                or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
                :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
                See "Attention Is All You Need" for more details.
            value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
                ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
                sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
                See "Attention Is All You Need" for more details.
            key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
                to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
                Binary and float masks are supported.
                For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
                the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.
            need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
                Set ``need_weights=False`` to use the optimized ``scaled_dot_product_attention``
                and achieve the best performance for MHA.
                Default: ``True``.
            attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
                :math:`(L, S)` or :math:`(N\\cdot\\text{num\\_heads}, L, S)`, where :math:`N` is the batch size,
                :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
                broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
                Binary and float masks are supported. For a binary mask, a ``True`` value indicates that the
                corresponding position is not allowed to attend. For a float mask, the mask values will be added to
                the attention weight.
                If both attn_mask and key_padding_mask are supplied, their types should match.
            average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
                heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
                effect when ``need_weights=True``. Default: ``True`` (i.e. average weights across heads)
            is_causal: If specified, applies a causal mask as attention mask.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``attn_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Outputs:
            - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
              :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
              where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
              embedding dimension ``embed_dim``.
            - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
              returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
              :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
              :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
              head of shape :math:`(\\text{num\\_heads}, L, S)` when input is unbatched or :math:`(N, \\text{num\\_heads}, L, S)`.

            .. note::
                `batch_first` argument is ignored for unbatched inputs.
        '''
    def merge_masks(self, attn_mask: Tensor | None, key_padding_mask: Tensor | None, query: Tensor) -> tuple[Tensor | None, int | None]:
        """Determine mask type and combine masks if necessary.

        If only one mask is provided, that mask
        and the corresponding mask type will be returned. If both masks are provided, they will be both
        expanded to shape ``(batch_size, num_heads, seq_len, seq_len)``, combined with logical ``or``
        and mask type 2 will be returned
        Args:
            attn_mask: attention mask of shape ``(seq_len, seq_len)``, mask type 0
            key_padding_mask: padding mask of shape ``(batch_size, seq_len)``, mask type 1
            query: query embeddings of shape ``(batch_size, seq_len, embed_dim)``
        Returns:
            merged_mask: merged mask
            mask_type: merged mask type (0, 1, or 2)
        """

class PReLU(Module):
    """Applies the element-wise PReLU function.

    .. math::
        \\text{PReLU}(x) = \\max(0,x) + a * \\min(0,x)

    or

    .. math::
        \\text{PReLU}(x) =
        \\begin{cases}
        x, & \\text{ if } x \\ge 0 \\\\\n        ax, & \\text{ otherwise }
        \\end{cases}

    Here :math:`a` is a learnable parameter. When called without arguments, `nn.PReLU()` uses a single
    parameter :math:`a` across all input channels. If called with `nn.PReLU(nChannels)`,
    a separate :math:`a` is used for each input channel.


    .. note::
        weight decay should not be used when learning :math:`a` for good performance.

    .. note::
        Channel dim is the 2nd dim of input. When input has dims < 2, then there is
        no channel dim and the number of channels = 1.

    Args:
        num_parameters (int): number of :math:`a` to learn.
            Although it takes an int as input, there is only two values are legitimate:
            1, or the number of channels at input. Default: 1
        init (float): the initial value of :math:`a`. Default: 0.25

    Shape:
        - Input: :math:`( *)` where `*` means, any number of additional
          dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Attributes:
        weight (Tensor): the learnable weights of shape (:attr:`num_parameters`).

    .. image:: ../scripts/activation_images/PReLU.png

    Examples::

        >>> m = nn.PReLU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__: Incomplete
    num_parameters: int
    init: Incomplete
    weight: Incomplete
    def __init__(self, num_parameters: int = 1, init: float = 0.25, device=None, dtype=None) -> None: ...
    def reset_parameters(self) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class Softsign(Module):
    """Applies the element-wise Softsign function.

    .. math::
        \\text{SoftSign}(x) = \\frac{x}{ 1 + |x|}

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Softsign.png

    Examples::

        >>> m = nn.Softsign()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    def forward(self, input: Tensor) -> Tensor: ...

class Tanhshrink(Module):
    """Applies the element-wise Tanhshrink function.

    .. math::
        \\text{Tanhshrink}(x) = x - \\tanh(x)

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Tanhshrink.png

    Examples::

        >>> m = nn.Tanhshrink()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    def forward(self, input: Tensor) -> Tensor: ...

class Softmin(Module):
    """Applies the Softmin function to an n-dimensional input Tensor.

    Rescales them so that the elements of the n-dimensional output Tensor
    lie in the range `[0, 1]` and sum to 1.

    Softmin is defined as:

    .. math::
        \\text{Softmin}(x_{i}) = \\frac{\\exp(-x_i)}{\\sum_j \\exp(-x_j)}

    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input

    Args:
        dim (int): A dimension along which Softmin will be computed (so every slice
            along dim will sum to 1).

    Returns:
        a Tensor of the same dimension and shape as the input, with
        values in the range [0, 1]

    Examples::

        >>> m = nn.Softmin(dim=1)
        >>> input = torch.randn(2, 3)
        >>> output = m(input)
    """
    __constants__: Incomplete
    dim: int | None
    def __init__(self, dim: int | None = None) -> None: ...
    def __setstate__(self, state) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self): ...

class Softmax(Module):
    """Applies the Softmax function to an n-dimensional input Tensor.

    Rescales them so that the elements of the n-dimensional output Tensor
    lie in the range [0,1] and sum to 1.

    Softmax is defined as:

    .. math::
        \\text{Softmax}(x_{i}) = \\frac{\\exp(x_i)}{\\sum_j \\exp(x_j)}

    When the input Tensor is a sparse tensor then the unspecified
    values are treated as ``-inf``.

    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]

    Args:
        dim (int): A dimension along which Softmax will be computed (so every slice
            along dim will sum to 1).

    .. note::
        This module doesn't work directly with NLLLoss,
        which expects the Log to be computed between the Softmax and itself.
        Use `LogSoftmax` instead (it's faster and has better numerical properties).

    Examples::

        >>> m = nn.Softmax(dim=1)
        >>> input = torch.randn(2, 3)
        >>> output = m(input)

    """
    __constants__: Incomplete
    dim: int | None
    def __init__(self, dim: int | None = None) -> None: ...
    def __setstate__(self, state) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class Softmax2d(Module):
    """Applies SoftMax over features to each spatial location.

    When given an image of ``Channels x Height x Width``, it will
    apply `Softmax` to each location :math:`(Channels, h_i, w_j)`

    Shape:
        - Input: :math:`(N, C, H, W)` or :math:`(C, H, W)`.
        - Output: :math:`(N, C, H, W)` or :math:`(C, H, W)` (same shape as input)

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]

    Examples::

        >>> m = nn.Softmax2d()
        >>> # you softmax over the 2nd dimension
        >>> input = torch.randn(2, 3, 12, 13)
        >>> output = m(input)
    """
    def forward(self, input: Tensor) -> Tensor: ...

class LogSoftmax(Module):
    """Applies the :math:`\\log(\\text{Softmax}(x))` function to an n-dimensional input Tensor.

    The LogSoftmax formulation can be simplified as:

    .. math::
        \\text{LogSoftmax}(x_{i}) = \\log\\left(\\frac{\\exp(x_i) }{ \\sum_j \\exp(x_j)} \\right)

    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input

    Args:
        dim (int): A dimension along which LogSoftmax will be computed.

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [-inf, 0)

    Examples::

        >>> m = nn.LogSoftmax(dim=1)
        >>> input = torch.randn(2, 3)
        >>> output = m(input)
    """
    __constants__: Incomplete
    dim: int | None
    def __init__(self, dim: int | None = None) -> None: ...
    def __setstate__(self, state) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self): ...
