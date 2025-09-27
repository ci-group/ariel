import torch
from _typeshed import Incomplete
from torch import Tensor
from typing import TypeVar
from typing_extensions import ParamSpec

__all__ = ['calculate_gain', 'uniform_', 'normal_', 'trunc_normal_', 'constant_', 'ones_', 'zeros_', 'eye_', 'dirac_', 'xavier_uniform_', 'xavier_normal_', 'kaiming_uniform_', 'kaiming_normal_', 'orthogonal_', 'sparse_', 'uniform', 'normal', 'constant', 'eye', 'dirac', 'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'orthogonal', 'sparse']

_R = TypeVar('_R')
_P = ParamSpec('_P')

def calculate_gain(nonlinearity: _NonlinearityType, param: int | float | None = None) -> float:
    '''Return the recommended gain value for the given nonlinearity function.

    The values are as follows:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\\frac{5}{3}`
    ReLU              :math:`\\sqrt{2}`
    Leaky Relu        :math:`\\sqrt{\\frac{2}{1 + \\text{negative\\_slope}^2}}`
    SELU              :math:`\\frac{3}{4}`
    ================= ====================================================

    .. warning::
        In order to implement `Self-Normalizing Neural Networks`_ ,
        you should use ``nonlinearity=\'linear\'`` instead of ``nonlinearity=\'selu\'``.
        This gives the initial weights a variance of ``1 / N``,
        which is necessary to induce a stable fixed point in the forward pass.
        In contrast, the default gain for ``SELU`` sacrifices the normalization
        effect for more stable gradient flow in rectangular layers.

    Args:
        nonlinearity: the non-linear function (`nn.functional` name)
        param: optional parameter for the non-linear function

    Examples:
        >>> gain = nn.init.calculate_gain(
        ...     "leaky_relu", 0.2
        ... )  # leaky_relu with negative_slope=0.2

    .. _Self-Normalizing Neural Networks: https://papers.nips.cc/paper/2017/hash/5d44ee6f2c3f71b73125876103c8f6c4-Abstract.html
    '''
def uniform_(tensor: Tensor, a: float = 0.0, b: float = 1.0, generator: torch.Generator | None = None) -> Tensor:
    """Fill the input Tensor with values drawn from the uniform distribution.

    :math:`\\mathcal{U}(a, b)`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the lower bound of the uniform distribution
        b: the upper bound of the uniform distribution
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.uniform_(w)
    """
def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0, generator: torch.Generator | None = None) -> Tensor:
    """Fill the input Tensor with values drawn from the normal distribution.

    :math:`\\mathcal{N}(\\text{mean}, \\text{std}^2)`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.normal_(w)
    """
def trunc_normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0, generator: torch.Generator | None = None) -> Tensor:
    """Fill the input Tensor with values drawn from a truncated normal distribution.

    The values are effectively drawn from the
    normal distribution :math:`\\mathcal{N}(\\text{mean}, \\text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \\leq \\text{mean} \\leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
def constant_(tensor: Tensor, val: float) -> Tensor:
    """Fill the input Tensor with the value :math:`\\text{val}`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        val: the value to fill the tensor with

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.constant_(w, 0.3)
    """
def ones_(tensor: Tensor) -> Tensor:
    """Fill the input Tensor with the scalar value `1`.

    Args:
        tensor: an n-dimensional `torch.Tensor`

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.ones_(w)
    """
def zeros_(tensor: Tensor) -> Tensor:
    """Fill the input Tensor with the scalar value `0`.

    Args:
        tensor: an n-dimensional `torch.Tensor`

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.zeros_(w)
    """
def eye_(tensor: Tensor) -> Tensor:
    """Fill the 2-dimensional input `Tensor` with the identity matrix.

    Preserves the identity of the inputs in `Linear` layers, where as
    many inputs are preserved as possible.

    Args:
        tensor: a 2-dimensional `torch.Tensor`

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.eye_(w)
    """
def dirac_(tensor: Tensor, groups: int = 1) -> Tensor:
    """Fill the {3, 4, 5}-dimensional input `Tensor` with the Dirac delta function.

    Preserves the identity of the inputs in `Convolutional`
    layers, where as many input channels are preserved as possible. In case
    of groups>1, each group of channels preserves identity

    Args:
        tensor: a {3, 4, 5}-dimensional `torch.Tensor`
        groups (int, optional): number of groups in the conv layer (default: 1)
    Examples:
        >>> w = torch.empty(3, 16, 5, 5)
        >>> nn.init.dirac_(w)
        >>> w = torch.empty(3, 24, 5, 5)
        >>> nn.init.dirac_(w, 3)
    """
def xavier_uniform_(tensor: Tensor, gain: float = 1.0, generator: torch.Generator | None = None) -> Tensor:
    '''Fill the input `Tensor` with values using a Xavier uniform distribution.

    The method is described in `Understanding the difficulty of training
    deep feedforward neural networks` - Glorot, X. & Bengio, Y. (2010).
    The resulting tensor will have values sampled from
    :math:`\\mathcal{U}(-a, a)` where

    .. math::
        a = \\text{gain} \\times \\sqrt{\\frac{6}{\\text{fan\\_in} + \\text{fan\\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain("relu"))

    Note:
        Be aware that ``fan_in`` and ``fan_out`` are calculated assuming
        that the weight matrix is used in a transposed manner,
        (i.e., ``x @ w.T`` in ``Linear`` layers, where ``w.shape = [fan_out, fan_in]``).
        This is important for correct initialization.
        If you plan to use ``x @ w``, where ``w.shape = [fan_in, fan_out]``,
        pass in a transposed weight matrix, i.e. ``nn.init.xavier_uniform_(w.T, ...)``.
    '''
def xavier_normal_(tensor: Tensor, gain: float = 1.0, generator: torch.Generator | None = None) -> Tensor:
    """Fill the input `Tensor` with values using a Xavier normal distribution.

    The method is described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010). The resulting tensor
    will have values sampled from :math:`\\mathcal{N}(0, \\text{std}^2)` where

    .. math::
        \\text{std} = \\text{gain} \\times \\sqrt{\\frac{2}{\\text{fan\\_in} + \\text{fan\\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_normal_(w)

    Note:
        Be aware that ``fan_in`` and ``fan_out`` are calculated assuming
        that the weight matrix is used in a transposed manner,
        (i.e., ``x @ w.T`` in ``Linear`` layers, where ``w.shape = [fan_out, fan_in]``).
        This is important for correct initialization.
        If you plan to use ``x @ w``, where ``w.shape = [fan_in, fan_out]``,
        pass in a transposed weight matrix, i.e. ``nn.init.xavier_normal_(w.T, ...)``.
    """
def kaiming_uniform_(tensor: Tensor, a: float = 0, mode: _FanMode = 'fan_in', nonlinearity: _NonlinearityType = 'leaky_relu', generator: torch.Generator | None = None) -> Tensor:
    '''Fill the input `Tensor` with values using a Kaiming uniform distribution.

    The method is described in `Delving deep into rectifiers: Surpassing
    human-level performance on ImageNet classification` - He, K. et al. (2015).
    The resulting tensor will have values sampled from
    :math:`\\mathcal{U}(-\\text{bound}, \\text{bound})` where

    .. math::
        \\text{bound} = \\text{gain} \\times \\sqrt{\\frac{3}{\\text{fan\\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (only
            used with ``\'leaky_relu\'``)
        mode: either ``\'fan_in\'`` (default) or ``\'fan_out\'``. Choosing ``\'fan_in\'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``\'fan_out\'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``\'relu\'`` or ``\'leaky_relu\'`` (default).
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode="fan_in", nonlinearity="relu")

    Note:
        Be aware that ``fan_in`` and ``fan_out`` are calculated assuming
        that the weight matrix is used in a transposed manner,
        (i.e., ``x @ w.T`` in ``Linear`` layers, where ``w.shape = [fan_out, fan_in]``).
        This is important for correct initialization.
        If you plan to use ``x @ w``, where ``w.shape = [fan_in, fan_out]``,
        pass in a transposed weight matrix, i.e. ``nn.init.kaiming_uniform_(w.T, ...)``.
    '''
def kaiming_normal_(tensor: Tensor, a: float = 0, mode: _FanMode = 'fan_in', nonlinearity: _NonlinearityType = 'leaky_relu', generator: torch.Generator | None = None) -> Tensor:
    '''Fill the input `Tensor` with values using a Kaiming normal distribution.

    The method is described in `Delving deep into rectifiers: Surpassing
    human-level performance on ImageNet classification` - He, K. et al. (2015).
    The resulting tensor will have values sampled from
    :math:`\\mathcal{N}(0, \\text{std}^2)` where

    .. math::
        \\text{std} = \\frac{\\text{gain}}{\\sqrt{\\text{fan\\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (only
            used with ``\'leaky_relu\'``)
        mode: either ``\'fan_in\'`` (default) or ``\'fan_out\'``. Choosing ``\'fan_in\'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``\'fan_out\'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``\'relu\'`` or ``\'leaky_relu\'`` (default).
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_normal_(w, mode="fan_out", nonlinearity="relu")

    Note:
        Be aware that ``fan_in`` and ``fan_out`` are calculated assuming
        that the weight matrix is used in a transposed manner,
        (i.e., ``x @ w.T`` in ``Linear`` layers, where ``w.shape = [fan_out, fan_in]``).
        This is important for correct initialization.
        If you plan to use ``x @ w``, where ``w.shape = [fan_in, fan_out]``,
        pass in a transposed weight matrix, i.e. ``nn.init.kaiming_normal_(w.T, ...)``.
    '''
def orthogonal_(tensor: Tensor, gain: float = 1, generator: torch.Generator | None = None) -> Tensor:
    """Fill the input `Tensor` with a (semi) orthogonal matrix.

    Described in `Exact solutions to the nonlinear dynamics of learning in deep
    linear neural networks` - Saxe, A. et al. (2013). The input tensor must have
    at least 2 dimensions, and for tensors with more than 2 dimensions the
    trailing dimensions are flattened.

    Args:
        tensor: an n-dimensional `torch.Tensor`, where :math:`n \\geq 2`
        gain: optional scaling factor
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_LAPACK)
        >>> w = torch.empty(3, 5)
        >>> nn.init.orthogonal_(w)
    """
def sparse_(tensor: Tensor, sparsity: float, std: float = 0.01, generator: torch.Generator | None = None) -> Tensor:
    """Fill the 2D input `Tensor` as a sparse matrix.

    The non-zero elements will be drawn from the normal distribution
    :math:`\\mathcal{N}(0, 0.01)`, as described in `Deep learning via
    Hessian-free optimization` - Martens, J. (2010).

    Args:
        tensor: an n-dimensional `torch.Tensor`
        sparsity: The fraction of elements in each column to be set to zero
        std: the standard deviation of the normal distribution used to generate
            the non-zero values
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.sparse_(w, sparsity=0.1)
    """

uniform: Incomplete
normal: Incomplete
constant: Incomplete
eye: Incomplete
dirac: Incomplete
xavier_uniform: Incomplete
xavier_normal: Incomplete
kaiming_uniform: Incomplete
kaiming_normal: Incomplete
orthogonal: Incomplete
sparse: Incomplete
