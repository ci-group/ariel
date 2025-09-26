import torch
from _typeshed import Incomplete
from torch import Tensor as Tensor, _VF as _VF
from torch._C import _add_docstr as _add_docstr, _infer_size as _infer_size
from torch._jit_internal import BroadcastingList1 as BroadcastingList1, BroadcastingList2 as BroadcastingList2, BroadcastingList3 as BroadcastingList3, _overload as _overload, boolean_dispatch as boolean_dispatch
from torch._torch_docs import reproducibility_notes as reproducibility_notes, sparse_support_notes as sparse_support_notes, tf32_notes as tf32_notes
from torch.nn import grad as grad
from torch.nn.modules.utils import _list_with_default as _list_with_default, _pair as _pair, _single as _single, _triple as _triple
from torch.overrides import handle_torch_function as handle_torch_function, has_torch_function as has_torch_function, has_torch_function_unary as has_torch_function_unary, has_torch_function_variadic as has_torch_function_variadic
from torch.types import _dtype as DType
from typing import Callable

conv1d: Incomplete
conv2d: Incomplete
conv3d: Incomplete
conv_transpose1d: Incomplete
conv_transpose2d: Incomplete
conv_transpose3d: Incomplete
conv_tbc: Incomplete
avg_pool1d: Incomplete
avg_pool2d: Incomplete
avg_pool3d: Incomplete

def fractional_max_pool2d_with_indices(input: Tensor, kernel_size: BroadcastingList2[int], output_size: BroadcastingList2[int] | None = None, output_ratio: BroadcastingList2[float] | None = None, return_indices: bool = False, _random_samples: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """
    fractional_max_pool2d(input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None)

    Applies 2D fractional max pooling over an input signal composed of several input planes.

    Fractional MaxPooling is described in detail in the paper `Fractional MaxPooling`_ by Ben Graham

    The max-pooling operation is applied in :math:`kH \\times kW` regions by a stochastic
    step size determined by the target output size.
    The number of output features is equal to the number of input planes.

    Args:
        kernel_size: the size of the window to take a max over.
                     Can be a single number :math:`k` (for a square kernel of :math:`k \\times k`)
                     or a tuple `(kH, kW)`
        output_size: the target output size of the image of the form :math:`oH \\times oW`.
                     Can be a tuple `(oH, oW)` or a single number :math:`oH` for a square image :math:`oH \\times oH`
        output_ratio: If one wants to have an output size as a ratio of the input size, this option can be given.
                      This has to be a number or tuple in the range (0, 1)
        return_indices: if ``True``, will return the indices along with the outputs.
                        Useful to pass to :func:`~torch.nn.functional.max_unpool2d`.

    Examples::
        >>> input = torch.randn(20, 16, 50, 32)
        >>> # pool of square window of size=3, and target output size 13x12
        >>> F.fractional_max_pool2d(input, 3, output_size=(13, 12))
        >>> # pool of square window and target output size being half of input image size
        >>> F.fractional_max_pool2d(input, 3, output_ratio=(0.5, 0.5))

    .. _Fractional MaxPooling:
        http://arxiv.org/abs/1412.6071
    """
def _fractional_max_pool2d(input: Tensor, kernel_size: BroadcastingList2[int], output_size: BroadcastingList2[int] | None = None, output_ratio: BroadcastingList2[float] | None = None, return_indices: bool = False, _random_samples: Tensor | None = None) -> Tensor: ...

fractional_max_pool2d: Incomplete

def fractional_max_pool3d_with_indices(input: Tensor, kernel_size: BroadcastingList3[int], output_size: BroadcastingList3[int] | None = None, output_ratio: BroadcastingList3[float] | None = None, return_indices: bool = False, _random_samples: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """
    fractional_max_pool3d(input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None)

    Applies 3D fractional max pooling over an input signal composed of several input planes.

    Fractional MaxPooling is described in detail in the paper `Fractional MaxPooling`_ by Ben Graham

    The max-pooling operation is applied in :math:`kT \\times kH \\times kW` regions by a stochastic
    step size determined by the target output size.
    The number of output features is equal to the number of input planes.

    Args:
        kernel_size: the size of the window to take a max over.
                     Can be a single number :math:`k` (for a square kernel of :math:`k \\times k \\times k`)
                     or a tuple `(kT, kH, kW)`
        output_size: the target output size of the form :math:`oT \\times oH \\times oW`.
                     Can be a tuple `(oT, oH, oW)` or a single number :math:`oH` for a cubic output
                     :math:`oH \\times oH \\times oH`
        output_ratio: If one wants to have an output size as a ratio of the input size, this option can be given.
                      This has to be a number or tuple in the range (0, 1)
        return_indices: if ``True``, will return the indices along with the outputs.
                        Useful to pass to :func:`~torch.nn.functional.max_unpool3d`.

    Shape:
        - Input: :math:`(N, C, T_{in}, H_{in}, W_{in})` or :math:`(C, T_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, T_{out}, H_{out}, W_{out})` or :math:`(C, T_{out}, H_{out}, W_{out})`, where
          :math:`(T_{out}, H_{out}, W_{out})=\\text{output\\_size}` or
          :math:`(T_{out}, H_{out}, W_{out})=\\text{output\\_ratio} \\times (T_{in}, H_{in}, W_{in})`

    Examples::
        >>> input = torch.randn(20, 16, 50, 32, 16)
        >>> # pool of cubic window of size=3, and target output size 13x12x11
        >>> F.fractional_max_pool3d(input, 3, output_size=(13, 12, 11))
        >>> # pool of cubic window and target output size being half of input size
        >>> F.fractional_max_pool3d(input, 3, output_ratio=(0.5, 0.5, 0.5))

    .. _Fractional MaxPooling:
        http://arxiv.org/abs/1412.6071
    """
def _fractional_max_pool3d(input: Tensor, kernel_size: BroadcastingList3[int], output_size: BroadcastingList3[int] | None = None, output_ratio: BroadcastingList3[float] | None = None, return_indices: bool = False, _random_samples: Tensor | None = None) -> Tensor: ...

fractional_max_pool3d: Incomplete

def max_pool1d_with_indices(input: Tensor, kernel_size: BroadcastingList1[int], stride: BroadcastingList1[int] | None = None, padding: BroadcastingList1[int] = 0, dilation: BroadcastingList1[int] = 1, ceil_mode: bool = False, return_indices: bool = False) -> tuple[Tensor, Tensor]:
    """
    max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)

    Applies a 1D max pooling over an input signal composed of several input
    planes.

    .. note::
        The order of :attr:`ceil_mode` and :attr:`return_indices` is different from
        what seen in :class:`~torch.nn.MaxPool1d`, and will change in a future release.

    See :class:`~torch.nn.MaxPool1d` for details.

    Args:
        input: input tensor of shape :math:`(\\text{minibatch} , \\text{in\\_channels} , iW)`, minibatch dim optional.
        kernel_size: the size of the window. Can be a single number or a
            tuple `(kW,)`
        stride: the stride of the window. Can be a single number or a tuple
            `(sW,)`. Default: :attr:`kernel_size`
        padding: Implicit negative infinity padding to be added on both sides, must be >= 0 and <= kernel_size / 2.
        dilation: The stride between elements within a sliding window, must be > 0.
        ceil_mode: If ``True``, will use `ceil` instead of `floor` to compute the output shape. This
                   ensures that every element in the input tensor is covered by a sliding window.
        return_indices: If ``True``, will return the argmax along with the max values.
                        Useful for :class:`torch.nn.functional.max_unpool1d` later
    """
def _max_pool1d(input: Tensor, kernel_size: BroadcastingList1[int], stride: BroadcastingList1[int] | None = None, padding: BroadcastingList1[int] = 0, dilation: BroadcastingList1[int] = 1, ceil_mode: bool = False, return_indices: bool = False) -> Tensor: ...

max_pool1d: Incomplete

def max_pool2d_with_indices(input: Tensor, kernel_size: BroadcastingList2[int], stride: BroadcastingList2[int] | None = None, padding: BroadcastingList2[int] = 0, dilation: BroadcastingList2[int] = 1, ceil_mode: bool = False, return_indices: bool = False) -> tuple[Tensor, Tensor]:
    """
    max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)

    Applies a 2D max pooling over an input signal composed of several input
    planes.

    .. note::
        The order of :attr:`ceil_mode` and :attr:`return_indices` is different from
        what seen in :class:`~torch.nn.MaxPool2d`, and will change in a future release.

    See :class:`~torch.nn.MaxPool2d` for details.

    Args:
        input: input tensor :math:`(\\text{minibatch} , \\text{in\\_channels} , iH , iW)`, minibatch dim optional.
        kernel_size: size of the pooling region. Can be a single number or a
            tuple `(kH, kW)`
        stride: stride of the pooling operation. Can be a single number or a
            tuple `(sH, sW)`. Default: :attr:`kernel_size`
        padding: Implicit negative infinity padding to be added on both sides, must be >= 0 and <= kernel_size / 2.
        dilation: The stride between elements within a sliding window, must be > 0.
        ceil_mode: If ``True``, will use `ceil` instead of `floor` to compute the output shape. This
                   ensures that every element in the input tensor is covered by a sliding window.
        return_indices: If ``True``, will return the argmax along with the max values.
                        Useful for :class:`torch.nn.functional.max_unpool2d` later
    """
def _max_pool2d(input: Tensor, kernel_size: BroadcastingList2[int], stride: BroadcastingList2[int] | None = None, padding: BroadcastingList2[int] = 0, dilation: BroadcastingList2[int] = 1, ceil_mode: bool = False, return_indices: bool = False) -> Tensor: ...

max_pool2d: Incomplete

def max_pool3d_with_indices(input: Tensor, kernel_size: BroadcastingList3[int], stride: BroadcastingList3[int] | None = None, padding: BroadcastingList3[int] = 0, dilation: BroadcastingList3[int] = 1, ceil_mode: bool = False, return_indices: bool = False) -> tuple[Tensor, Tensor]:
    """
    max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)

    Applies a 3D max pooling over an input signal composed of several input
    planes.

    .. note::
        The order of :attr:`ceil_mode` and :attr:`return_indices` is different from
        what seen in :class:`~torch.nn.MaxPool3d`, and will change in a future release.

    See :class:`~torch.nn.MaxPool3d` for details.

    Args:
        input: input tensor :math:`(\\text{minibatch} , \\text{in\\_channels} , iD, iH , iW)`, minibatch dim optional.
        kernel_size: size of the pooling region. Can be a single number or a
                     tuple `(kT, kH, kW)`
        stride: stride of the pooling operation. Can be a single number or a
                tuple `(sT, sH, sW)`. Default: :attr:`kernel_size`
        padding: Implicit negative infinity padding to be added on both sides, must be >= 0 and <= kernel_size / 2.
        dilation: The stride between elements within a sliding window, must be > 0.
        ceil_mode: If ``True``, will use `ceil` instead of `floor` to compute the output shape. This
                   ensures that every element in the input tensor is covered by a sliding window.
        return_indices: If ``True``, will return the argmax along with the max values.
                        Useful for :class:`torch.nn.functional.max_unpool3d` later
    """
def _max_pool3d(input: Tensor, kernel_size: BroadcastingList3[int], stride: BroadcastingList3[int] | None = None, padding: BroadcastingList3[int] = 0, dilation: BroadcastingList3[int] = 1, ceil_mode: bool = False, return_indices: bool = False) -> Tensor: ...

max_pool3d: Incomplete

def _unpool_output_size(input: Tensor, kernel_size: list[int], stride: list[int], padding: list[int], output_size: list[int] | None) -> list[int]: ...
def max_unpool1d(input: Tensor, indices: Tensor, kernel_size: BroadcastingList1[int], stride: BroadcastingList1[int] | None = None, padding: BroadcastingList1[int] = 0, output_size: BroadcastingList1[int] | None = None) -> Tensor:
    """Compute a partial inverse of :class:`MaxPool1d`.

    See :class:`~torch.nn.MaxUnpool1d` for details.
    """
def max_unpool2d(input: Tensor, indices: Tensor, kernel_size: BroadcastingList2[int], stride: BroadcastingList2[int] | None = None, padding: BroadcastingList2[int] = 0, output_size: BroadcastingList2[int] | None = None) -> Tensor:
    """Compute a partial inverse of :class:`MaxPool2d`.

    See :class:`~torch.nn.MaxUnpool2d` for details.
    """
def max_unpool3d(input: Tensor, indices: Tensor, kernel_size: BroadcastingList3[int], stride: BroadcastingList3[int] | None = None, padding: BroadcastingList3[int] = 0, output_size: BroadcastingList3[int] | None = None) -> Tensor:
    """Compute a partial inverse of :class:`MaxPool3d`.

    See :class:`~torch.nn.MaxUnpool3d` for details.
    """
def lp_pool3d(input: Tensor, norm_type: int | float, kernel_size: BroadcastingList3[int], stride: BroadcastingList3[int] | None = None, ceil_mode: bool = False) -> Tensor:
    """
    Apply a 3D power-average pooling over an input signal composed of several input planes.

    If the sum of all inputs to the power of `p` is
    zero, the gradient is set to zero as well.

    See :class:`~torch.nn.LPPool3d` for details.
    """
def lp_pool2d(input: Tensor, norm_type: int | float, kernel_size: BroadcastingList2[int], stride: BroadcastingList2[int] | None = None, ceil_mode: bool = False) -> Tensor:
    """
    Apply a 2D power-average pooling over an input signal composed of several input planes.

    If the sum of all inputs to the power of `p` is
    zero, the gradient is set to zero as well.

    See :class:`~torch.nn.LPPool2d` for details.
    """
def lp_pool1d(input: Tensor, norm_type: int | float, kernel_size: int, stride: BroadcastingList1[int] | None = None, ceil_mode: bool = False) -> Tensor:
    """Apply a 1D power-average pooling over an input signal composed of several input planes.

    If the sum of all inputs to the power of `p` is
    zero, the gradient is set to zero as well.

    See :class:`~torch.nn.LPPool1d` for details.
    """
def adaptive_max_pool1d_with_indices(input: Tensor, output_size: BroadcastingList1[int], return_indices: bool = False) -> tuple[Tensor, Tensor]:
    """
    adaptive_max_pool1d(input, output_size, return_indices=False)

    Applies a 1D adaptive max pooling over an input signal composed of
    several input planes.

    See :class:`~torch.nn.AdaptiveMaxPool1d` for details and output shape.

    Args:
        output_size: the target output size (single integer)
        return_indices: whether to return pooling indices. Default: ``False``
    """
def _adaptive_max_pool1d(input: Tensor, output_size: BroadcastingList1[int], return_indices: bool = False) -> Tensor: ...

adaptive_max_pool1d: Incomplete

def adaptive_max_pool2d_with_indices(input: Tensor, output_size: BroadcastingList2[int], return_indices: bool = False) -> tuple[Tensor, Tensor]:
    """adaptive_max_pool2d(input, output_size, return_indices=False)

    Applies a 2D adaptive max pooling over an input signal composed of
    several input planes.

    See :class:`~torch.nn.AdaptiveMaxPool2d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
            double-integer tuple)
        return_indices: whether to return pooling indices. Default: ``False``
    """
def _adaptive_max_pool2d(input: Tensor, output_size: BroadcastingList2[int], return_indices: bool = False) -> Tensor: ...

adaptive_max_pool2d: Incomplete

def adaptive_max_pool3d_with_indices(input: Tensor, output_size: BroadcastingList3[int], return_indices: bool = False) -> tuple[Tensor, Tensor]:
    """
    adaptive_max_pool3d(input, output_size, return_indices=False)

    Applies a 3D adaptive max pooling over an input signal composed of
    several input planes.

    See :class:`~torch.nn.AdaptiveMaxPool3d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
            triple-integer tuple)
        return_indices: whether to return pooling indices. Default: ``False``
    """
def _adaptive_max_pool3d(input: Tensor, output_size: BroadcastingList3[int], return_indices: bool = False) -> Tensor: ...

adaptive_max_pool3d: Incomplete
adaptive_avg_pool1d: Incomplete

def adaptive_avg_pool2d(input: Tensor, output_size: BroadcastingList2[int]) -> Tensor:
    """Apply a 2D adaptive average pooling over an input signal composed of several input planes.

    See :class:`~torch.nn.AdaptiveAvgPool2d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
            double-integer tuple)
    """
def adaptive_avg_pool3d(input: Tensor, output_size: BroadcastingList3[int]) -> Tensor:
    """Apply a 3D adaptive average pooling over an input signal composed of several input planes.

    See :class:`~torch.nn.AdaptiveAvgPool3d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
            triple-integer tuple)
    """
def dropout(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:
    """During training, randomly zeroes some elements of the input tensor with probability :attr:`p`.

    Uses samples from a Bernoulli distribution.

    See :class:`~torch.nn.Dropout` for details.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
def alpha_dropout(input: Tensor, p: float = 0.5, training: bool = False, inplace: bool = False) -> Tensor:
    """Apply alpha dropout to the input.

    See :class:`~torch.nn.AlphaDropout` for details.
    """
def dropout1d(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:
    """Randomly zero out entire channels (a channel is a 1D feature map).

    For example, the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 1D tensor :math:`\\text{input}[i, j]` of the input tensor.
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    See :class:`~torch.nn.Dropout1d` for details.

    Args:
        p: probability of a channel to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
def dropout2d(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:
    """Randomly zero out entire channels (a channel is a 2D feature map).

    For example, the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 2D tensor :math:`\\text{input}[i, j]` of the input tensor.
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    See :class:`~torch.nn.Dropout2d` for details.

    Args:
        p: probability of a channel to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
def dropout3d(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:
    """Randomly zero out entire channels (a channel is a 3D feature map).

    For example, the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 3D tensor :math:`\\text{input}[i, j]` of the input tensor.
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    See :class:`~torch.nn.Dropout3d` for details.

    Args:
        p: probability of a channel to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
def feature_alpha_dropout(input: Tensor, p: float = 0.5, training: bool = False, inplace: bool = False) -> Tensor:
    """Randomly masks out entire channels (a channel is a feature map).

    For example, the :math:`j`-th channel of the :math:`i`-th sample in the batch input
    is a tensor :math:`\\text{input}[i, j]` of the input tensor. Instead of
    setting activations to zero, as in regular Dropout, the activations are set
    to the negative saturation value of the SELU activation function.

    Each element will be masked independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.
    The elements to be masked are randomized on every forward call, and scaled
    and shifted to maintain zero mean and unit variance.

    See :class:`~torch.nn.FeatureAlphaDropout` for details.

    Args:
        p: dropout probability of a channel to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
def _threshold(input: Tensor, threshold: float, value: float, inplace: bool = False) -> Tensor:
    """Apply a threshold to each element of the input Tensor.

    See :class:`~torch.nn.Threshold` for more details.
    """
threshold = _threshold
threshold_: Incomplete

def relu(input: Tensor, inplace: bool = False) -> Tensor:
    """relu(input, inplace=False) -> Tensor

    Applies the rectified linear unit function element-wise. See
    :class:`~torch.nn.ReLU` for more details.
    """

relu_: Incomplete

def glu(input: Tensor, dim: int = -1) -> Tensor:
    """
    glu(input, dim=-1) -> Tensor

    The gated linear unit. Computes:

    .. math ::
        \\text{GLU}(a, b) = a \\otimes \\sigma(b)

    where `input` is split in half along `dim` to form `a` and `b`, :math:`\\sigma`
    is the sigmoid function and :math:`\\otimes` is the element-wise product between matrices.

    See `Language Modeling with Gated Convolutional Networks <https://arxiv.org/abs/1612.08083>`_.

    Args:
        input (Tensor): input tensor
        dim (int): dimension on which to split the input. Default: -1
    """
def hardtanh(input: Tensor, min_val: float = -1.0, max_val: float = 1.0, inplace: bool = False) -> Tensor:
    """
    hardtanh(input, min_val=-1., max_val=1., inplace=False) -> Tensor

    Applies the HardTanh function element-wise. See :class:`~torch.nn.Hardtanh` for more
    details.
    """

hardtanh_: Incomplete

def relu6(input: Tensor, inplace: bool = False) -> Tensor:
    """relu6(input, inplace=False) -> Tensor

    Applies the element-wise function :math:`\\text{ReLU6}(x) = \\min(\\max(0,x), 6)`.

    See :class:`~torch.nn.ReLU6` for more details.
    """
def elu(input: Tensor, alpha: float = 1.0, inplace: bool = False) -> Tensor:
    """Apply the Exponential Linear Unit (ELU) function element-wise.

    See :class:`~torch.nn.ELU` for more details.
    """

elu_: Incomplete

def selu(input: Tensor, inplace: bool = False) -> Tensor:
    """selu(input, inplace=False) -> Tensor

    Applies element-wise,
    :math:`\\text{SELU}(x) = scale * (\\max(0,x) + \\min(0, \\alpha * (\\exp(x) - 1)))`,
    with :math:`\\alpha=1.6732632423543772848170429916717` and
    :math:`scale=1.0507009873554804934193349852946`.

    See :class:`~torch.nn.SELU` for more details.
    """

selu_: Incomplete

def celu(input: Tensor, alpha: float = 1.0, inplace: bool = False) -> Tensor:
    """celu(input, alpha=1., inplace=False) -> Tensor

    Applies element-wise,
    :math:`\\text{CELU}(x) = \\max(0,x) + \\min(0, \\alpha * (\\exp(x/\\alpha) - 1))`.

    See :class:`~torch.nn.CELU` for more details.
    """

celu_: Incomplete

def leaky_relu(input: Tensor, negative_slope: float = 0.01, inplace: bool = False) -> Tensor:
    """
    leaky_relu(input, negative_slope=0.01, inplace=False) -> Tensor

    Applies element-wise,
    :math:`\\text{LeakyReLU}(x) = \\max(0, x) + \\text{negative\\_slope} * \\min(0, x)`

    See :class:`~torch.nn.LeakyReLU` for more details.
    """

leaky_relu_: Incomplete
prelu: Incomplete

def rrelu(input: Tensor, lower: float = ..., upper: float = ..., training: bool = False, inplace: bool = False) -> Tensor:
    """rrelu(input, lower=1./8, upper=1./3, training=False, inplace=False) -> Tensor

    Randomized leaky ReLU.

    See :class:`~torch.nn.RReLU` for more details.
    """

rrelu_: Incomplete
logsigmoid: Incomplete
gelu: Incomplete
hardshrink: Incomplete

def tanhshrink(input):
    """tanhshrink(input) -> Tensor

    Applies element-wise, :math:`\\text{Tanhshrink}(x) = x - \\text{Tanh}(x)`

    See :class:`~torch.nn.Tanhshrink` for more details.
    """
def softsign(input):
    """softsign(input) -> Tensor

    Applies element-wise, the function :math:`\\text{SoftSign}(x) = \\frac{x}{1 + |x|}`

    See :class:`~torch.nn.Softsign` for more details.
    """

softplus: Incomplete

def _get_softmax_dim(name: str, ndim: int, stacklevel: int) -> int: ...
def softmin(input: Tensor, dim: int | None = None, _stacklevel: int = 3, dtype: DType | None = None) -> Tensor:
    """Apply a softmin function.

    Note that :math:`\\text{Softmin}(x) = \\text{Softmax}(-x)`. See softmax definition for mathematical formula.

    See :class:`~torch.nn.Softmin` for more details.

    Args:
        input (Tensor): input
        dim (int): A dimension along which softmin will be computed (so every slice
            along dim will sum to 1).
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
          If specified, the input tensor is casted to :attr:`dtype` before the operation
          is performed. This is useful for preventing data type overflows. Default: None.
    """
def softmax(input: Tensor, dim: int | None = None, _stacklevel: int = 3, dtype: DType | None = None) -> Tensor:
    """Apply a softmax function.

    Softmax is defined as:

    :math:`\\text{Softmax}(x_{i}) = \\frac{\\exp(x_i)}{\\sum_j \\exp(x_j)}`

    It is applied to all slices along dim, and will re-scale them so that the elements
    lie in the range `[0, 1]` and sum to 1.

    See :class:`~torch.nn.Softmax` for more details.

    Args:
        input (Tensor): input
        dim (int): A dimension along which softmax will be computed.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
          If specified, the input tensor is casted to :attr:`dtype` before the operation
          is performed. This is useful for preventing data type overflows. Default: None.

    .. note::
        This function doesn't work directly with NLLLoss,
        which expects the Log to be computed between the Softmax and itself.
        Use log_softmax instead (it's faster and has better numerical properties).

    """
def gumbel_softmax(logits: Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> Tensor:
    '''
    Sample from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretize.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)

    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    .. _Link 2:
        https://arxiv.org/abs/1611.01144
    '''
def log_softmax(input: Tensor, dim: int | None = None, _stacklevel: int = 3, dtype: DType | None = None) -> Tensor:
    """Apply a softmax followed by a logarithm.

    While mathematically equivalent to log(softmax(x)), doing these two
    operations separately is slower and numerically unstable. This function
    uses an alternative formulation to compute the output and gradient correctly.

    See :class:`~torch.nn.LogSoftmax` for more details.

    Args:
        input (Tensor): input
        dim (int): A dimension along which log_softmax will be computed.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
          If specified, the input tensor is cast to :attr:`dtype` before the operation
          is performed. This is useful for preventing data type overflows. Default: None.
    """

softshrink: Incomplete

def tanh(input):
    """tanh(input) -> Tensor

    Applies element-wise,
    :math:`\\text{Tanh}(x) = \\tanh(x) = \\frac{\\exp(x) - \\exp(-x)}{\\exp(x) + \\exp(-x)}`

    See :class:`~torch.nn.Tanh` for more details.
    """
def sigmoid(input):
    """sigmoid(input) -> Tensor

    Applies the element-wise function :math:`\\text{Sigmoid}(x) = \\frac{1}{1 + \\exp(-x)}`

    See :class:`~torch.nn.Sigmoid` for more details.
    """
def hardsigmoid(input: Tensor, inplace: bool = False) -> Tensor:
    """Apply the Hardsigmoid function element-wise.

    .. math::
        \\text{Hardsigmoid}(x) = \\begin{cases}
            0 & \\text{if~} x \\le -3, \\\\\n            1 & \\text{if~} x \\ge +3, \\\\\n            x / 6 + 1 / 2 & \\text{otherwise}
        \\end{cases}

    Args:
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``

    See :class:`~torch.nn.Hardsigmoid` for more details.
    """

linear: Incomplete
bilinear: Incomplete

def silu(input: Tensor, inplace: bool = False) -> Tensor:
    """Apply the Sigmoid Linear Unit (SiLU) function, element-wise.

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

    See :class:`~torch.nn.SiLU` for more details.
    """
def mish(input: Tensor, inplace: bool = False) -> Tensor:
    """Apply the Mish function, element-wise.

    Mish: A Self Regularized Non-Monotonic Neural Activation Function.

    .. math::
        \\text{Mish}(x) = x * \\text{Tanh}(\\text{Softplus}(x))

    .. note::
        See `Mish: A Self Regularized Non-Monotonic Neural Activation Function <https://arxiv.org/abs/1908.08681>`_

    See :class:`~torch.nn.Mish` for more details.
    """
def hardswish(input: Tensor, inplace: bool = False) -> Tensor:
    """Apply hardswish function, element-wise.

    Follows implementation as described in the paper:
    `Searching for MobileNetV3`_.

    .. math::
        \\text{Hardswish}(x) = \\begin{cases}
            0 & \\text{if~} x \\le -3, \\\\\n            x & \\text{if~} x \\ge +3, \\\\\n            x \\cdot (x + 3) /6 & \\text{otherwise}
        \\end{cases}

    See :class:`~torch.nn.Hardswish` for more details.

    .. _`Searching for MobileNetV3`:
        https://arxiv.org/abs/1905.02244
    """
def _no_grad_embedding_renorm_(weight: Tensor, input: Tensor, max_norm: float, norm_type: float) -> tuple[Tensor, Tensor]: ...
def embedding(input: Tensor, weight: Tensor, padding_idx: int | None = None, max_norm: float | None = None, norm_type: float = 2.0, scale_grad_by_freq: bool = False, sparse: bool = False) -> Tensor:
    '''Generate a simple lookup table that looks up embeddings in a fixed dictionary and size.

    This module is often used to retrieve word embeddings using indices.
    The input to the module is a list of indices, and the embedding matrix,
    and the output is the corresponding word embeddings.

    See :class:`torch.nn.Embedding` for more details.

    .. note::
        Note that the analytical gradients of this function with respect to
        entries in :attr:`weight` at the row specified by :attr:`padding_idx`
        are expected to differ from the numerical ones.

    .. note::
        Note that `:class:`torch.nn.Embedding` differs from this function in
        that it initializes the row of :attr:`weight` specified by
        :attr:`padding_idx` to all zeros on construction.

    Args:
        input (LongTensor): Tensor containing indices into the embedding matrix
        weight (Tensor): The embedding matrix with number of rows equal to the maximum possible index + 1,
            and number of columns equal to the embedding size
        padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the gradient;
                                     therefore, the embedding vector at :attr:`padding_idx` is not updated during training,
                                     i.e. it remains as a fixed "pad".
        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`
                                    is renormalized to have norm :attr:`max_norm`.
                                    Note: this will modify :attr:`weight` in-place.
        norm_type (float, optional): The p of the p-norm to compute for the :attr:`max_norm` option. Default ``2``.
        scale_grad_by_freq (bool, optional): If given, this will scale gradients by the inverse of frequency of
                                                the words in the mini-batch. Default ``False``.
        sparse (bool, optional): If ``True``, gradient w.r.t. :attr:`weight` will be a sparse tensor. See Notes under
                                 :class:`torch.nn.Embedding` for more details regarding sparse gradients.

    Shape:
        - Input: LongTensor of arbitrary shape containing the indices to extract
        - Weight: Embedding matrix of floating point type with shape `(V, embedding_dim)`,
          where V = maximum index + 1 and embedding_dim = the embedding size
        - Output: `(*, embedding_dim)`, where `*` is the input shape

    Examples::

        >>> # a batch of 2 samples of 4 indices each
        >>> input = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]])
        >>> # an embedding matrix containing 10 tensors of size 3
        >>> embedding_matrix = torch.rand(10, 3)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> F.embedding(input, embedding_matrix)
        tensor([[[ 0.8490,  0.9625,  0.6753],
                 [ 0.9666,  0.7761,  0.6108],
                 [ 0.6246,  0.9751,  0.3618],
                 [ 0.4161,  0.2419,  0.7383]],

                [[ 0.6246,  0.9751,  0.3618],
                 [ 0.0237,  0.7794,  0.0528],
                 [ 0.9666,  0.7761,  0.6108],
                 [ 0.3385,  0.8612,  0.1867]]])

        >>> # example with padding_idx
        >>> weights = torch.rand(10, 3)
        >>> weights[0, :].zero_()
        >>> embedding_matrix = weights
        >>> input = torch.tensor([[0, 2, 0, 5]])
        >>> F.embedding(input, embedding_matrix, padding_idx=0)
        tensor([[[ 0.0000,  0.0000,  0.0000],
                 [ 0.5609,  0.5384,  0.8720],
                 [ 0.0000,  0.0000,  0.0000],
                 [ 0.6262,  0.2438,  0.7471]]])
    '''
def embedding_bag(input: Tensor, weight: Tensor, offsets: Tensor | None = None, max_norm: float | None = None, norm_type: float = 2, scale_grad_by_freq: bool = False, mode: str = 'mean', sparse: bool = False, per_sample_weights: Tensor | None = None, include_last_offset: bool = False, padding_idx: int | None = None) -> Tensor:
    '''Compute sums, means or maxes of `bags` of embeddings.

    Calculation is done without instantiating the intermediate embeddings.
    See :class:`torch.nn.EmbeddingBag` for more details.

    Note:
        {backward_reproducibility_note}

    Args:
        input (LongTensor): Tensor containing bags of indices into the embedding matrix
        weight (Tensor): The embedding matrix with number of rows equal to the maximum possible index + 1,
            and number of columns equal to the embedding size
        offsets (LongTensor, optional): Only used when :attr:`input` is 1D. :attr:`offsets` determines
                             the starting index position of each bag (sequence) in :attr:`input`.
        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`
                                    is renormalized to have norm :attr:`max_norm`.
                                    Note: this will modify :attr:`weight` in-place.
        norm_type (float, optional): The ``p`` in the ``p``-norm to compute for the :attr:`max_norm` option.
                                     Default ``2``.
        scale_grad_by_freq (bool, optional): if given, this will scale gradients by the inverse of frequency of
                                                the words in the mini-batch. Default ``False``.
                                                Note: this option is not supported when ``mode="max"``.
        mode (str, optional): ``"sum"``, ``"mean"`` or ``"max"``. Specifies the way to reduce the bag.
                                 Default: ``"mean"``
        sparse (bool, optional): if ``True``, gradient w.r.t. :attr:`weight` will be a sparse tensor. See Notes under
                                 :class:`torch.nn.Embedding` for more details regarding sparse gradients.
                                 Note: this option is not supported when ``mode="max"``.
        per_sample_weights (Tensor, optional): a tensor of float / double weights, or None
            to indicate all weights should be taken to be 1. If specified, :attr:`per_sample_weights`
            must have exactly the same shape as input and is treated as having the same
            :attr:`offsets`, if those are not None.

        include_last_offset (bool, optional): if ``True``, the size of offsets is equal to the number of bags + 1.
            The last element is the size of the input, or the ending index position of the last bag (sequence).

        padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the
                                     gradient; therefore, the embedding vector at :attr:`padding_idx` is not updated
                                     during training, i.e. it remains as a fixed "pad". Note that the embedding
                                     vector at :attr:`padding_idx` is excluded from the reduction.

    Shape:
        - :attr:`input` (LongTensor) and :attr:`offsets` (LongTensor, optional)

          - If :attr:`input` is 2D of shape `(B, N)`, it will be treated as ``B`` bags (sequences)
            each of fixed length ``N``, and this will return ``B`` values aggregated in a way
            depending on the :attr:`mode`. :attr:`offsets` is ignored and required to be ``None`` in this case.

          - If :attr:`input` is 1D of shape `(N)`, it will be treated as a concatenation of
            multiple bags (sequences). :attr:`offsets` is required to be a 1D tensor containing
            the starting index positions of each bag in :attr:`input`. Therefore, for :attr:`offsets`
            of shape `(B)`, :attr:`input` will be viewed as having ``B`` bags.
            Empty bags (i.e., having 0-length) will have returned vectors filled by zeros.

        - :attr:`weight` (Tensor): the learnable weights of the module of shape `(num_embeddings, embedding_dim)`

        - :attr:`per_sample_weights` (Tensor, optional). Has the same shape as :attr:`input`.

        - :attr:`output`: aggregated embedding values of shape `(B, embedding_dim)`

    Examples::

        >>> # an Embedding module containing 10 tensors of size 3
        >>> embedding_matrix = torch.rand(10, 3)
        >>> # a batch of 2 samples of 4 indices each
        >>> input = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9])
        >>> offsets = torch.tensor([0, 4])
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> F.embedding_bag(input, embedding_matrix, offsets)
        tensor([[ 0.3397,  0.3552,  0.5545],
                [ 0.5893,  0.4386,  0.5882]])

        >>> # example with padding_idx
        >>> embedding_matrix = torch.rand(10, 3)
        >>> input = torch.tensor([2, 2, 2, 2, 4, 3, 2, 9])
        >>> offsets = torch.tensor([0, 4])
        >>> F.embedding_bag(input, embedding_matrix, offsets, padding_idx=2, mode=\'sum\')
        tensor([[ 0.0000,  0.0000,  0.0000],
                [-0.7082,  3.2145, -2.6251]])
    '''
def _verify_batch_size(size: list[int]) -> None: ...
def batch_norm(input: Tensor, running_mean: Tensor | None, running_var: Tensor | None, weight: Tensor | None = None, bias: Tensor | None = None, training: bool = False, momentum: float = 0.1, eps: float = 1e-05) -> Tensor:
    """Apply Batch Normalization for each channel across a batch of data.

    See :class:`~torch.nn.BatchNorm1d`, :class:`~torch.nn.BatchNorm2d`,
    :class:`~torch.nn.BatchNorm3d` for details.
    """
def _verify_spatial_size(size: list[int]) -> None: ...
def instance_norm(input: Tensor, running_mean: Tensor | None = None, running_var: Tensor | None = None, weight: Tensor | None = None, bias: Tensor | None = None, use_input_stats: bool = True, momentum: float = 0.1, eps: float = 1e-05) -> Tensor:
    """Apply Instance Normalization independently for each channel in every data sample within a batch.

    See :class:`~torch.nn.InstanceNorm1d`, :class:`~torch.nn.InstanceNorm2d`,
    :class:`~torch.nn.InstanceNorm3d` for details.
    """
def layer_norm(input: Tensor, normalized_shape: list[int], weight: Tensor | None = None, bias: Tensor | None = None, eps: float = 1e-05) -> Tensor:
    """Apply Layer Normalization for last certain number of dimensions.

    See :class:`~torch.nn.LayerNorm` for details.
    """
def rms_norm(input: Tensor, normalized_shape: list[int], weight: Tensor | None = None, eps: float | None = None) -> Tensor:
    """Apply Root Mean Square Layer Normalization.

    See :class:`~torch.nn.RMSNorm` for details.
    """
def group_norm(input: Tensor, num_groups: int, weight: Tensor | None = None, bias: Tensor | None = None, eps: float = 1e-05) -> Tensor:
    """Apply Group Normalization for last certain number of dimensions.

    See :class:`~torch.nn.GroupNorm` for details.
    """
def local_response_norm(input: Tensor, size: int, alpha: float = 0.0001, beta: float = 0.75, k: float = 1.0) -> Tensor:
    """Apply local response normalization over an input signal.

    The input signal is composed of several input planes, where channels occupy the second dimension.
    Normalization is applied across channels.

    See :class:`~torch.nn.LocalResponseNorm` for details.
    """
def ctc_loss(log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor, blank: int = 0, reduction: str = 'mean', zero_infinity: bool = False) -> Tensor:
    """Compute the Connectionist Temporal Classification loss.

    See :class:`~torch.nn.CTCLoss` for details.

    Note:
        {cudnn_reproducibility_note}

    Note:
        {backward_reproducibility_note}

    Args:
        log_probs: :math:`(T, N, C)` or :math:`(T, C)` where `C = number of characters in alphabet including blank`,
            `T = input length`, and `N = batch size`.
            The logarithmized probabilities of the outputs
            (e.g. obtained with :func:`torch.nn.functional.log_softmax`).
        targets: :math:`(N, S)` or `(sum(target_lengths))`.
                May be an empty tensor if all entries in `target_lengths` are zero.
                In the second form, the targets are assumed to be concatenated.
        input_lengths: :math:`(N)` or :math:`()`.
            Lengths of the inputs (must each be :math:`\\leq T`)
        target_lengths: :math:`(N)` or :math:`()`.
            Lengths of the targets
        blank (int, optional):
            Blank label. Default :math:`0`.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the output losses will be divided by the target lengths and
            then the mean over the batch is taken, ``'sum'``: the output will be
            summed. Default: ``'mean'``
        zero_infinity (bool, optional):
            Whether to zero infinite losses and the associated gradients.
            Default: ``False``
            Infinite losses mainly occur when the inputs are too short
            to be aligned to the targets.

    Example::

        >>> log_probs = torch.randn(50, 16, 20).log_softmax(2).detach().requires_grad_()
        >>> targets = torch.randint(1, 20, (16, 30), dtype=torch.long)
        >>> input_lengths = torch.full((16,), 50, dtype=torch.long)
        >>> target_lengths = torch.randint(10, 30, (16,), dtype=torch.long)
        >>> loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        >>> loss.backward()
    """
def nll_loss(input: Tensor, target: Tensor, weight: Tensor | None = None, size_average: bool | None = None, ignore_index: int = -100, reduce: bool | None = None, reduction: str = 'mean') -> Tensor:
    """Compute the negative log likelihood loss.

    See :class:`~torch.nn.NLLLoss` for details.

    Args:
        input: :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`
            in case of 2D Loss, or :math:`(N, C, d_1, d_2, ..., d_K)` where :math:`K \\geq 1`
            in the case of K-dimensional loss. `input` is expected to be log-probabilities.
        target: :math:`(N)` where each value is :math:`0 \\leq \\text{targets}[i] \\leq C-1`,
            or :math:`(N, d_1, d_2, ..., d_K)` where :math:`K \\geq 1` for
            K-dimensional loss.
        weight (Tensor, optional): A manual rescaling weight given to each
            class. If given, has to be a Tensor of size `C`
        size_average (bool, optional): Deprecated (see :attr:`reduction`).
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets. Default: -100
        reduce (bool, optional): Deprecated (see :attr:`reduction`).
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Example::

        >>> # input is of size N x C = 3 x 5
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> # each element in target has to have 0 <= value < C
        >>> target = torch.tensor([1, 0, 4])
        >>> output = F.nll_loss(F.log_softmax(input, dim=1), target)
        >>> output.backward()
    """
def poisson_nll_loss(input: Tensor, target: Tensor, log_input: bool = True, full: bool = False, size_average: bool | None = None, eps: float = 1e-08, reduce: bool | None = None, reduction: str = 'mean') -> Tensor:
    """Compute the Poisson negative log likelihood loss.

    See :class:`~torch.nn.PoissonNLLLoss` for details.

    Args:
        input: Expectation of underlying Poisson distribution.
        target: Random sample :math:`target \\sim \\text{Poisson}(input)`.
        log_input: If ``True`` the loss is computed as
            :math:`\\exp(\\text{input}) - \\text{target} * \\text{input}`, if ``False`` then loss is
            :math:`\\text{input} - \\text{target} * \\log(\\text{input}+\\text{eps})`. Default: ``True``
        full: Whether to compute full loss, i. e. to add the Stirling
            approximation term. Default: ``False``
            :math:`\\text{target} * \\log(\\text{target}) - \\text{target} + 0.5 * \\log(2 * \\pi * \\text{target})`.
        size_average (bool, optional): Deprecated (see :attr:`reduction`).
        eps (float, optional): Small value to avoid evaluation of :math:`\\log(0)` when
            :attr:`log_input`\\ =\\ ``False``. Default: 1e-8
        reduce (bool, optional): Deprecated (see :attr:`reduction`).
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    """
def gaussian_nll_loss(input: Tensor, target: Tensor, var: Tensor | float, full: bool = False, eps: float = 1e-06, reduction: str = 'mean') -> Tensor:
    """Compute the Gaussian negative log likelihood loss.

    See :class:`~torch.nn.GaussianNLLLoss` for details.

    Args:
        input: Expectation of the Gaussian distribution.
        target: Sample from the Gaussian distribution.
        var: Tensor of positive variance(s), one for each of the expectations
            in the input (heteroscedastic), or a single one (homoscedastic),
            or a positive scalar value to be used for all expectations.
        full (bool, optional): Whether to include the constant term in the loss calculation. Default: ``False``.
        eps (float, optional): Value added to var, for stability. Default: 1e-6.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the output is the average of all batch member losses,
            ``'sum'``: the output is the sum of all batch member losses.
            Default: ``'mean'``.
    """
def kl_div(input: Tensor, target: Tensor, size_average: bool | None = None, reduce: bool | None = None, reduction: str = 'mean', log_target: bool = False) -> Tensor:
    """Compute the KL Divergence loss.

    Refer - The `Kullback-Leibler divergence Loss
    <https://en.wikipedia.org/wiki/Kullback-Leibler_divergence>`__

    See :class:`~torch.nn.KLDivLoss` for details.

    Args:
        input: Tensor of arbitrary shape in log-probabilities.
        target: Tensor of the same shape as input. See :attr:`log_target` for
            the target's interpretation.
        size_average (bool, optional): Deprecated (see :attr:`reduction`).
        reduce (bool, optional): Deprecated (see :attr:`reduction`).
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
            ``'none'``: no reduction will be applied
            ``'batchmean'``: the sum of the output will be divided by the batchsize
            ``'sum'``: the output will be summed
            ``'mean'``: the output will be divided by the number of elements in the output
            Default: ``'mean'``
        log_target (bool): A flag indicating whether ``target`` is passed in the log space.
            It is recommended to pass certain distributions (like ``softmax``)
            in the log space to avoid numerical issues caused by explicit ``log``.
            Default: ``False``

    .. note::
        :attr:`size_average` and :attr:`reduce` are in the process of being deprecated,
        and in the meantime, specifying either of those two args will override :attr:`reduction`.

    .. warning::
        :attr:`reduction` = ``'mean'`` doesn't return the true kl divergence value, please use
        :attr:`reduction` = ``'batchmean'`` which aligns with KL math definition.
    """
def cross_entropy(input: Tensor, target: Tensor, weight: Tensor | None = None, size_average: bool | None = None, ignore_index: int = -100, reduce: bool | None = None, reduction: str = 'mean', label_smoothing: float = 0.0) -> Tensor:
    """Compute the cross entropy loss between input logits and target.

    See :class:`~torch.nn.CrossEntropyLoss` for details.

    Args:
        input (Tensor) : Predicted unnormalized logits;
            see Shape section below for supported shapes.
        target (Tensor) : Ground truth class indices or class probabilities;
            see Shape section below for supported shapes.
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, has to be a Tensor of size `C`
        size_average (bool, optional): Deprecated (see :attr:`reduction`).
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets. Note that
            :attr:`ignore_index` is only applicable when the target contains class indices.
            Default: -100
        reduce (bool, optional): Deprecated (see :attr:`reduction`).
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        label_smoothing (float, optional): A float in [0.0, 1.0]. Specifies the amount
            of smoothing when computing the loss, where 0.0 means no smoothing. The targets
            become a mixture of the original ground truth and a uniform distribution as described in
            `Rethinking the Inception Architecture for Computer Vision <https://arxiv.org/abs/1512.00567>`__. Default: :math:`0.0`.

    Shape:
        - Input: Shape :math:`(C)`, :math:`(N, C)` or :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \\geq 1`
          in the case of `K`-dimensional loss.
        - Target: If containing class indices, shape :math:`()`, :math:`(N)` or :math:`(N, d_1, d_2, ..., d_K)` with
          :math:`K \\geq 1` in the case of K-dimensional loss where each value should be between :math:`[0, C)`.
          If containing class probabilities, same shape as the input and each value should be between :math:`[0, 1]`.

        where:

        .. math::
            \\begin{aligned}
                C ={} & \\text{number of classes} \\\\\n                N ={} & \\text{batch size} \\\\\n            \\end{aligned}

    Examples::

        >>> # Example of target with class indices
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randint(5, (3,), dtype=torch.int64)
        >>> loss = F.cross_entropy(input, target)
        >>> loss.backward()
        >>>
        >>> # Example of target with class probabilities
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5).softmax(dim=1)
        >>> loss = F.cross_entropy(input, target)
        >>> loss.backward()
    """
def binary_cross_entropy(input: Tensor, target: Tensor, weight: Tensor | None = None, size_average: bool | None = None, reduce: bool | None = None, reduction: str = 'mean') -> Tensor:
    """Compute Binary Cross Entropy between the target and input probabilities.

    See :class:`~torch.nn.BCELoss` for details.

    Args:
        input: Tensor of arbitrary shape as probabilities.
        target: Tensor of the same shape as input with values between 0 and 1.
        weight (Tensor, optional): a manual rescaling weight
                if provided it's repeated to match input tensor shape
        size_average (bool, optional): Deprecated (see :attr:`reduction`).
        reduce (bool, optional): Deprecated (see :attr:`reduction`).
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Examples::

        >>> input = torch.randn(3, 2, requires_grad=True)
        >>> target = torch.rand(3, 2, requires_grad=False)
        >>> loss = F.binary_cross_entropy(torch.sigmoid(input), target)
        >>> loss.backward()
    """
def binary_cross_entropy_with_logits(input: Tensor, target: Tensor, weight: Tensor | None = None, size_average: bool | None = None, reduce: bool | None = None, reduction: str = 'mean', pos_weight: Tensor | None = None) -> Tensor:
    """Compute Binary Cross Entropy between target and input logits.

    See :class:`~torch.nn.BCEWithLogitsLoss` for details.

    Args:
        input: Tensor of arbitrary shape as unnormalized scores (often referred to as logits).
        target: Tensor of the same shape as input with values between 0 and 1
        weight (Tensor, optional): a manual rescaling weight
            if provided it's repeated to match input tensor shape
        size_average (bool, optional): Deprecated (see :attr:`reduction`).
        reduce (bool, optional): Deprecated (see :attr:`reduction`).
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        pos_weight (Tensor, optional): a weight of positive examples to be broadcasted with target.
            Must be a tensor with equal size along the class dimension to the number of classes.
            Pay close attention to PyTorch's broadcasting semantics in order to achieve the desired
            operations. For a target of size [B, C, H, W] (where B is batch size) pos_weight of
            size [B, C, H, W] will apply different pos_weights to each element of the batch or
            [C, H, W] the same pos_weights across the batch. To apply the same positive weight
            along all spatial dimensions for a 2D multi-class target [C, H, W] use: [C, 1, 1].
            Default: ``None``

    Examples::

         >>> input = torch.randn(3, requires_grad=True)
         >>> target = torch.empty(3).random_(2)
         >>> loss = F.binary_cross_entropy_with_logits(input, target)
         >>> loss.backward()
    """
def smooth_l1_loss(input: Tensor, target: Tensor, size_average: bool | None = None, reduce: bool | None = None, reduction: str = 'mean', beta: float = 1.0) -> Tensor:
    """Compute the Smooth L1 loss.

    Function uses a squared term if the absolute
    element-wise error falls below beta and an L1 term otherwise.

    See :class:`~torch.nn.SmoothL1Loss` for details.

    Args:
        input (Tensor): Predicted values.
        target (Tensor): Ground truth values.
        size_average (bool, optional): Deprecated (see :attr:`reduction`).
        reduce (bool, optional): Deprecated (see :attr:`reduction`).
        reduction (str, optional): Specifies the reduction to apply to the output:
                                   'none' | 'mean' | 'sum'. 'mean': the mean of the output is taken.
                                   'sum': the output will be summed. 'none': no reduction will be applied.
                                   Default: 'mean'.
        beta (float, optional): Specifies the threshold at which to change from the squared
            term to the L1 term in the loss calculation. This value must be positive.
            Default: 1.0.

    Returns:
        Tensor: L1 loss (optionally weighted).
    """
def huber_loss(input: Tensor, target: Tensor, reduction: str = 'mean', delta: float = 1.0, weight: Tensor | None = None) -> Tensor:
    """Compute the Huber loss, with optional weighting.

    Function uses a squared term if the absolute
    element-wise error falls below delta and a delta-scaled L1 term otherwise.

    When delta equals 1, this loss is equivalent to SmoothL1Loss.
    In general, Huber loss differs from SmoothL1Loss by a factor of delta (AKA beta in Smooth L1).

    See :class:`~torch.nn.HuberLoss` for details.

    Args:
        input (Tensor): Predicted values.
        target (Tensor): Ground truth values.
        reduction (str, optional): Specifies the reduction to apply to the output:
                                   'none' | 'mean' | 'sum'. 'mean': the mean of the output is taken.
                                   'sum': the output will be summed. 'none': no reduction will be applied.
                                   Default: 'mean'.
        delta (float, optional): The threshold at which to change between delta-scaled L1 and L2 loss. Default: 1.0.
        weight (Tensor, optional): Weights for each sample. Default: None.

    Returns:
        Tensor: Huber loss (optionally weighted).
    """
def l1_loss(input: Tensor, target: Tensor, size_average: bool | None = None, reduce: bool | None = None, reduction: str = 'mean', weight: Tensor | None = None) -> Tensor:
    """Compute the L1 loss, with optional weighting.

    Function that takes the mean element-wise absolute value difference.

    See :class:`~torch.nn.L1Loss` for details.

    Args:
        input (Tensor): Predicted values.
        target (Tensor): Ground truth values.
        size_average (bool, optional): Deprecated (see :attr:`reduction`).
        reduce (bool, optional): Deprecated (see :attr:`reduction`).
        reduction (str, optional): Specifies the reduction to apply to the output:
                                   'none' | 'mean' | 'sum'. 'mean': the mean of the output is taken.
                                   'sum': the output will be summed. 'none': no reduction will be applied.
                                   Default: 'mean'.
        weight (Tensor, optional): Weights for each sample. Default: None.

    Returns:
        Tensor: L1 loss (optionally weighted).
    """
def mse_loss(input: Tensor, target: Tensor, size_average: bool | None = None, reduce: bool | None = None, reduction: str = 'mean', weight: Tensor | None = None) -> Tensor:
    """Compute the element-wise mean squared error, with optional weighting.

    See :class:`~torch.nn.MSELoss` for details.

    Args:
        input (Tensor): Predicted values.
        target (Tensor): Ground truth values.
        size_average (bool, optional): Deprecated (see :attr:`reduction`).
        reduce (bool, optional): Deprecated (see :attr:`reduction`).
        reduction (str, optional): Specifies the reduction to apply to the output:
                                   'none' | 'mean' | 'sum'. 'mean': the mean of the output is taken.
                                   'sum': the output will be summed. 'none': no reduction will be applied.
                                   Default: 'mean'.
        weight (Tensor, optional): Weights for each sample. Default: None.

    Returns:
        Tensor: Mean Squared Error loss (optionally weighted).
    """
def margin_ranking_loss(input1: Tensor, input2: Tensor, target: Tensor, margin: float = 0, size_average: bool | None = None, reduce: bool | None = None, reduction: str = 'mean') -> Tensor:
    """Compute the margin ranking loss.

    See :class:`~torch.nn.MarginRankingLoss` for details.

    Args:
        input1 (Tensor): Predicted values.
        input2 (Tensor): Predicted values.
        target (Tensor): Ground truth values.
        size_average (bool, optional): Deprecated (see :attr:`reduction`).
        reduce (bool, optional): Deprecated (see :attr:`reduction`).
        reduction (str, optional): Specifies the reduction to apply to the output:
                                   'none' | 'mean' | 'sum'. 'mean': the mean of the output is taken.
                                   'sum': the output will be summed. 'none': no reduction will be applied.
                                   Default: 'mean'.

    Returns:
        Tensor: Margin ranking loss.
    """
def hinge_embedding_loss(input: Tensor, target: Tensor, margin: float = 1.0, size_average: bool | None = None, reduce: bool | None = None, reduction: str = 'mean') -> Tensor:
    """Compute the hinge embedding loss.

    See :class:`~torch.nn.HingeEmbeddingLoss` for details.

    Args:
       input (Tensor): Predicted values.
       target (Tensor): Ground truth values.
       margin (float, optional): Margin for hinge loss. Has a default value of 1.
       size_average (bool, optional): Deprecated (see :attr:`reduction`).
       reduce (bool, optional): Deprecated (see :attr:`reduction`).
       reduction (str, optional): Specifies the reduction to apply to the output:
                                  'none' | 'mean' | 'sum'. 'mean': the mean of the output is taken.
                                  'sum': the output will be summed. 'none': no reduction will be applied.
                                  Default: 'mean'.

    Returns:
       Tensor: Hinge embedding loss.
    """
def multilabel_margin_loss(input: Tensor, target: Tensor, size_average: bool | None = None, reduce: bool | None = None, reduction: str = 'mean') -> Tensor:
    """Compute the multilabel margin loss.

    See :class:`~torch.nn.MultiLabelMarginLoss` for details.

    Args:
       input (Tensor): Predicted values.
       target (Tensor): Ground truth values.
       size_average (bool, optional): Deprecated (see :attr:`reduction`).
       reduce (bool, optional): Deprecated (see :attr:`reduction`).
       reduction (str, optional): Specifies the reduction to apply to the output:
                                  'none' | 'mean' | 'sum'. 'mean': the mean of the output is taken.
                                  'sum': the output will be summed. 'none': no reduction will be applied.
                                  Default: 'mean'.

    Returns:
       Tensor: Mutilabel margin loss.
    """
def soft_margin_loss(input: Tensor, target: Tensor, size_average: bool | None = None, reduce: bool | None = None, reduction: str = 'mean') -> Tensor:
    """Compute the soft margin loss.

    See :class:`~torch.nn.SoftMarginLoss` for details.

    Args:
       input (Tensor): Predicted values.
       target (Tensor): Ground truth values.
       size_average (bool, optional): Deprecated (see :attr:`reduction`).
       reduce (bool, optional): Deprecated (see :attr:`reduction`).
       reduction (str, optional): Specifies the reduction to apply to the output:
                                  'none' | 'mean' | 'sum'. 'mean': the mean of the output is taken.
                                  'sum': the output will be summed. 'none': no reduction will be applied.
                                  Default: 'mean'.

    Returns:
       Tensor: Soft margin loss.
    """
def multilabel_soft_margin_loss(input: Tensor, target: Tensor, weight: Tensor | None = None, size_average: bool | None = None, reduce: bool | None = None, reduction: str = 'mean') -> Tensor:
    """Compute the multilabel soft margin loss.

    See :class:`~torch.nn.MultiLabelSoftMarginLoss` for details.

    Args:
       input (Tensor): Predicted values.
       target (Tensor): Ground truth values.
       size_average (bool, optional): Deprecated (see :attr:`reduction`).
       reduce (bool, optional): Deprecated (see :attr:`reduction`).
       reduction (str, optional): Specifies the reduction to apply to the output:
                                  'none' | 'mean' | 'sum'. 'mean': the mean of the output is taken.
                                  'sum': the output will be summed. 'none': no reduction will be applied.
                                  Default: 'mean'.

    Returns:
       Tensor: Mutilabel soft margin loss.
    """
def cosine_embedding_loss(input1: Tensor, input2: Tensor, target: Tensor, margin: float = 0, size_average: bool | None = None, reduce: bool | None = None, reduction: str = 'mean') -> Tensor:
    """Compute the cosine embedding loss.

    See :class:`~torch.nn.CosineEmbeddingLoss` for details.

    Args:
       input1 (Tensor): Predicted values.
       input2 (Tensor): Predicted values.
       target (Tensor): Ground truth values.
       margin (float, optional): Margin for cosine embedding. Has a default value of 0.
       size_average (bool, optional): Deprecated (see :attr:`reduction`).
       reduce (bool, optional): Deprecated (see :attr:`reduction`).
       reduction (str, optional): Specifies the reduction to apply to the output:
                                  'none' | 'mean' | 'sum'. 'mean': the mean of the output is taken.
                                  'sum': the output will be summed. 'none': no reduction will be applied.
                                  Default: 'mean'.

    Returns:
       Tensor: Cosine embedding loss.
    """
def multi_margin_loss(input: Tensor, target: Tensor, p: int = 1, margin: float = 1.0, weight: Tensor | None = None, size_average: bool | None = None, reduce: bool | None = None, reduction: str = 'mean') -> Tensor:
    """Compute the multi margin loss, with optional weighting.

    See :class:`~torch.nn.MultiMarginLoss` for details.

    Args:
        input (Tensor): Predicted values.
        target (Tensor): Ground truth values.
        p (int, optional): Has a default value of 1. 1 and 2 are the only supported values.
        margin (float, optional): Margin for multi margin loss. Has a default value of 1.
        weight (Tensor, optional): Weights for each sample. Default: None.
        size_average (bool, optional): Deprecated (see :attr:`reduction`).
        reduce (bool, optional): Deprecated (see :attr:`reduction`).
        reduction (str, optional): Specifies the reduction to apply to the output:
                                  'none' | 'mean' | 'sum'. 'mean': the mean of the output is taken.
                                  'sum': the output will be summed. 'none': no reduction will be applied.
                                  Default: 'mean'.

    Returns:
        Tensor: Multi margin loss (optionally weighted).
    """

pixel_shuffle: Incomplete
pixel_unshuffle: Incomplete
channel_shuffle: Incomplete
native_channel_shuffle: Incomplete

def _is_integer(x) -> bool:
    """Type check the input number is an integer.

    Will return True for int, SymInt, Numpy integers and Tensors with integer elements.
    """

GRID_SAMPLE_INTERPOLATION_MODES: Incomplete
GRID_SAMPLE_PADDING_MODES: Incomplete

def grid_sample(input: Tensor, grid: Tensor, mode: str = 'bilinear', padding_mode: str = 'zeros', align_corners: bool | None = None) -> Tensor:
    '''Compute grid sample.

    Given an :attr:`input` and a flow-field :attr:`grid`, computes the
    ``output`` using :attr:`input` values and pixel locations from :attr:`grid`.

    Currently, only spatial (4-D) and volumetric (5-D) :attr:`input` are
    supported.

    In the spatial (4-D) case, for :attr:`input` with shape
    :math:`(N, C, H_\\text{in}, W_\\text{in})` and :attr:`grid` with shape
    :math:`(N, H_\\text{out}, W_\\text{out}, 2)`, the output will have shape
    :math:`(N, C, H_\\text{out}, W_\\text{out})`.

    For each output location ``output[n, :, h, w]``, the size-2 vector
    ``grid[n, h, w]`` specifies :attr:`input` pixel locations ``x`` and ``y``,
    which are used to interpolate the output value ``output[n, :, h, w]``.
    In the case of 5D inputs, ``grid[n, d, h, w]`` specifies the
    ``x``, ``y``, ``z`` pixel locations for interpolating
    ``output[n, :, d, h, w]``. :attr:`mode` argument specifies ``nearest`` or
    ``bilinear`` interpolation method to sample the input pixels.

    :attr:`grid` specifies the sampling pixel locations normalized by the
    :attr:`input` spatial dimensions. Therefore, it should have most values in
    the range of ``[-1, 1]``. For example, values ``x = -1, y = -1`` is the
    left-top pixel of :attr:`input`, and values  ``x = 1, y = 1`` is the
    right-bottom pixel of :attr:`input`.

    If :attr:`grid` has values outside the range of ``[-1, 1]``, the corresponding
    outputs are handled as defined by :attr:`padding_mode`. Options are

        * ``padding_mode="zeros"``: use ``0`` for out-of-bound grid locations,
        * ``padding_mode="border"``: use border values for out-of-bound grid locations,
        * ``padding_mode="reflection"``: use values at locations reflected by
          the border for out-of-bound grid locations. For location far away
          from the border, it will keep being reflected until becoming in bound,
          e.g., (normalized) pixel location ``x = -3.5`` reflects by border ``-1``
          and becomes ``x\' = 1.5``, then reflects by border ``1`` and becomes
          ``x\'\' = -0.5``.

    Note:
        This function is often used in conjunction with :func:`affine_grid`
        to build `Spatial Transformer Networks`_ .

    Note:
        When using the CUDA backend, this operation may induce nondeterministic
        behaviour in its backward pass that is not easily switched off.
        Please see the notes on :doc:`/notes/randomness` for background.

    Note:
        NaN values in :attr:`grid` would be interpreted as ``-1``.

    Args:
        input (Tensor): input of shape :math:`(N, C, H_\\text{in}, W_\\text{in})` (4-D case)
                        or :math:`(N, C, D_\\text{in}, H_\\text{in}, W_\\text{in})` (5-D case)
        grid (Tensor): flow-field of shape :math:`(N, H_\\text{out}, W_\\text{out}, 2)` (4-D case)
                       or :math:`(N, D_\\text{out}, H_\\text{out}, W_\\text{out}, 3)` (5-D case)
        mode (str): interpolation mode to calculate output values
            ``\'bilinear\'`` | ``\'nearest\'`` | ``\'bicubic\'``. Default: ``\'bilinear\'``
            Note: ``mode=\'bicubic\'`` supports only 4-D input.
            When ``mode=\'bilinear\'`` and the input is 5-D, the interpolation mode
            used internally will actually be trilinear. However, when the input is 4-D,
            the interpolation mode will legitimately be bilinear.
        padding_mode (str): padding mode for outside grid values
            ``\'zeros\'`` | ``\'border\'`` | ``\'reflection\'``. Default: ``\'zeros\'``
        align_corners (bool, optional): Geometrically, we consider the pixels of the
            input  as squares rather than points.
            If set to ``True``, the extrema (``-1`` and ``1``) are considered as referring
            to the center points of the input\'s corner pixels. If set to ``False``, they
            are instead considered as referring to the corner points of the input\'s corner
            pixels, making the sampling more resolution agnostic.
            This option parallels the ``align_corners`` option in
            :func:`interpolate`, and so whichever option is used here
            should also be used there to resize the input image before grid sampling.
            Default: ``False``

    Returns:
        output (Tensor): output Tensor

    .. _`Spatial Transformer Networks`:
        https://arxiv.org/abs/1506.02025

    .. warning::
        When ``align_corners = True``, the grid positions depend on the pixel
        size relative to the input image size, and so the locations sampled by
        :func:`grid_sample` will differ for the same input given at different
        resolutions (that is, after being upsampled or downsampled).
        The default behavior up to version 1.2.0 was ``align_corners = True``.
        Since then, the default behavior has been changed to ``align_corners = False``,
        in order to bring it in line with the default for :func:`interpolate`.

    .. note::
        ``mode=\'bicubic\'`` is implemented using the `cubic convolution algorithm`_ with :math:`\\alpha=-0.75`.
        The constant :math:`\\alpha` might be different from packages to packages.
        For example, `PIL`_ and `OpenCV`_ use -0.5 and -0.75 respectively.
        This algorithm may "overshoot" the range of values it\'s interpolating.
        For example, it may produce negative values or values greater than 255 when interpolating input in [0, 255].
        Clamp the results with :func:`torch.clamp` to ensure they are within the valid range.
    .. _`cubic convolution algorithm`: https://en.wikipedia.org/wiki/Bicubic_interpolation
    .. _`PIL`: https://github.com/python-pillow/Pillow/blob/4634eafe3c695a014267eefdce830b4a825beed7/src/libImaging/Resample.c#L51
    .. _`OpenCV`: https://github.com/opencv/opencv/blob/f345ed564a06178670750bad59526cfa4033be55/modules/imgproc/src/resize.cpp#L908
    '''
def affine_grid(theta: Tensor, size: list[int], align_corners: bool | None = None) -> Tensor:
    """Generate 2D or 3D flow field (sampling grid), given a batch of affine matrices :attr:`theta`.

    .. note::
        This function is often used in conjunction with :func:`grid_sample`
        to build `Spatial Transformer Networks`_ .

    Args:
        theta (Tensor): input batch of affine matrices with shape
            (:math:`N \\times 2 \\times 3`) for 2D or
            (:math:`N \\times 3 \\times 4`) for 3D
        size (torch.Size): the target output image size.
            (:math:`N \\times C \\times H \\times W` for 2D or
            :math:`N \\times C \\times D \\times H \\times W` for 3D)
            Example: torch.Size((32, 3, 24, 24))
        align_corners (bool, optional): if ``True``, consider ``-1`` and ``1``
            to refer to the centers of the corner pixels rather than the image corners.
            Refer to :func:`grid_sample` for a more complete description.
            A grid generated by :func:`affine_grid` should be passed to :func:`grid_sample`
            with the same setting for this option.
            Default: ``False``

    Returns:
        output (Tensor): output Tensor of size (:math:`N \\times H \\times W \\times 2`)

    .. _`Spatial Transformer Networks`:
        https://arxiv.org/abs/1506.02025

    .. warning::
        When ``align_corners = True``, the grid positions depend on the pixel
        size relative to the input image size, and so the locations sampled by
        :func:`grid_sample` will differ for the same input given at different
        resolutions (that is, after being upsampled or downsampled).
        The default behavior up to version 1.2.0 was ``align_corners = True``.
        Since then, the default behavior has been changed to ``align_corners = False``,
        in order to bring it in line with the default for :func:`interpolate`.
    .. warning::
        When ``align_corners = True``, 2D affine transforms on 1D data and
        3D affine transforms on 2D data (that is, when one of the spatial
        dimensions has unit size) are ill-defined, and not an intended use case.
        This is not a problem when ``align_corners = False``.
        Up to version 1.2.0, all grid points along a unit dimension were
        considered arbitrarily to be at ``-1``.
        From version 1.3.0, under ``align_corners = True`` all grid points
        along a unit dimension are considered to be at ``0``
        (the center of the input image).
    """
def pad(input: Tensor, pad: list[int], mode: str = 'constant', value: float | None = None) -> Tensor:
    '''
    pad(input, pad, mode="constant", value=None) -> Tensor

    Pads tensor.

    Padding size:
        The padding size by which to pad some dimensions of :attr:`input`
        are described starting from the last dimension and moving forward.
        :math:`\\left\\lfloor\\frac{\\text{len(pad)}}{2}\\right\\rfloor` dimensions
        of ``input`` will be padded.
        For example, to pad only the last dimension of the input tensor, then
        :attr:`pad` has the form
        :math:`(\\text{padding\\_left}, \\text{padding\\_right})`;
        to pad the last 2 dimensions of the input tensor, then use
        :math:`(\\text{padding\\_left}, \\text{padding\\_right},`
        :math:`\\text{padding\\_top}, \\text{padding\\_bottom})`;
        to pad the last 3 dimensions, use
        :math:`(\\text{padding\\_left}, \\text{padding\\_right},`
        :math:`\\text{padding\\_top}, \\text{padding\\_bottom}`
        :math:`\\text{padding\\_front}, \\text{padding\\_back})`.

    Padding mode:
        See :class:`torch.nn.CircularPad2d`, :class:`torch.nn.ConstantPad2d`,
        :class:`torch.nn.ReflectionPad2d`, and :class:`torch.nn.ReplicationPad2d`
        for concrete examples on how each of the padding modes works. Constant
        padding is implemented for arbitrary dimensions. Circular, replicate and
        reflection padding are implemented for padding the last 3 dimensions of a
        4D or 5D input tensor, the last 2 dimensions of a 3D or 4D input tensor,
        or the last dimension of a 2D or 3D input tensor.

    Note:
        When using the CUDA backend, this operation may induce nondeterministic
        behaviour in its backward pass that is not easily switched off.
        Please see the notes on :doc:`/notes/randomness` for background.

    Args:
        input (Tensor): N-dimensional tensor
        pad (tuple): m-elements tuple, where
            :math:`\\frac{m}{2} \\leq` input dimensions and :math:`m` is even.
        mode: ``\'constant\'``, ``\'reflect\'``, ``\'replicate\'`` or ``\'circular\'``.
            Default: ``\'constant\'``
        value: fill value for ``\'constant\'`` padding. Default: ``0``

    Examples::

        >>> t4d = torch.empty(3, 3, 4, 2)
        >>> p1d = (1, 1) # pad last dim by 1 on each side
        >>> out = F.pad(t4d, p1d, "constant", 0)  # effectively zero padding
        >>> print(out.size())
        torch.Size([3, 3, 4, 4])
        >>> p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
        >>> out = F.pad(t4d, p2d, "constant", 0)
        >>> print(out.size())
        torch.Size([3, 3, 8, 4])
        >>> t4d = torch.empty(3, 3, 4, 2)
        >>> p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
        >>> out = F.pad(t4d, p3d, "constant", 0)
        >>> print(out.size())
        torch.Size([3, 9, 7, 3])
    '''

pairwise_distance: Incomplete
pdist: Incomplete
cosine_similarity: Incomplete
one_hot: Incomplete

def triplet_margin_loss(anchor: Tensor, positive: Tensor, negative: Tensor, margin: float = 1.0, p: float = 2, eps: float = 1e-06, swap: bool = False, size_average: bool | None = None, reduce: bool | None = None, reduction: str = 'mean') -> Tensor:
    """Compute the triplet loss between given input tensors and a margin greater than 0.

    See :class:`~torch.nn.TripletMarginLoss` for details.
    """
def triplet_margin_with_distance_loss(anchor: Tensor, positive: Tensor, negative: Tensor, *, distance_function: Callable[[Tensor, Tensor], Tensor] | None = None, margin: float = 1.0, swap: bool = False, reduction: str = 'mean') -> Tensor:
    """Compute the triplet margin loss for input tensors using a custom distance function.

    See :class:`~torch.nn.TripletMarginWithDistanceLoss` for details.
    """
def normalize(input: Tensor, p: float = 2.0, dim: int = 1, eps: float = 1e-12, out: Tensor | None = None) -> Tensor:
    """Perform :math:`L_p` normalization of inputs over specified dimension.

    For a tensor :attr:`input` of sizes :math:`(n_0, ..., n_{dim}, ..., n_k)`, each
    :math:`n_{dim}` -element vector :math:`v` along dimension :attr:`dim` is transformed as

    .. math::
        v = \\frac{v}{\\max(\\lVert v \\rVert_p, \\epsilon)}.

    With the default arguments it uses the Euclidean norm over vectors along dimension :math:`1` for normalization.

    Args:
        input: input tensor of any shape
        p (float): the exponent value in the norm formulation. Default: 2
        dim (int or tuple of ints): the dimension to reduce. Default: 1
        eps (float): small value to avoid division by zero. Default: 1e-12
        out (Tensor, optional): the output tensor. If :attr:`out` is used, this
                                operation won't be differentiable.
    """
def assert_int_or_pair(arg: list[int], arg_name: str, message: str) -> None: ...
def unfold(input: Tensor, kernel_size: BroadcastingList2[int], dilation: BroadcastingList2[int] = 1, padding: BroadcastingList2[int] = 0, stride: BroadcastingList2[int] = 1) -> Tensor:
    """Extract sliding local blocks from a batched input tensor.

    .. warning::
        Currently, only 4-D input tensors (batched image-like tensors) are
        supported.

    .. warning::

        More than one element of the unfolded tensor may refer to a single
        memory location. As a result, in-place operations (especially ones that
        are vectorized) may result in incorrect behavior. If you need to write
        to the tensor, please clone it first.


    See :class:`torch.nn.Unfold` for details
    """
def fold(input: Tensor, output_size: BroadcastingList2[int], kernel_size: BroadcastingList2[int], dilation: BroadcastingList2[int] = 1, padding: BroadcastingList2[int] = 0, stride: BroadcastingList2[int] = 1) -> Tensor:
    """Combine an array of sliding local blocks into a large containing tensor.

    .. warning::
        Currently, only unbatched (3D) or batched (4D) image-like output tensors are supported.

    See :class:`torch.nn.Fold` for details
    """
def _in_projection_packed(q: Tensor, k: Tensor, v: Tensor, w: Tensor, b: Tensor | None = None) -> list[Tensor]:
    """Perform the in-projection step of the attention operation, using packed weights.

    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.

    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension

        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
def _in_projection(q: Tensor, k: Tensor, v: Tensor, w_q: Tensor, w_k: Tensor, w_v: Tensor, b_q: Tensor | None = None, b_k: Tensor | None = None, b_v: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
    """Perform the in-projection step of the attention operation.

    This is simply a triple of linear projections,
    with shape constraints on the weights which
    ensure embedding dimension uniformity in the projected outputs.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected.
        w_q, w_k, w_v: weights for q, k and v, respectively.
        b_q, b_k, b_v: optional biases for q, k and v, respectively.

    Shape:
        Inputs:
        - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
            number of leading dimensions.
        - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
            number of leading dimensions.
        - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
            number of leading dimensions.
        - w_q: :math:`(Eq, Eq)`
        - w_k: :math:`(Eq, Ek)`
        - w_v: :math:`(Eq, Ev)`
        - b_q: :math:`(Eq)`
        - b_k: :math:`(Eq)`
        - b_v: :math:`(Eq)`

        Output: in output triple :math:`(q', k', v')`,
         - q': :math:`[Qdims..., Eq]`
         - k': :math:`[Kdims..., Eq]`
         - v': :math:`[Vdims..., Eq]`

    """

scaled_dot_product_attention: Incomplete

def _mha_shape_check(query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Tensor | None, attn_mask: Tensor | None, num_heads: int): ...
def _canonical_mask(mask: Tensor | None, mask_name: str, other_type: DType | None, other_name: str, target_type: DType, check_other: bool = True) -> Tensor | None: ...
def _none_or_dtype(input: Tensor | None) -> DType | None: ...
def _check_key_padding_mask(key_padding_mask: torch.Tensor, src_len: int, bsz: int) -> None: ...
def multi_head_attention_forward(query: Tensor, key: Tensor, value: Tensor, embed_dim_to_check: int, num_heads: int, in_proj_weight: Tensor | None, in_proj_bias: Tensor | None, bias_k: Tensor | None, bias_v: Tensor | None, add_zero_attn: bool, dropout_p: float, out_proj_weight: Tensor, out_proj_bias: Tensor | None, training: bool = True, key_padding_mask: Tensor | None = None, need_weights: bool = True, attn_mask: Tensor | None = None, use_separate_proj_weight: bool = False, q_proj_weight: Tensor | None = None, k_proj_weight: Tensor | None = None, v_proj_weight: Tensor | None = None, static_k: Tensor | None = None, static_v: Tensor | None = None, average_attn_weights: bool = True, is_causal: bool = False) -> tuple[Tensor, Tensor | None]:
    '''Forward method for MultiHeadAttention.

    See :class:`torch.nn.MultiheadAttention` for details.

    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
            Default: `True`
            Note: `needs_weight` defaults to `True`, but should be set to `False`
            For best performance when attention weights are not needed.
            *Setting needs_weights to `True`
            leads to a significant performance degradation.*
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        is_causal: If specified, applies a causal mask as attention mask, and ignores
            attn_mask for computing scaled dot product attention.
            Default: ``False``.
            .. warning::
                is_causal is provides a hint that the attn_mask is the
                causal mask.Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across heads.
            Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an effect
            when ``need_weights=True.``. Default: True


    Shape:
        Inputs:
        - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a FloatTensor is provided, it will be directly added to the value.
          If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns
          attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.
    '''
