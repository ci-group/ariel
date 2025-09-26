import torch
from _typeshed import Incomplete
from torch._prims_common import NumberType, ShapeType, TensorLikeType
from torch._prims_common.wrappers import elementwise_unary_scalar_wrapper
from typing import TypeVar
from typing_extensions import ParamSpec

__all__ = ['alpha_dropout', 'celu', 'celu_', 'channel_shuffle', 'dropout', 'elu', 'elu_', 'gelu', 'glu', 'group_norm', 'hardshrink', 'hardtanh', 'hinge_embedding_loss', 'huber_loss', 'l1_loss', 'layer_norm', 'leaky_relu', 'log_softmax', 'margin_ranking_loss', 'mish', 'mish_', 'mse_loss', 'nll_loss', 'pairwise_distance', 'pdist', 'poisson_nll_loss', 'prelu', 'relu', 'relu6', 'selu', 'selu_', 'smooth_l1_loss', 'softmax', 'softmin', 'softplus', 'softshrink', 'tanhshrink', 'threshold', 'threshold_', 'triplet_margin_loss']

_T = TypeVar('_T')
_P = ParamSpec('_P')
Tensor = torch.Tensor
DispatchKey = torch._C.DispatchKey

def alpha_dropout(self, p: float = 0.5, training: bool = False, inplace: bool = False) -> TensorLikeType: ...
@_inplace_wrapper
def celu(a: TensorLikeType, alpha: NumberType | None = None, inplace: bool = False) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.celu
    """
@_inplace_wrapper
def dropout(a: TensorLikeType, p: float = 0.5, training: bool = True, inplace: bool = False) -> TensorLikeType: ...
@_inplace_wrapper
def elu(a: TensorLikeType, alpha: NumberType = 1.0, scale: NumberType = 1.0, input_scale: NumberType = 1.0, inplace: bool = False) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.elu
    """
@_inplace_wrapper
def relu(a: TensorLikeType, inplace: bool = False) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.relu
    """
def channel_shuffle(input: TensorLikeType, groups: int) -> TensorLikeType:
    """
    Reference implementation of :func:`torch.nn.functional.channel_shuffle`.
    """
def group_norm(input: Tensor, num_groups: int, weight: Tensor | None = None, bias: Tensor | None = None, eps: float = 1e-05) -> Tensor:
    """
    Reference implementation of :func:`torch.nn.functional.group_norm`.
    """
def layer_norm(input: Tensor, normalized_shape: ShapeType, weight: Tensor | None = None, bias: Tensor | None = None, eps: float = 1e-05) -> Tensor:
    """
    Reference implementation of :func:`torch.nn.functional.layer_norm`.
    """
@_inplace_wrapper
def leaky_relu(a: TensorLikeType, negative_slope: float = 0.01, inplace: bool = False) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.leaky_relu
    """
@_inplace_wrapper
def mish(a: TensorLikeType, inplace: bool = False) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.mish
    """
@_inplace_wrapper
def selu(a: TensorLikeType, inplace: bool = False) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.selu
    """
def softmax(a: TensorLikeType, dim: int | None = None, _stacklevel: int = 3, dtype: torch.dtype | None = None) -> TensorLikeType: ...
def softmin(a: TensorLikeType, dim: int | None = None, _stacklevel: int = 3, dtype: torch.dtype | None = None) -> TensorLikeType: ...
@_inplace_wrapper
def softplus(a: TensorLikeType, beta: NumberType | None = None, threshold: NumberType = 20, inplace: bool = False) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.softplus
    """
def hardshrink(a: TensorLikeType, lambd: float = 0.5): ...
def softshrink(a: TensorLikeType, lambd: float = 0.5): ...
def l1_loss(input: TensorLikeType, target: TensorLikeType, size_average: bool | None = None, reduce: bool | None = None, reduction: str = 'mean') -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.l1_loss
    """
def smooth_l1_loss(input: TensorLikeType, target: TensorLikeType, size_average: bool | None = None, reduce: bool | None = None, reduction: str = 'mean', beta: float = 1.0) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.smooth_l1_loss
    """
def log_softmax(a: TensorLikeType, dim: int | None = None, _stacklevel: int = 3, dtype: torch.dtype | None = None) -> TensorLikeType: ...
def margin_ranking_loss(input1: TensorLikeType, input2: TensorLikeType, target: TensorLikeType, margin: float = 0.0, reduction: str = 'mean') -> TensorLikeType: ...
def mse_loss(input: TensorLikeType, target: TensorLikeType, size_average: bool | None = None, reduce: bool | None = None, reduction: str = 'mean') -> TensorLikeType: ...
def hinge_embedding_loss(input: TensorLikeType, target: TensorLikeType, margin: float = 1.0, reduction: str = 'mean') -> TensorLikeType: ...
def nll_loss(input: TensorLikeType, target: TensorLikeType, weight: TensorLikeType | None = None, size_average: bool | None = None, ignore_index: int = -100, reduce: bool | None = None, reduction: str = 'mean') -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.nll_loss
    """
def huber_loss(input: TensorLikeType, target: TensorLikeType, reduction: str | int = 'mean', delta: float = 1.0) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.huber_loss
    """
@elementwise_unary_scalar_wrapper
def tanhshrink(a: TensorLikeType) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.tanhshrink
    """
@_inplace_wrapper
def threshold(a: TensorLikeType, threshold: NumberType, value: bool | int | float, inplace: bool = False) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.threshold
    """
def triplet_margin_loss(anchor: TensorLikeType, positive: TensorLikeType, negative: TensorLikeType, margin: float = 1.0, p: float = 2, eps: float = 1e-06, swap: bool = False, size_average: bool | None = None, reduce: bool | None = None, reduction: str = 'mean') -> TensorLikeType: ...
@_inplace_wrapper
@elementwise_unary_scalar_wrapper
def hardtanh(a: TensorLikeType, min_val: NumberType = -1, max_val: NumberType = 1, inplace: bool = False) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.hardtanh
    """
@elementwise_unary_scalar_wrapper
def gelu(a: TensorLikeType, approximate: str = 'none') -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.gelu
    """
def poisson_nll_loss(input: TensorLikeType, target: TensorLikeType, log_input: bool = True, full: bool = False, size_average: bool | None = None, eps: float = 1e-08, reduce: bool | None = None, reduction: str = 'mean') -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.poisson_nll_loss
    """
def prelu(a: TensorLikeType, weight: TensorLikeType) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.prelu
    """
@_inplace_wrapper
def relu6(a: TensorLikeType, inplace: bool = False) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.relu6
    """
def glu(a: TensorLikeType, dim: int = -1) -> TensorLikeType: ...
def pairwise_distance(x1: TensorLikeType, x2: TensorLikeType, p: NumberType = 2.0, eps: NumberType = 1e-06, keepdim: bool = False) -> TensorLikeType: ...
def pdist(a: TensorLikeType, p: float = 2) -> TensorLikeType: ...

celu_: Incomplete
elu_: Incomplete
mish_: Incomplete
selu_: Incomplete
threshold_: Incomplete
