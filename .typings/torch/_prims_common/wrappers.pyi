import torch
from _typeshed import Incomplete
from collections.abc import Sequence
from torch._prims_common import CustomOutParamAnnotation as CustomOutParamAnnotation, ELEMENTWISE_TYPE_PROMOTION_KIND as ELEMENTWISE_TYPE_PROMOTION_KIND, Number as Number, NumberType as NumberType, ShapeType as ShapeType, TensorLike as TensorLike, TensorLikeType as TensorLikeType
from torch.utils._pytree import tree_flatten as tree_flatten, tree_unflatten as tree_unflatten
from typing import Callable, TypeVar, overload
from typing_extensions import ParamSpec

_T = TypeVar('_T')
_P = ParamSpec('_P')

@overload
def _maybe_convert_to_dtype(a: TensorLikeType, dtype: torch.dtype) -> TensorLikeType: ...
@overload
def _maybe_convert_to_dtype(a: NumberType, dtype: torch.dtype) -> NumberType: ...
@overload
def _maybe_convert_to_dtype(a: Sequence, dtype: torch.dtype) -> Sequence: ...
@overload
def _maybe_convert_to_dtype(a: None, dtype: torch.dtype) -> None: ...
def _maybe_convert_to_type(a: NumberType, typ: type) -> NumberType: ...
def _annotation_has_type(*, typ, annotation): ...

class elementwise_type_promotion_wrapper:
    """
    Adds elementwise type promotion to a Python reference implementation.

    Takes two kwargs, type_promoting_args and type_promotion_kind.

    type_promoting_args must be a string Sequence specifiying the argument names of all
    arguments that participate in type promotion (and should be type promoted). If the
    arg specifies a Sequence-type then every element of the Sequence will participate in
    type promotion.

    type_promotion_kind must be one of the kinds specified by ELEMENTWISE_TYPE_PROMOTION_KIND.
    See its documentation for details.

    The return_dtype will be coerced to the wrapped function's dtype arg if it is available and
    not None.

    Other type promotion behavior, like validating the Python type of scalar arguments, must
    be handled separately.
    """
    type_promoting_arg_names: Incomplete
    type_promotion_kind: Incomplete
    def __init__(self, *, type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND, type_promoting_args: Sequence[str] | None = None) -> None: ...
    def __call__(self, fn: Callable) -> Callable: ...

def _resize_output_check(out: TensorLikeType, shape: ShapeType): ...
def _maybe_resize_out(out: TensorLikeType, shape: ShapeType, memory_format: torch.memory_format | None = None): ...
def is_cpu_scalar(x: TensorLikeType) -> bool: ...
def check_copy_devices(*, copy_from: TensorLikeType, copy_to: TensorLikeType) -> None: ...
def _safe_copy_out(*, copy_from: TensorLikeType, copy_to: TensorLikeType, exact_dtype: bool = False): ...
def out_wrapper(*out_names: str, exact_dtype: bool = False, pass_is_out: bool = False, preserve_memory_format: bool = False) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...
def _maybe_remove_out_wrapper(fn: Callable): ...
def backwards_not_supported(prim): ...
def elementwise_unary_scalar_wrapper(fn: Callable[_P, _T]) -> Callable[_P, _T | NumberType]:
    """
    Allows unary operators that accept tensors to work with Python numbers.
    """
