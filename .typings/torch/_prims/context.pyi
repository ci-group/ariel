import functools
import torch.overrides
from _typeshed import Incomplete
from collections.abc import Sequence
from torch._prims_common import torch_function_passthrough as torch_function_passthrough
from typing import Any, Callable, TypeVar
from typing_extensions import ParamSpec

_P = ParamSpec('_P')
_R = TypeVar('_R')

@functools.cache
def torch_to_refs_map() -> dict[Any, Any]:
    """
    Mapping of torch API functions to torch._refs functions.
    E.g. torch_to_refs_map()[torch.add] == torch._refs.add
    """
@functools.cache
def all_prims() -> set[Any]:
    """
    Set of all prim functions, e.g., torch._prims.add in all_prims()
    """

class TorchRefsMode(torch.overrides.TorchFunctionMode):
    """
    Switches the interpretation of torch.* functions and Tensor methods to
    use PrimTorch refs in torch._refs.  (Direct calls to _refs are unaffected.)

    >>> # xdoctest: +SKIP
    >>> with TorchRefsMode():
    ...     torch.add(x, y)  # calls torch._refs.add(x, y)

    By default, this context manager will fall back on the torch.* if the
    ref does not exist; set strict=True to error if this occurs.
    If the ref exists we still would like to fall back on the torch.* sometimes,
    this behavior can be customized by passing a function to should_fallback_fn.
    """
    strict: Incomplete
    should_fallback_fn: Incomplete
    prims_mode_cls: Incomplete
    def __init__(self, strict: bool = False, should_fallback_fn: Callable[..., bool] = ..., prims_mode_cls: type = ...) -> None: ...
    def __torch_function__(self, orig_func: Callable[_P, _R], types: Sequence[type], args: Sequence[Any] = (), kwargs: dict[str, Any] | None = None) -> Any: ...
