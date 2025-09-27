from .core import _map_mt_args_kwargs as _map_mt_args_kwargs, _wrap_result as _wrap_result
from _typeshed import Incomplete

__all__: Incomplete
UNARY_NAMES: Incomplete
INPLACE_UNARY_NAMES: Incomplete
UNARY_NAMES_UNSUPPORTED: Incomplete

def _unary_helper(fn, args, kwargs, inplace): ...
def _torch_unary(fn_name): ...
def _torch_inplace_unary(fn_name): ...

NATIVE_UNARY_MAP: Incomplete
NATIVE_INPLACE_UNARY_MAP: Incomplete
NATIVE_UNARY_FNS: Incomplete
NATIVE_INPLACE_UNARY_FNS: Incomplete

def _is_native_unary(fn): ...
def _apply_native_unary(fn, *args, **kwargs): ...
