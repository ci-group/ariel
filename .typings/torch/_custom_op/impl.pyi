import dataclasses
import torch._C as _C
import torch.library as library
import typing
from _typeshed import Incomplete
from torch.library import get_ctx as get_ctx

__all__ = ['custom_op', 'CustomOp', 'get_ctx']

def custom_op(qualname: str, manual_schema: str | None = None) -> typing.Callable:
    """
    This API is deprecated, please use torch.library.custom_op instead
    """

class CustomOp:
    """
    This API is deprecated, please use torch.library.custom_op instead
    """
    _schema: Incomplete
    _cpp_ns: Incomplete
    _lib: library.Library
    _ophandle: _C._DispatchOperatorHandle
    _opname: str
    _qualname: str
    __name__: Incomplete
    _impls: dict[str, FuncAndLocation | None]
    _registered_autograd_kernel_indirection: bool
    def __init__(self, lib, cpp_ns, schema, operator_name, ophandle, *, _private_access: bool = False) -> None: ...
    def _register_autograd_kernel_indirection(self) -> None: ...
    def _register_impl(self, kind, func, stacklevel: int = 2) -> None: ...
    def _get_impl(self, kind): ...
    def _has_impl(self, kind): ...
    def _destroy(self) -> None: ...
    def __repr__(self) -> str: ...
    def __call__(self, *args, **kwargs): ...
    def impl(self, device_types: str | typing.Iterable[str], _stacklevel: int = 2) -> typing.Callable:
        """
        This API is deprecated, please use torch.library.custom_op instead
        """
    def _check_doesnt_have_library_impl(self, device_type) -> None: ...
    def impl_factory(self) -> typing.Callable:
        """Register an implementation for a factory function."""
    def impl_abstract(self, _stacklevel: int = 2) -> typing.Callable:
        """
        This API is deprecated, please use torch.library.custom_op instead
        """
    def _check_can_register_backward(self) -> None: ...
    def _check_doesnt_have_library_autograd_impl(self) -> None: ...
    def _check_doesnt_have_library_meta_impl(self) -> None: ...
    def _register_autograd_kernel(self) -> None: ...
    def impl_save_for_backward(self, _stacklevel: int = 2):
        """Register a function that tells us what to save for backward.

        Please see impl_backward for more details.
        """
    _output_differentiability: Incomplete
    def impl_backward(self, output_differentiability=None, _stacklevel: int = 2):
        """
        This API is deprecated, please use torch.library.custom_op instead
        """

@dataclasses.dataclass
class FuncAndLocation:
    func: typing.Callable
    location: str
