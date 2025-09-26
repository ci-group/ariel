from torch.onnx._internal.exporter import _registration
from typing import Callable, TypeVar

__all__ = ['onnx_impl', 'get_torchlib_ops']

_T = TypeVar('_T', bound=Callable)

def onnx_impl(target: _registration.TorchOp | tuple[_registration.TorchOp, ...], *, trace_only: bool = False, complex: bool = False, opset_introduced: int = 18, no_compile: bool = False, private: bool = False) -> Callable[[_T], _T]:
    """Register an ONNX implementation of a torch op."""
def get_torchlib_ops() -> tuple[_registration.OnnxDecompMeta, ...]: ...
