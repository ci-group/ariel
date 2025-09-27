import onnx_ir
from _typeshed import Incomplete
from typing import Any

class _LazyModule:
    """Lazily import a module."""
    _name: Incomplete
    _module: Any
    def __init__(self, module_name: str) -> None: ...
    def __repr__(self) -> str: ...
    def __getattr__(self, attr: str) -> object: ...
onnxscript_ir = onnx_ir
