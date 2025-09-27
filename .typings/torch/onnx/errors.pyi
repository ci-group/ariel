from torch import _C

__all__ = ['OnnxExporterWarning', 'SymbolicValueError', 'UnsupportedOperatorError']

class OnnxExporterWarning(UserWarning):
    """Warnings in the ONNX exporter."""
class OnnxExporterError(RuntimeError):
    """Errors raised by the ONNX exporter. This is the base class for all exporter errors."""

class UnsupportedOperatorError(OnnxExporterError):
    """Raised when an operator is unsupported by the exporter."""
    def __init__(self, name: str, version: int, supported_version: int | None) -> None: ...

class SymbolicValueError(OnnxExporterError):
    """Errors around TorchScript values and nodes."""
    def __init__(self, msg: str, value: _C.Value) -> None: ...
