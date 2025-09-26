import types
from _typeshed import Incomplete
from torch._dynamo.exc import BackendCompilerFailed as BackendCompilerFailed, ShortenTraceback as ShortenTraceback
from torch.cuda import _CudaDeviceProperties as _CudaDeviceProperties
from typing import Any

def _record_missing_op(target: Any) -> None: ...

class OperatorIssue(RuntimeError):
    @staticmethod
    def operator_str(target: Any, args: list[Any], kwargs: dict[str, Any]) -> str: ...

class MissingOperatorWithoutDecomp(OperatorIssue):
    def __init__(self, target: Any, args: list[Any], kwargs: dict[str, Any]) -> None: ...

class MissingOperatorWithDecomp(OperatorIssue):
    def __init__(self, target: Any, args: list[Any], kwargs: dict[str, Any]) -> None: ...

class LoweringException(OperatorIssue):
    def __init__(self, exc: Exception, target: Any, args: list[Any], kwargs: dict[str, Any]) -> None: ...

class SubgraphLoweringException(RuntimeError): ...

class InvalidCxxCompiler(RuntimeError):
    def __init__(self) -> None: ...

class CppWrapperCodegenError(RuntimeError):
    def __init__(self, msg: str) -> None: ...

class CppCompileError(RuntimeError):
    def __init__(self, cmd: list[str], output: str) -> None: ...

class CUDACompileError(CppCompileError): ...

class TritonMissing(ShortenTraceback):
    def __init__(self, first_useful_frame: types.FrameType | None) -> None: ...

class GPUTooOldForTriton(ShortenTraceback):
    def __init__(self, device_props: _CudaDeviceProperties, first_useful_frame: types.FrameType | None) -> None: ...

class InductorError(BackendCompilerFailed):
    backend_name: str
    inner_exception: Incomplete
    def __init__(self, inner_exception: Exception, first_useful_frame: types.FrameType | None) -> None: ...
