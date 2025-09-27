from typing import Any, Callable, TypeVar

_is_onnx_exporting: bool
TCallable = TypeVar('TCallable', bound=Callable[..., Any])

def set_onnx_exporting_flag(func: TCallable) -> TCallable: ...
