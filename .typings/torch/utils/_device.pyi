import torch
import types
from _typeshed import Incomplete
from torch._C import _len_torch_function_stack as _len_torch_function_stack
from torch.overrides import TorchFunctionMode as TorchFunctionMode, _pop_mode as _pop_mode, _push_mode as _push_mode
from torch.utils._contextlib import context_decorator as context_decorator

CURRENT_DEVICE: torch.device | None

def _device_constructors(): ...

class DeviceContext(TorchFunctionMode):
    device: Incomplete
    def __init__(self, device) -> None: ...
    old_device: Incomplete
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: types.TracebackType | None) -> None: ...
    def __torch_function__(self, func, types, args=(), kwargs=None): ...

def device_decorator(device, func): ...
def set_device(device):
    """
    Set the default device inside of the wrapped function by decorating it with this function.

    If you would like to use this as a context manager, use device as a
    context manager directly, e.g., ``with torch.device(device)``.
    """
