from .. import device as _device
from _typeshed import Incomplete
from contextlib import AbstractContextManager
from typing import Any

__all__ = ['is_available', 'is_initialized', 'synchronize', 'current_device', 'current_stream', 'stream', 'set_device', 'device_count', 'Stream', 'StreamContext', 'Event']

_device_t = _device | str | int | None

def is_available() -> bool:
    """Returns a bool indicating if CPU is currently available.

    N.B. This function only exists to facilitate device-agnostic code

    """
def synchronize(device: _device_t = None) -> None:
    """Waits for all kernels in all streams on the CPU device to complete.

    Args:
        device (torch.device or int, optional): ignored, there's only one CPU device.

    N.B. This function only exists to facilitate device-agnostic code.
    """

class Stream:
    """
    N.B. This class only exists to facilitate device-agnostic code
    """
    def __init__(self, priority: int = -1) -> None: ...
    def wait_stream(self, stream) -> None: ...
    def record_event(self) -> None: ...
    def wait_event(self, event) -> None: ...

class Event:
    def query(self) -> bool: ...
    def record(self, stream=None) -> None: ...
    def synchronize(self) -> None: ...
    def wait(self, stream=None) -> None: ...
_current_stream = _default_cpu_stream

def current_stream(device: _device_t = None) -> Stream:
    """Returns the currently selected :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): Ignored.

    N.B. This function only exists to facilitate device-agnostic code

    """

class StreamContext(AbstractContextManager):
    """Context-manager that selects a given stream.

    N.B. This class only exists to facilitate device-agnostic code

    """
    cur_stream: Stream | None
    stream: Incomplete
    prev_stream: Incomplete
    def __init__(self, stream) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, type: Any, value: Any, traceback: Any) -> None: ...

def stream(stream: Stream) -> AbstractContextManager:
    """Wrapper around the Context-manager StreamContext that
    selects a given stream.

    N.B. This function only exists to facilitate device-agnostic code
    """
def device_count() -> int:
    """Returns number of CPU devices (not cores). Always 1.

    N.B. This function only exists to facilitate device-agnostic code
    """
def set_device(device: _device_t) -> None:
    """Sets the current device, in CPU we do nothing.

    N.B. This function only exists to facilitate device-agnostic code
    """
def current_device() -> str:
    """Returns current device for cpu. Always 'cpu'.

    N.B. This function only exists to facilitate device-agnostic code
    """
def is_initialized() -> bool:
    """Returns True if the CPU is initialized. Always True.

    N.B. This function only exists to facilitate device-agnostic code
    """
