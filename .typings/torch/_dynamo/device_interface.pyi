import torch
from _typeshed import Incomplete
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Callable

get_cuda_stream: Callable[[int], int] | None
caching_worker_device_properties: dict[str, Any]
caching_worker_current_devices: dict[str, int]

class DeviceInterface:
    """
    This is a simple device runtime interface for Inductor. It enables custom
    backends to be integrated with Inductor in a device-agnostic semantic.
    """
    class device:
        def __new__(cls, device: torch.types.Device): ...
    class Event:
        def __new__(cls, *args, **kwargs) -> None: ...
    class Stream:
        def __new__(cls, *args, **kwargs) -> None: ...
    class Worker:
        """
        Worker API to query device properties that will work in multi processing
        workers that cannot use the GPU APIs (due to processing fork() and
        initialization time issues). Properties are recorded in the main process
        before we fork the workers.
        """
        @staticmethod
        def set_device(device: int): ...
        @staticmethod
        def current_device() -> int: ...
        @staticmethod
        def get_device_properties(device: torch.types.Device = None): ...
    @staticmethod
    def current_device() -> None: ...
    @staticmethod
    def set_device(device: torch.types.Device): ...
    @staticmethod
    def maybe_exchange_device(device: int) -> int: ...
    @staticmethod
    def exchange_device(device: int) -> int: ...
    @staticmethod
    def device_count() -> None: ...
    @staticmethod
    def is_available() -> bool: ...
    @staticmethod
    def stream(stream: torch.Stream): ...
    @staticmethod
    def current_stream() -> None: ...
    @staticmethod
    def set_stream(stream: torch.Stream): ...
    @staticmethod
    def _set_stream_by_id(stream_id: int, device_index: int, device_type: int): ...
    @staticmethod
    def get_raw_stream(device_idx: int) -> int: ...
    @staticmethod
    def synchronize(device: torch.types.Device = None): ...
    @classmethod
    def get_device_properties(cls, device: torch.types.Device = None): ...
    @staticmethod
    def get_compute_capability(device: torch.types.Device = None): ...
    @staticmethod
    def is_bf16_supported(including_emulation: bool = False): ...
    @classmethod
    def is_dtype_supported(cls, dtype: torch.dtype, including_emulation: bool = False) -> bool: ...
    @staticmethod
    def memory_allocated(device: torch.types.Device = None) -> int: ...
    @staticmethod
    def is_triton_capable(device: torch.types.Device = None) -> bool:
        """
        Returns True if the device has Triton support, False otherwise, even if
        the appropriate Triton backend is not available.
        """
    @classmethod
    def raise_if_triton_unavailable(cls, device: torch.types.Device = None) -> None:
        """
        Raises a `RuntimeError` with the appropriate human-readable instructions
        to resolve the issue if Triton is not available for the given device, or
        the default device if `device` is `None`.

        The caller should ensure the presence of the 'triton' package before
        calling this method.
        """

class DeviceGuard:
    """
    This class provides a context manager for device switching. This is a stripped
    down version of torch.{device_name}.device.

    The context manager changes the current device to the given device index
    on entering the context and restores the original device on exiting.
    The device is switched using the provided device interface.
    """
    device_interface: Incomplete
    idx: Incomplete
    prev_idx: int
    def __init__(self, device_interface: type[DeviceInterface], index: int | None) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, type: Any, value: Any, traceback: Any): ...

class CudaInterface(DeviceInterface):
    device = torch.cuda.device
    Event = torch.cuda.Event
    Stream = torch.cuda.Stream
    class Worker:
        @staticmethod
        def set_device(device: int): ...
        @staticmethod
        def current_device() -> int: ...
        @staticmethod
        def get_device_properties(device: torch.types.Device = None): ...
    current_device: Incomplete
    set_device: Incomplete
    device_count: Incomplete
    stream: Incomplete
    current_stream: Incomplete
    set_stream: Incomplete
    _set_stream_by_id: Incomplete
    synchronize: Incomplete
    get_device_properties: Incomplete
    get_raw_stream: Incomplete
    exchange_device: Incomplete
    maybe_exchange_device: Incomplete
    memory_allocated: Incomplete
    is_bf16_supported: Incomplete
    @staticmethod
    def is_available() -> bool: ...
    @staticmethod
    def get_compute_capability(device: torch.types.Device = None): ...
    @staticmethod
    def is_triton_capable(device: torch.types.Device = None) -> bool: ...
    @staticmethod
    def raise_if_triton_unavailable(device: torch.types.Device = None) -> None: ...

get_xpu_stream: Callable[[int], int] | None

class XpuInterface(DeviceInterface):
    device = torch.xpu.device
    Event = torch.xpu.Event
    Stream = torch.xpu.Stream
    class Worker:
        @staticmethod
        def set_device(device: int): ...
        @staticmethod
        def current_device() -> int: ...
        @staticmethod
        def get_device_properties(device: torch.types.Device = None): ...
    current_device: Incomplete
    set_device: Incomplete
    device_count: Incomplete
    stream: Incomplete
    current_stream: Incomplete
    set_stream: Incomplete
    _set_stream_by_id: Incomplete
    synchronize: Incomplete
    get_device_properties: Incomplete
    get_raw_stream: Incomplete
    exchange_device: Incomplete
    maybe_exchange_device: Incomplete
    memory_allocated: Incomplete
    @staticmethod
    def is_available() -> bool: ...
    @staticmethod
    def get_compute_capability(device: torch.types.Device = None): ...
    @staticmethod
    def is_bf16_supported(including_emulation: bool = False) -> bool: ...
    @staticmethod
    def is_triton_capable(device: torch.types.Device = None) -> bool: ...
    @staticmethod
    def raise_if_triton_unavailable(evice: torch.types.Device = None) -> None: ...

@dataclass
class CpuDeviceProperties:
    multi_processor_count: int

class CpuInterface(DeviceInterface):
    class Event(torch.Event):
        time: float
        def __init__(self, enable_timing: bool = True) -> None: ...
        def elapsed_time(self, end_event) -> float: ...
        def record(self, stream=None) -> None: ...
    class Worker:
        @staticmethod
        def get_device_properties(device: torch.types.Device = None): ...
    @staticmethod
    def is_available() -> bool: ...
    @staticmethod
    def is_bf16_supported(including_emulation: bool = False): ...
    @staticmethod
    def get_compute_capability(device: torch.types.Device = None) -> str: ...
    @staticmethod
    def get_raw_stream(device_idx) -> int: ...
    @staticmethod
    def current_device(): ...
    @staticmethod
    def synchronize(device: torch.types.Device = None): ...
    @staticmethod
    def is_triton_capable(device: torch.types.Device = None) -> bool: ...
    @staticmethod
    def raise_if_triton_unavailable(device: torch.types.Device = None) -> None: ...

class MpsInterface(DeviceInterface):
    @staticmethod
    def is_bf16_supported(including_emulation: bool = False) -> bool: ...
    @classmethod
    def is_dtype_supported(cls, dtype: torch.dtype, including_emulation: bool = False) -> bool: ...
    @staticmethod
    def is_available() -> bool: ...
    @staticmethod
    def current_device(): ...
    @staticmethod
    def get_compute_capability(device: torch.types.Device = None) -> str: ...
    @staticmethod
    def synchronize(device: torch.types.Device = None): ...
    class Worker:
        @staticmethod
        def get_device_properties(device: torch.types.Device = None): ...
        @staticmethod
        def current_device(): ...

device_interfaces: dict[str, type[DeviceInterface]]
_device_initialized: bool

def register_interface_for_device(device: str | torch.device, device_interface: type[DeviceInterface]): ...
def get_interface_for_device(device: str | torch.device) -> type[DeviceInterface]: ...
def get_registered_device_interfaces() -> Iterable[tuple[str, type[DeviceInterface]]]: ...
def init_device_reg() -> None: ...
