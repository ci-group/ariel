import torch
import torch._C
from .memory import empty_cache as empty_cache, max_memory_allocated as max_memory_allocated, max_memory_reserved as max_memory_reserved, mem_get_info as mem_get_info, memory_allocated as memory_allocated, memory_reserved as memory_reserved, memory_stats as memory_stats, memory_stats_as_nested_dict as memory_stats_as_nested_dict, reset_accumulated_memory_stats as reset_accumulated_memory_stats, reset_peak_memory_stats as reset_peak_memory_stats
from .random import get_rng_state as get_rng_state, get_rng_state_all as get_rng_state_all, initial_seed as initial_seed, manual_seed as manual_seed, manual_seed_all as manual_seed_all, seed as seed, seed_all as seed_all, set_rng_state as set_rng_state, set_rng_state_all as set_rng_state_all
from .streams import Event as Event, Stream as Stream
from _typeshed import Incomplete
from torch import device as _device
from typing import Any

__all__ = ['Event', 'Stream', 'StreamContext', 'current_device', 'current_stream', 'default_generators', 'device', 'device_of', 'device_count', 'empty_cache', 'get_arch_list', 'get_device_capability', 'get_device_name', 'get_device_properties', 'get_gencode_flags', 'get_rng_state', 'get_rng_state_all', 'get_stream_from_external', 'init', 'initial_seed', 'is_available', 'is_bf16_supported', 'is_initialized', 'manual_seed', 'manual_seed_all', 'max_memory_allocated', 'max_memory_reserved', 'mem_get_info', 'memory_allocated', 'memory_reserved', 'memory_stats', 'memory_stats_as_nested_dict', 'reset_accumulated_memory_stats', 'reset_peak_memory_stats', 'seed', 'seed_all', 'set_device', 'set_rng_state', 'set_rng_state_all', 'set_stream', 'stream', 'streams', 'synchronize']

_device_t = _device | str | int | None
default_generators: tuple[torch._C.Generator]
_XpuDeviceProperties = torch._C._XpuDeviceProperties
_exchange_device = torch._C._xpu_exchangeDevice
_maybe_exchange_device = torch._C._xpu_maybeExchangeDevice

def device_count() -> int:
    """Return the number of XPU device available."""
def is_available() -> bool:
    """Return a bool indicating if XPU is currently available."""
def is_bf16_supported(including_emulation: bool = True) -> bool:
    """Return a bool indicating if the current XPU device supports dtype bfloat16."""
def is_initialized():
    """Return whether PyTorch's XPU state has been initialized."""
def init() -> None:
    """Initialize PyTorch's XPU state.
    This is a Python API about lazy initialization that avoids initializing
    XPU until the first time it is accessed. Does nothing if the XPU state is
    already initialized.
    """

class _DeviceGuard:
    idx: Incomplete
    prev_idx: int
    def __init__(self, index: int) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, type: Any, value: Any, traceback: Any): ...

class device:
    """Context-manager that changes the selected device.

    Args:
        device (torch.device or int or str): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """
    idx: Incomplete
    prev_idx: int
    def __init__(self, device: Any) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, type: Any, value: Any, traceback: Any): ...

class device_of(device):
    """Context-manager that changes the current device to that of given object.

    You can use both tensors and storages as arguments. If a given object is
    not allocated on a XPU, this is a no-op.

    Args:
        obj (Tensor or Storage): object allocated on the selected device.
    """
    def __init__(self, obj) -> None: ...

def set_device(device: _device_t) -> None:
    """Set the current device.

    Args:
        device (torch.device or int or str): selected device. This function is a
            no-op if this argument is negative.
    """
def get_device_name(device: _device_t | None = None) -> str:
    """Get the name of a device.

    Args:
        device (torch.device or int or str, optional): device for which to
            return the name. This function is a no-op if this argument is a
            negative integer. It uses the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).

    Returns:
        str: the name of the device
    """
def get_device_capability(device: _device_t | None = None) -> dict[str, Any]:
    """Get the xpu capability of a device.

    Args:
        device (torch.device or int or str, optional): device for which to
            return the device capability. This function is a no-op if this
            argument is a negative integer. It uses the current device, given by
            :func:`~torch.xpu.current_device`, if :attr:`device` is ``None``
            (default).

    Returns:
        Dict[str, Any]: the xpu capability dictionary of the device
    """
def get_device_properties(device: _device_t | None = None) -> _XpuDeviceProperties:
    """Get the properties of a device.

    Args:
        device (torch.device or int or str): device for which to return the
            properties of the device.

    Returns:
        _XpuDeviceProperties: the properties of the device
    """
def current_device() -> int:
    """Return the index of a currently selected device."""

class StreamContext:
    """Context-manager that selects a given stream.

    All XPU kernels queued within its context will be enqueued on a selected
    stream.

    Args:
        Stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note:: Streams are per-device.
    """
    cur_stream: torch.xpu.Stream | None
    stream: Incomplete
    idx: Incomplete
    def __init__(self, stream: torch.xpu.Stream | None) -> None: ...
    src_prev_stream: Incomplete
    dst_prev_stream: Incomplete
    def __enter__(self) -> None: ...
    def __exit__(self, type: Any, value: Any, traceback: Any): ...

def stream(stream: torch.xpu.Stream | None) -> StreamContext:
    """Wrap around the Context-manager StreamContext that selects a given stream.

    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's ``None``.
    """
def set_stream(stream: Stream):
    """Set the current stream.This is a wrapper API to set the stream.
        Usage of this function is discouraged in favor of the ``stream``
        context manager.

    Args:
        stream (Stream): selected stream. This function is a no-op
            if this argument is ``None``.
    """
def current_stream(device: _device_t | None = None) -> Stream:
    """Return the currently selected :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            the currently selected :class:`Stream` for the current device, given
            by :func:`~torch.xpu.current_device`, if :attr:`device` is ``None``
            (default).
    """
def get_stream_from_external(data_ptr: int, device: _device_t | None = None) -> Stream:
    """Return a :class:`Stream` from an external SYCL queue.

    This function is used to wrap SYCL queue created in other libraries in order
    to facilitate data exchange and multi-library interactions.

    .. note:: This function doesn't manage the queue life-cycle, it is the user
       responsibility to keep the referenced queue alive while this returned stream is
       being used. The different SYCL queue pointers will result in distinct
       :class:`Stream` objects, even if the SYCL queues they dereference are equivalent.

    Args:
        data_ptr(int): Integer representation of the `sycl::queue*` value passed externally.
        device(torch.device or int, optional): the device where the queue was originally created.
            It is the user responsibility to ensure the device is specified correctly.
    """
def synchronize(device: _device_t = None) -> None:
    """Wait for all kernels in all streams on a XPU device to complete.

    Args:
        device (torch.device or int, optional): device for which to synchronize.
            It uses the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
def get_arch_list() -> list[str]:
    """Return list XPU architectures this library was compiled for."""
def get_gencode_flags() -> str:
    """Return XPU AOT(ahead-of-time) build flags this library was compiled with."""

# Names in __all__ with no definition:
#   streams
