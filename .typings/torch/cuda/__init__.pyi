from .memory import *
from .random import *
import ctypes
import torch
import torch._C
from . import amp as amp, jiterator as jiterator, nvtx as nvtx, profiler as profiler, sparse as sparse, tunable as tunable
from .graphs import CUDAGraph as CUDAGraph, graph as graph, graph_pool_handle as graph_pool_handle, is_current_stream_capturing as is_current_stream_capturing, make_graphed_callables as make_graphed_callables
from .streams import Event as Event, ExternalStream as ExternalStream, Stream as Stream
from _typeshed import Incomplete
from pathlib import Path
from torch._utils import classproperty
from torch.storage import _LegacyStorage
from torch.types import Device
from typing import Any

__all__ = ['BFloat16Storage', 'BFloat16Tensor', 'BoolStorage', 'BoolTensor', 'ByteStorage', 'ByteTensor', 'CharStorage', 'CharTensor', 'ComplexDoubleStorage', 'ComplexFloatStorage', 'DoubleStorage', 'DoubleTensor', 'FloatStorage', 'FloatTensor', 'HalfStorage', 'HalfTensor', 'IntStorage', 'IntTensor', 'LongStorage', 'LongTensor', 'ShortStorage', 'ShortTensor', 'CUDAGraph', 'CudaError', 'DeferredCudaCallError', 'Event', 'ExternalStream', 'Stream', 'StreamContext', 'amp', 'caching_allocator_alloc', 'caching_allocator_delete', 'caching_allocator_enable', 'can_device_access_peer', 'check_error', 'cudaStatus', 'cudart', 'current_blas_handle', 'current_device', 'current_stream', 'default_generators', 'default_stream', 'device', 'device_count', 'device_memory_used', 'device_of', 'empty_cache', 'get_allocator_backend', 'CUDAPluggableAllocator', 'change_current_allocator', 'get_arch_list', 'get_device_capability', 'get_device_name', 'get_device_properties', 'get_gencode_flags', 'get_per_process_memory_fraction', 'get_rng_state', 'get_rng_state_all', 'get_stream_from_external', 'get_sync_debug_mode', 'graph', 'graph_pool_handle', 'graphs', 'has_half', 'has_magma', 'host_memory_stats', 'host_memory_stats_as_nested_dict', 'init', 'initial_seed', 'ipc_collect', 'is_available', 'is_bf16_supported', 'is_current_stream_capturing', 'is_initialized', 'is_tf32_supported', 'jiterator', 'list_gpu_processes', 'make_graphed_callables', 'manual_seed', 'manual_seed_all', 'max_memory_allocated', 'max_memory_cached', 'max_memory_reserved', 'mem_get_info', 'memory', 'memory_allocated', 'memory_cached', 'memory_reserved', 'memory_snapshot', 'memory_stats', 'memory_stats_as_nested_dict', 'memory_summary', 'memory_usage', 'MemPool', 'use_mem_pool', 'temperature', 'power_draw', 'clock_rate', 'nccl', 'nvtx', 'profiler', 'random', 'reset_accumulated_host_memory_stats', 'reset_accumulated_memory_stats', 'reset_max_memory_allocated', 'reset_max_memory_cached', 'reset_peak_host_memory_stats', 'reset_peak_memory_stats', 'seed', 'seed_all', 'set_device', 'set_per_process_memory_fraction', 'set_rng_state', 'set_rng_state_all', 'set_stream', 'set_sync_debug_mode', 'sparse', 'stream', 'streams', 'synchronize', 'tunable', 'utilization']

class _amdsmi_cdll_hook:
    original_CDLL: Incomplete
    paths: list[str]
    def __init__(self) -> None: ...
    def hooked_CDLL(self, name: str | Path | None, *args: Any, **kwargs: Any) -> ctypes.CDLL: ...
    def __enter__(self) -> None: ...
    def __exit__(self, type: Any, value: Any, traceback: Any) -> None: ...
_PYNVML_ERR = err
_CudaDeviceProperties = torch._C._CudaDeviceProperties
_exchange_device = torch._C._cuda_exchangeDevice
_maybe_exchange_device = torch._C._cuda_maybeExchangeDevice
has_half: bool
has_magma: bool
default_generators: tuple[torch._C.Generator]

def is_available() -> bool:
    """
    Return a bool indicating if CUDA is currently available.

    .. note:: This function will NOT poison fork if the environment variable
        ``PYTORCH_NVML_BASED_CUDA_CHECK=1`` is set. For more details, see
        :ref:`multiprocessing-poison-fork-note`.
    """
def is_bf16_supported(including_emulation: bool = True):
    """Return a bool indicating if the current CUDA/ROCm device supports dtype bfloat16."""
def is_tf32_supported() -> bool:
    """Return a bool indicating if the current CUDA/ROCm device supports dtype tf32."""
def is_initialized():
    """Return whether PyTorch's CUDA state has been initialized."""

class DeferredCudaCallError(Exception): ...
AcceleratorError = torch._C.AcceleratorError
OutOfMemoryError = torch._C.OutOfMemoryError

def init() -> None:
    """Initialize PyTorch's CUDA state.

    You may need to call this explicitly if you are interacting with
    PyTorch via its C API, as Python bindings for CUDA functionality
    will not be available until this initialization takes place.
    Ordinary users should not need this, as all of PyTorch's CUDA methods
    automatically initialize CUDA state on-demand.

    Does nothing if the CUDA state is already initialized.
    """
def cudart():
    '''Retrieves the CUDA runtime API module.


    This function initializes the CUDA runtime environment if it is not already
    initialized and returns the CUDA runtime API module (_cudart). The CUDA
    runtime API module provides access to various CUDA runtime functions.

    Args:
        ``None``

    Returns:
        module: The CUDA runtime API module (_cudart).

    Raises:
        RuntimeError: If CUDA cannot be re-initialized in a forked subprocess.
        AssertionError: If PyTorch is not compiled with CUDA support or if libcudart functions are unavailable.

    Example of CUDA operations with profiling:
        >>> import torch
        >>> from torch.cuda import cudart, check_error
        >>> import os
        >>>
        >>> os.environ[\'CUDA_PROFILE\'] = \'1\'
        >>>
        >>> def perform_cuda_operations_with_streams():
        >>>     stream = torch.cuda.Stream()
        >>>     with torch.cuda.stream(stream):
        >>>         x = torch.randn(100, 100, device=\'cuda\')
        >>>         y = torch.randn(100, 100, device=\'cuda\')
        >>>         z = torch.mul(x, y)
        >>>     return z
        >>>
        >>> torch.cuda.synchronize()
        >>> print("====== Start nsys profiling ======")
        >>> check_error(cudart().cudaProfilerStart())
        >>> with torch.autograd.profiler.emit_nvtx():
        >>>     result = perform_cuda_operations_with_streams()
        >>>     print("CUDA operations completed.")
        >>> check_error(torch.cuda.cudart().cudaProfilerStop())
        >>> print("====== End nsys profiling ======")

    To run this example and save the profiling information, execute:
        >>> $ nvprof --profile-from-start off --csv --print-summary -o trace_name.prof -f -- python cudart_test.py

    This command profiles the CUDA operations in the provided script and saves
    the profiling information to a file named `trace_name.prof`.
    The `--profile-from-start off` option ensures that profiling starts only
    after the `cudaProfilerStart` call in the script.
    The `--csv` and `--print-summary` options format the profiling output as a
    CSV file and print a summary, respectively.
    The `-o` option specifies the output file name, and the `-f` option forces the
    overwrite of the output file if it already exists.
    '''

class cudaStatus:
    SUCCESS: int
    ERROR_NOT_READY: int

class CudaError(RuntimeError):
    def __init__(self, code: int) -> None: ...

def check_error(res: int) -> None: ...

class _DeviceGuard:
    idx: Incomplete
    prev_idx: int
    def __init__(self, index: int) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, type: Any, value: Any, traceback: Any): ...

class device:
    """Context-manager that changes the selected device.

    Args:
        device (torch.device or int): device index to select. It's a no-op if
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
    not allocated on a GPU, this is a no-op.

    Args:
        obj (Tensor or Storage): object allocated on the selected device.
    """
    def __init__(self, obj) -> None: ...

def set_device(device: Device) -> None:
    """Set the current device.

    Usage of this function is discouraged in favor of :any:`device`. In most
    cases it's better to use ``CUDA_VISIBLE_DEVICES`` environmental variable.

    Args:
        device (torch.device or int): selected device. This function is a no-op
            if this argument is negative.
    """
def get_device_name(device: Device = None) -> str:
    """Get the name of a device.

    Args:
        device (torch.device or int or str, optional): device for which to return the
            name. This function is a no-op if this argument is a negative
            integer. It uses the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Returns:
        str: the name of the device
    """
def get_device_capability(device: Device = None) -> tuple[int, int]:
    """Get the cuda capability of a device.

    Args:
        device (torch.device or int or str, optional): device for which to return the
            device capability. This function is a no-op if this argument is
            a negative integer. It uses the current device, given by
            :func:`~torch.cuda.current_device`, if :attr:`device` is ``None``
            (default).

    Returns:
        tuple(int, int): the major and minor cuda capability of the device
    """
def get_device_properties(device: Device = None) -> _CudaDeviceProperties:
    """Get the properties of a device.

    Args:
        device (torch.device or int or str, optional): device for which to return the
            properties of the device.  It uses the current device, given by
            :func:`~torch.cuda.current_device`, if :attr:`device` is ``None``
            (default).

    Returns:
        _CudaDeviceProperties: the properties of the device
    """
def can_device_access_peer(device: Device, peer_device: Device) -> bool:
    """Check if peer access between two devices is possible."""

class StreamContext:
    """Context-manager that selects a given stream.

    All CUDA kernels queued within its context will be enqueued on a selected
    stream.

    Args:
        Stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note:: Streams are per-device.
    """
    cur_stream: torch.cuda.Stream | None
    stream: Incomplete
    idx: Incomplete
    src_prev_stream: Incomplete
    dst_prev_stream: Incomplete
    def __init__(self, stream: torch.cuda.Stream | None) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, type: Any, value: Any, traceback: Any): ...

def stream(stream: torch.cuda.Stream | None) -> StreamContext:
    """Wrap around the Context-manager StreamContext that selects a given stream.

    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note::
        In eager mode stream is of type Stream class while in JIT it is
        an object of the custom class ``torch.classes.cuda.Stream``.
    """
def set_stream(stream: Stream):
    """Set the current stream.This is a wrapper API to set the stream.
        Usage of this function is discouraged in favor of the ``stream``
        context manager.

    Args:
        stream (Stream): selected stream. This function is a no-op
            if this argument is ``None``.
    """
def device_count() -> int:
    """
    Return the number of GPUs available.

    .. note:: This API will NOT posion fork if NVML discovery succeeds.
        See :ref:`multiprocessing-poison-fork-note` for more details.
    """
def get_arch_list() -> list[str]:
    """Return list CUDA architectures this library was compiled for."""
def get_gencode_flags() -> str:
    """Return NVCC gencode flags this library was compiled with."""
def current_device() -> int:
    """Return the index of a currently selected device."""
def synchronize(device: Device = None) -> None:
    """Wait for all kernels in all streams on a CUDA device to complete.

    Args:
        device (torch.device or int, optional): device for which to synchronize.
            It uses the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).
    """
def ipc_collect():
    """Force collects GPU memory after it has been released by CUDA IPC.

    .. note::
        Checks if any sent CUDA tensors could be cleaned from the memory. Force
        closes shared memory file used for reference counting if there is no
        active counters. Useful when the producer process stopped actively sending
        tensors and want to release unused memory.
    """
def current_stream(device: Device = None) -> Stream:
    """Return the currently selected :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            the currently selected :class:`Stream` for the current device, given
            by :func:`~torch.cuda.current_device`, if :attr:`device` is ``None``
            (default).
    """
def default_stream(device: Device = None) -> Stream:
    """Return the default :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            the default :class:`Stream` for the current device, given by
            :func:`~torch.cuda.current_device`, if :attr:`device` is ``None``
            (default).
    """
def get_stream_from_external(data_ptr: int, device: Device = None) -> Stream:
    """Return a :class:`Stream` from an externally allocated CUDA stream.

    This function is used to wrap streams allocated in other libraries in order
    to facilitate data exchange and multi-library interactions.

    .. note:: This function doesn't manage the stream life-cycle, it is the user
       responsibility to keep the referenced stream alive while this returned
       stream is being used.

    Args:
        data_ptr(int): Integer representation of the `cudaStream_t` value that
            is allocated externally.
        device(torch.device or int, optional): the device where the stream
            was originally allocated. If device is specified incorrectly,
            subsequent launches using this stream may fail.
    """
def current_blas_handle():
    """Return cublasHandle_t pointer to current cuBLAS handle"""
def set_sync_debug_mode(debug_mode: int | str) -> None:
    '''Set the debug mode for cuda synchronizing operations.

    Args:
        debug_mode(str or int): if "default" or 0, don\'t error or warn on synchronizing operations,
            if "warn" or 1, warn on synchronizing operations, if "error" or 2, error out synchronizing operations.

    Warning:
        This is an experimental feature, and not all synchronizing operations will trigger warning or error. In
        particular, operations in torch.distributed and torch.sparse namespaces are not covered yet.
    '''
def get_sync_debug_mode() -> int:
    """Return current value of debug mode for cuda synchronizing operations."""
def device_memory_used(device: Device = None) -> int:
    """Return used global (device) memory in bytes as given by `nvidia-smi` or `amd-smi`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    """
def memory_usage(device: Device = None) -> int:
    """Return the percent of time over the past sample period during which global (device)
    memory was being read or written as given by `nvidia-smi`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Warning: Each sample period may be between 1 second and 1/6 second,
    depending on the product being queried.
    """
def utilization(device: Device = None) -> int:
    """Return the percent of time over the past sample period during which one or
    more kernels was executing on the GPU as given by `nvidia-smi`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Warning: Each sample period may be between 1 second and 1/6 second,
    depending on the product being queried.
    """
def temperature(device: Device = None) -> int:
    """Return the average temperature of the GPU sensor in Degrees C (Centigrades).

    The average temperature is computed based on past sample period as given by `nvidia-smi`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Warning: Each sample period may be between 1 second and 1/6 second,
    depending on the product being queried.
    """
def power_draw(device: Device = None) -> int:
    """Return the average power draw of the GPU sensor in mW (MilliWatts)
        over the past sample period as given by `nvidia-smi` for Fermi or newer fully supported devices.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Warning: Each sample period may be between 1 second and 1/6 second,
    depending on the product being queried.
    """
def clock_rate(device: Device = None) -> int:
    """Return the clock speed of the GPU SM in MHz (megahertz) over the past sample period as given by `nvidia-smi`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Warning: Each sample period may be between 1 second and 1/6 second,
    depending on the product being queried.
    """

class _CudaBase:
    is_cuda: bool
    is_sparse: bool
    def type(self, *args, **kwargs): ...
    __new__ = _lazy_new

class _CudaLegacyStorage(_LegacyStorage):
    @classmethod
    def from_buffer(cls, *args, **kwargs) -> None: ...
    @classmethod
    def _new_with_weak_ptr(cls, *args, **kwargs) -> None: ...
    @classmethod
    def _new_shared_filename(cls, manager, obj, size, *, device=None, dtype=None) -> None: ...

class ByteStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self): ...
    @classproperty
    def _dtype(self): ...

class DoubleStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self): ...
    @classproperty
    def _dtype(self): ...

class FloatStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self): ...
    @classproperty
    def _dtype(self): ...

class HalfStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self): ...
    @classproperty
    def _dtype(self): ...

class LongStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self): ...
    @classproperty
    def _dtype(self): ...

class IntStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self): ...
    @classproperty
    def _dtype(self): ...

class ShortStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self): ...
    @classproperty
    def _dtype(self): ...

class CharStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self): ...
    @classproperty
    def _dtype(self): ...

class BoolStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self): ...
    @classproperty
    def _dtype(self): ...

class BFloat16Storage(_CudaLegacyStorage):
    @classproperty
    def dtype(self): ...
    @classproperty
    def _dtype(self): ...

class ComplexDoubleStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self): ...
    @classproperty
    def _dtype(self): ...

class ComplexFloatStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self): ...
    @classproperty
    def _dtype(self): ...

class _WrappedTritonKernel:
    """Just a simple wrapper to store some metadata for testing purposes."""
    kernel: Incomplete
    kernel_invoked: bool
    def __init__(self, kernel) -> None: ...
    def __call__(self, *args, **kwargs): ...

# Names in __all__ with no definition:
#   BFloat16Tensor
#   BoolTensor
#   ByteTensor
#   CUDAPluggableAllocator
#   CharTensor
#   DoubleTensor
#   FloatTensor
#   HalfTensor
#   IntTensor
#   LongTensor
#   MemPool
#   ShortTensor
#   caching_allocator_alloc
#   caching_allocator_delete
#   caching_allocator_enable
#   change_current_allocator
#   empty_cache
#   get_allocator_backend
#   get_per_process_memory_fraction
#   get_rng_state
#   get_rng_state_all
#   graphs
#   host_memory_stats
#   host_memory_stats_as_nested_dict
#   initial_seed
#   list_gpu_processes
#   manual_seed
#   manual_seed_all
#   max_memory_allocated
#   max_memory_cached
#   max_memory_reserved
#   mem_get_info
#   memory
#   memory_allocated
#   memory_cached
#   memory_reserved
#   memory_snapshot
#   memory_stats
#   memory_stats_as_nested_dict
#   memory_summary
#   nccl
#   random
#   reset_accumulated_host_memory_stats
#   reset_accumulated_memory_stats
#   reset_max_memory_allocated
#   reset_max_memory_cached
#   reset_peak_host_memory_stats
#   reset_peak_memory_stats
#   seed
#   seed_all
#   set_per_process_memory_fraction
#   set_rng_state
#   set_rng_state_all
#   streams
#   use_mem_pool
