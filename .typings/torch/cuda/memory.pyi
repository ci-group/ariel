import contextlib
import torch
from _typeshed import Incomplete
from torch._C import _MemPool, _cuda_CUDAAllocator
from torch.types import Device
from typing import Any

__all__ = ['caching_allocator_alloc', 'caching_allocator_delete', 'caching_allocator_enable', 'get_per_process_memory_fraction', 'set_per_process_memory_fraction', 'empty_cache', 'memory_stats', 'memory_stats_as_nested_dict', 'reset_accumulated_memory_stats', 'reset_peak_memory_stats', 'reset_max_memory_allocated', 'reset_max_memory_cached', 'host_memory_stats', 'host_memory_stats_as_nested_dict', 'reset_accumulated_host_memory_stats', 'reset_peak_host_memory_stats', 'memory_allocated', 'max_memory_allocated', 'memory_reserved', 'max_memory_reserved', 'memory_cached', 'max_memory_cached', 'memory_snapshot', 'memory_summary', 'list_gpu_processes', 'mem_get_info', 'get_allocator_backend', 'CUDAPluggableAllocator', 'change_current_allocator', 'MemPool', 'use_mem_pool']

def caching_allocator_alloc(size, device: Device = None, stream=None):
    """Perform a memory allocation using the CUDA memory allocator.

    Memory is allocated for a given device and a stream, this
    function is intended to be used for interoperability with other
    frameworks. Allocated memory is released through
    :func:`~torch.cuda.caching_allocator_delete`.

    Args:
        size (int): number of bytes to be allocated.
        device (torch.device or int, optional): selected device. If it is
            ``None`` the default CUDA device is used.
        stream (torch.cuda.Stream or int, optional): selected stream. If is ``None`` then
            the default stream for the selected device is used.

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
def caching_allocator_delete(mem_ptr) -> None:
    """Delete memory allocated using the CUDA memory allocator.

    Memory allocated with :func:`~torch.cuda.caching_allocator_alloc`.
    is freed here. The associated device and stream are tracked inside
    the allocator.

    Args:
        mem_ptr (int): memory address to be freed by the allocator.

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
def caching_allocator_enable(value: bool = True) -> None:
    """Enable or disable the CUDA memory allocator. On by default."""
def set_per_process_memory_fraction(fraction, device: Device = None) -> None:
    """Set memory fraction for a process.

    The fraction is used to limit an caching allocator to allocated memory on a CUDA device.
    The allowed value equals the total visible memory multiplied fraction.
    If trying to allocate more than the allowed value in a process, will raise an out of
    memory error in allocator.

    Args:
        fraction(float): Range: 0~1. Allowed memory equals total_memory * fraction.
        device (torch.device or int, optional): selected device. If it is
            ``None`` the default CUDA device is used.
    .. note::
        In general, the total available free memory is less than the total capacity.
    """
def get_per_process_memory_fraction(device: Device = None) -> float:
    """Get memory fraction for a process.

    Args:
        device (torch.device or int, optional): selected device. If it is
            ``None`` the default CUDA device is used.
    Returns:
        memory fraction, in range 0~1. Allowed memory equals total_memory * fraction.
    """
def empty_cache() -> None:
    """Release all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other GPU application and visible in
    `nvidia-smi`.

    .. note::
        :func:`~torch.cuda.empty_cache` doesn't increase the amount of GPU
        memory available for PyTorch. However, it may help reduce fragmentation
        of GPU memory in certain cases. See :ref:`cuda-memory-management` for
        more details about GPU memory management.
    """
def memory_stats(device: Device = None) -> dict[str, Any]:
    '''Return a dictionary of CUDA memory allocator statistics for a given device.

    The return value of this function is a dictionary of statistics, each of
    which is a non-negative integer.

    Core statistics:

    - ``"allocated.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of allocation requests received by the memory allocator.
    - ``"allocated_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of allocated memory.
    - ``"segment.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of reserved segments from ``cudaMalloc()``.
    - ``"reserved_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of reserved memory.
    - ``"active.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of active memory blocks.
    - ``"active_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of active memory.
    - ``"inactive_split.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of inactive, non-releasable memory blocks.
    - ``"inactive_split_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of inactive, non-releasable memory.

    For these core statistics, values are broken down as follows.

    Pool type:

    - ``all``: combined statistics across all memory pools.
    - ``large_pool``: statistics for the large allocation pool
      (as of October 2019, for size >= 1MB allocations).
    - ``small_pool``: statistics for the small allocation pool
      (as of October 2019, for size < 1MB allocations).

    Metric type:

    - ``current``: current value of this metric.
    - ``peak``: maximum value of this metric.
    - ``allocated``: historical total increase in this metric.
    - ``freed``: historical total decrease in this metric.

    In addition to the core statistics, we also provide some simple event
    counters:

    - ``"num_alloc_retries"``: number of failed ``cudaMalloc`` calls that
      result in a cache flush and retry.
    - ``"num_ooms"``: number of out-of-memory errors thrown.
    - ``"num_sync_all_streams"``: number of ``synchronize_and_free_events`` calls.
    - ``"num_device_alloc"``: number of CUDA allocation calls. This includes both
      cuMemMap and cudaMalloc.
    - ``"num_device_free"``: number of CUDA free calls. This includes both cuMemUnmap
      and cudaFree.

    The caching allocator can be configured via ENV to not split blocks larger than a
    defined size (see Memory Management section of the Cuda Semantics documentation).
    This helps avoid memory fragmentation but may have a performance
    penalty. Additional outputs to assist with tuning and evaluating impact:

    - ``"max_split_size"``: blocks above this size will not be split.
    - ``"oversize_allocations.{current,peak,allocated,freed}"``:
      number of over-size allocation requests received by the memory allocator.
    - ``"oversize_segments.{current,peak,allocated,freed}"``:
      number of over-size reserved segments from ``cudaMalloc()``.

    The caching allocator can be configured via ENV to round memory allocations in order
    to reduce fragmentation. Sometimes the overhead from rounding can be higher than
    the fragmentation it helps reduce. The following stat can be used to check if
    rounding adds too much overhead:

    - ``"requested_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      memory requested by client code, compare this with allocated_bytes to check if
      allocation rounding adds too much overhead.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistics for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.

    .. note::
        With :ref:`backend:cudaMallocAsync<cuda-memory-envvars>`, some stats are not
        meaningful, and are always reported as zero.
    '''
def memory_stats_as_nested_dict(device: Device = None) -> dict[str, Any]:
    """Return the result of :func:`~torch.cuda.memory_stats` as a nested dictionary."""
def reset_accumulated_memory_stats(device: Device = None) -> None:
    '''Reset the "accumulated" (historical) stats tracked by the CUDA memory allocator.

    See :func:`~torch.cuda.memory_stats` for details. Accumulated stats correspond to
    the `"allocated"` and `"freed"` keys in each individual stat dict, as well as
    `"num_alloc_retries"` and `"num_ooms"`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    '''
def reset_peak_memory_stats(device: Device = None) -> None:
    '''Reset the "peak" stats tracked by the CUDA memory allocator.

    See :func:`~torch.cuda.memory_stats` for details. Peak stats correspond to the
    `"peak"` key in each individual stat dict.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    '''
def host_memory_stats() -> dict[str, Any]:
    '''Return a dictionary of CUDA memory allocator statistics for a given device.

     The return value of this function is a dictionary of statistics, each of
     which is a non-negative integer.

     Core statistics:

     - ``"allocated.{current,peak,allocated,freed}"``:
       number of allocation requests received by the memory allocator.
     - ``"allocated_bytes.{current,peak,allocated,freed}"``:
       amount of allocated memory.
     - ``"segment.{current,peak,allocated,freed}"``:
       number of reserved segments from ``cudaMalloc()``.
     - ``"reserved_bytes.{current,peak,allocated,freed}"``:
       amount of reserved memory.

     For these core statistics, values are broken down as follows.

     Metric type:

     - ``current``: current value of this metric.
     - ``peak``: maximum value of this metric.
     - ``allocated``: historical total increase in this metric.
     - ``freed``: historical total decrease in this metric.

     In addition to the core statistics, we also provide some simple event
     counters:

     - ``"num_host_alloc"``: number of CUDA allocation calls. This includes both
       cudaHostAlloc and cudaHostRegister.
     - ``"num_host_free"``: number of CUDA free calls. This includes both cudaHostFree
       and cudaHostUnregister.

     Finally, we also provide some simple timing counters:

     - ``"host_alloc_time.{total,max,min,count,avg}"``:
       timing of allocation requests going through CUDA calls.
     - ``"host_free_time.{total,max,min,count,avg}"``:
       timing of free requests going through CUDA calls.

    For these timing statistics, values are broken down as follows.

     Metric type:

     - ``total``: total time spent.
     - ``max``: maximum value per call.
     - ``min``: minimum value per call.
     - ``count``: number of times it was called.
     - ``avg``: average time per call.
    '''
def host_memory_stats_as_nested_dict() -> dict[str, Any]:
    """Return the result of :func:`~torch.cuda.host_memory_stats` as a nested dictionary."""
def reset_accumulated_host_memory_stats() -> None:
    '''Reset the "accumulated" (historical) stats tracked by the host memory allocator.

    See :func:`~torch.cuda.host_memory_stats` for details. Accumulated stats correspond to
    the `"allocated"` and `"freed"` keys in each individual stat dict.
    '''
def reset_peak_host_memory_stats() -> None:
    '''Reset the "peak" stats tracked by the host memory allocator.

    See :func:`~torch.cuda.host_memory_stats` for details. Peak stats correspond to the
    `"peak"` key in each individual stat dict.
    '''
def reset_max_memory_allocated(device: Device = None) -> None:
    """Reset the starting point in tracking maximum GPU memory occupied by tensors for a given device.

    See :func:`~torch.cuda.max_memory_allocated` for details.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. warning::
        This function now calls :func:`~torch.cuda.reset_peak_memory_stats`, which resets
        /all/ peak memory stats.

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
def reset_max_memory_cached(device: Device = None) -> None:
    """Reset the starting point in tracking maximum GPU memory managed by the caching allocator for a given device.

    See :func:`~torch.cuda.max_memory_cached` for details.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. warning::
        This function now calls :func:`~torch.cuda.reset_peak_memory_stats`, which resets
        /all/ peak memory stats.

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
def memory_allocated(device: Device = None) -> int:
    """Return the current GPU memory occupied by tensors in bytes for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        This is likely less than the amount shown in `nvidia-smi` since some
        unused memory can be held by the caching allocator and some context
        needs to be created on GPU. See :ref:`cuda-memory-management` for more
        details about GPU memory management.
    """
def max_memory_allocated(device: Device = None) -> int:
    """Return the maximum GPU memory occupied by tensors in bytes for a given device.

    By default, this returns the peak allocated memory since the beginning of
    this program. :func:`~torch.cuda.reset_peak_memory_stats` can be used to
    reset the starting point in tracking this metric. For example, these two
    functions can measure the peak allocated memory usage of each iteration in a
    training loop.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
def memory_reserved(device: Device = None) -> int:
    """Return the current GPU memory managed by the caching allocator in bytes for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
def max_memory_reserved(device: Device = None) -> int:
    """Return the maximum GPU memory managed by the caching allocator in bytes for a given device.

    By default, this returns the peak cached memory since the beginning of this
    program. :func:`~torch.cuda.reset_peak_memory_stats` can be used to reset
    the starting point in tracking this metric. For example, these two functions
    can measure the peak cached memory amount of each iteration in a training
    loop.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
def memory_cached(device: Device = None) -> int:
    """Deprecated; see :func:`~torch.cuda.memory_reserved`."""
def max_memory_cached(device: Device = None) -> int:
    """Deprecated; see :func:`~torch.cuda.max_memory_reserved`."""
def memory_snapshot(mempool_id=None):
    """Return a snapshot of the CUDA memory allocator state across all devices.

    Interpreting the output of this function requires familiarity with the
    memory allocator internals.

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
def memory_summary(device: Device = None, abbreviated: bool = False) -> str:
    """Return a human-readable printout of the current memory allocator statistics for a given device.

    This can be useful to display periodically during training, or when
    handling out-of-memory exceptions.

    Args:
        device (torch.device or int, optional): selected device. Returns
            printout for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).
        abbreviated (bool, optional): whether to return an abbreviated summary
            (default: False).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
def list_gpu_processes(device: Device = None) -> str:
    """Return a human-readable printout of the running processes and their GPU memory use for a given device.

    This can be useful to display periodically during training, or when
    handling out-of-memory exceptions.

    Args:
        device (torch.device or int, optional): selected device. Returns
            printout for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).
    """
def mem_get_info(device: Device = None) -> tuple[int, int]:
    """Return the global free and total GPU memory for a given device using cudaMemGetInfo.

    Args:
        device (torch.device or int or str, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default) or if the device index is not specified.

    .. note::
        See :ref:`cuda-memory-management` for more
        details about GPU memory management.
    """
def get_allocator_backend() -> str:
    """Return a string describing the active allocator backend as set by
    ``PYTORCH_CUDA_ALLOC_CONF``. Currently available backends are
    ``native`` (PyTorch's native caching allocator) and `cudaMallocAsync``
    (CUDA's built-in asynchronous allocator).

    .. note::
        See :ref:`cuda-memory-management` for details on choosing the allocator backend.
    """

class _CUDAAllocator:
    """Wrapper over internal CUDA memory allocators."""
    _allocator: Incomplete
    def __init__(self, allocator: torch._C._cuda_CUDAAllocator) -> None: ...
    def allocator(self): ...

class CUDAPluggableAllocator(_CUDAAllocator):
    """CUDA memory allocator loaded from a so file."""
    _allocator: Incomplete
    def __init__(self, path_to_so_file: str, alloc_fn_name: str, free_fn_name: str) -> None:
        """Memory allocators are compiled in .so files and loaded dynamically using ctypes.

        To change the active allocator use the :func:`torch.memory.cuda.change_current_allocator` function.

        Args:
            path_to_so_file(str): Path in the filesystem to the `.so` file containing
                the allocator functions
            alloc_fn_name(str): Name of the function to perform the memory allocation
                in the so file. The signature must be:
                void* alloc_fn_name(ssize_t size, int device, cudaStream_t stream);
            free_fn_name(str): Name of the function to perform the memory release
                in the so file. The signature must be:
                void free_fn_name(void* ptr, size_t size, cudaStream_t stream);

        .. warning::
            This is currently supported only in unix OSs

        .. note::
            See :ref:`cuda-memory-management` for details on creating and using a custom allocator
        """

def change_current_allocator(allocator: _CUDAAllocator) -> None:
    """Change the currently used memory allocator to be the one provided.

    If the current allocator has already been used/initialized, this function will error.


    Args:
        allocator (torch.cuda.memory._CUDAAllocator): allocator to be set as the active one.
    .. note::
        See :ref:`cuda-memory-management` for details on creating and using a custom allocator
    """

class MemPool(_MemPool):
    """MemPool represents a pool of memory in a caching allocator. Currently,
    it's just the ID of the pool object maintained in the CUDACachingAllocator.

    Args:
        allocator(torch._C._cuda_CUDAAllocator, optional): a
            torch._C._cuda_CUDAAllocator object that can be used to
            define how memory gets allocated in the pool. If :attr:`allocator`
            is ``None`` (default), memory allocation follows the default/
            current configuration of the CUDACachingAllocator.
        use_on_oom(bool): a bool that indicates if this pool can be used
            as a last resort if a memory allocation outside of the pool fails due
            to Out Of Memory. This is False by default.
        symmetric(bool): a bool that indicates if this pool is symmetrical
            across ranks. This is False by default.
    """
    def __init__(self, allocator: _cuda_CUDAAllocator | None = None, use_on_oom: bool = False, symmetric: bool = False) -> None: ...
    @property
    def id(self) -> tuple[int, int]:
        """Returns the ID of this pool as a tuple of two ints."""
    @property
    def is_symmetric(self) -> bool:
        """Returns whether this pool is used for NCCL's symmetric memory."""
    @property
    def allocator(self) -> _cuda_CUDAAllocator | None:
        """Returns the allocator this MemPool routes allocations to."""
    def use_count(self) -> int:
        """Returns the reference count of this pool."""
    def snapshot(self):
        """Return a snapshot of the CUDA memory allocator pool state across all
        devices.

        Interpreting the output of this function requires familiarity with the
        memory allocator internals.

        .. note::
            See :ref:`cuda-memory-management` for more details about GPU memory
            management.
        """

@contextlib.contextmanager
def use_mem_pool(pool: MemPool, device: Device = None):
    """A context manager that routes allocations to a given pool.

    Args:
        pool(torch.cuda.MemPool): a MemPool object to be made active so that
            allocations route to this pool.
        device (torch.device or int, optional): selected device. Uses MemPool on
            the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        This context manager makes only current thread's allocations route to
        the given pool. If a new thread is spawned inside the context manager
        (e.g. by calling backward) the allocations in that thread will not
        route to the given pool.
    """
