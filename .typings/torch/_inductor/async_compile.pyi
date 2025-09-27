from _typeshed import Incomplete
from concurrent.futures import ThreadPoolExecutor
from torch._dynamo.device_interface import get_registered_device_interfaces as get_registered_device_interfaces
from torch._dynamo.utils import counters as counters, dynamo_timed as dynamo_timed, get_metrics_context as get_metrics_context, set_feature_use as set_feature_use
from torch._inductor import config as config
from torch._inductor.codecache import CUDACodeCache as CUDACodeCache, CodeCacheFuture as CodeCacheFuture, CppCodeCache as CppCodeCache, CppPythonBindingsCodeCache as CppPythonBindingsCodeCache, HalideCodeCache as HalideCodeCache, LambdaFuture as LambdaFuture, ROCmCodeCache as ROCmCodeCache, StaticAutotunerFuture as StaticAutotunerFuture, _load_triton_kernel_from_source as _load_triton_kernel_from_source, code_hash as code_hash, torch_key as torch_key
from torch._inductor.compile_worker.subproc_pool import AnyPool as AnyPool, SubprocPool as SubprocPool
from torch._inductor.compile_worker.tracked_process_pool import TrackedProcessPoolExecutor as TrackedProcessPoolExecutor
from torch._inductor.compile_worker.utils import _async_compile_initializer as _async_compile_initializer
from torch._inductor.runtime.compile_tasks import _set_triton_ptxas_path as _set_triton_ptxas_path, _worker_compile_triton as _worker_compile_triton
from torch._inductor.runtime.hints import HalideMeta as HalideMeta
from torch._inductor.runtime.triton_heuristics import CachingAutotuner as CachingAutotuner
from torch._inductor.utils import clear_on_fresh_cache as clear_on_fresh_cache
from torch._inductor.virtualized import V as V
from torch.hub import _Faketqdm as _Faketqdm, tqdm as tqdm
from torch.utils._ordered_set import OrderedSet as OrderedSet
from torch.utils._triton import has_triton_package as has_triton_package
from typing import Any, Callable

_cumulative_compile_time: float
_t0: float | None
kernel_code_log: Incomplete
log: Incomplete
_triton_kernel_metrics: dict[str, dict[str, Any]] | None

def pre_fork_setup() -> None:
    """
    Setup that must be done prior to forking with a process pool.
    """
def caching_device_properties() -> None: ...
def _compile_start() -> None: ...
def _compile_end() -> None: ...
def _add_triton_kernel_info(kernel_name: str, info: dict[str, Any]): ...

_IS_WINDOWS: Incomplete
_pool_set: Incomplete

def shutdown_compile_workers() -> None:
    """Shut down all outstanding compile-worker pools."""
def after_fork() -> None:
    """Reset pools to initial state without shutting them down"""
def get_compile_threads() -> int:
    """
    Temporary for internal rollout. Assign config.compile_threads lazily and return it.
    TODO: remove after rollout.
    """

class CompiledTritonKernels:
    """
    In memory cache for storing compiled triton kernels.

    Each triton kernel is keyed by the hash of its source code. Each value stored
    in the cache is a return value of AsyncCompile.triton().

    Currently, the cache stores Future objects, but it should be generalizable for any kernels.
    """
    _cache: dict[str, CodeCacheFuture]
    @staticmethod
    def key(kernel_src: str):
        """
        Generates a cache key given a triton kernel's full source code.
        This source includes the inductor meta, compilation metadata, the kernel itself, etc.
        `kernel_src` should be the exact string passed to async_compile.triton()'s first argument.
        """
    @staticmethod
    def save(kernel_src: str, future: CodeCacheFuture):
        """
        Saves a compiled triton kernel to the cache.
        TODO: We store a LambdaFuture as that's the callable returned by async_compile.triton,
        but the real type we want to return here is actually an abstract triton kernel.

        TODO: Source code here is not just the kernel's source code, but also includes the inductor preamble, etc.
        so it could be less strict.
        """
    @staticmethod
    def get(kernel_src: str) -> CodeCacheFuture | None: ...
    @staticmethod
    def cache_clear() -> None: ...
    @staticmethod
    def remove_future(kernel_src: str) -> None: ...

class AsyncCompile:
    def __init__(self) -> None: ...
    @staticmethod
    def pool() -> ThreadPoolExecutor: ...
    @staticmethod
    def _get_ready():
        """No-op function to help mark when the subprocess pool is ready."""
    @staticmethod
    def process_pool() -> AnyPool: ...
    @classmethod
    def warm_pool(cls) -> None: ...
    @classmethod
    def submit(cls, task: Callable[..., Any]) -> Any: ...
    def use_process_pool(self): ...
    def triton(self, kernel_name: str, source_code: str, device_str: str = 'cuda'):
        """
        Async_compile.triton is more complicated than the other backends because
        we're trying to optimize compile time as much as possible for this hot callsite.

        First of all, the function is cached by CompiledTritonKernels; if there's a kernel
        already compiled, we grab it directly from the cache and return.

        Otherwise, if we have multiple compile threads, we kick off triton compilations on each
        worker process by giving it a kernel and source code to compile. The worker initializes
        a CachingAutotuner, runs triton compilation, and pickles the kernel back to us.
        We use TritonCompileResult to represent the objects being pickled back to us by each
        worker.

        Some maybe not obvious things that are pickled back to us:
        - Most of the time, we can avoid sending back CachingAutotuner.fn and other metadata
          and do not have to pay the cost of loading the triton kernel on the parent. But certain
          cases, like coordesc tuning and dynamic_scale_rblock, require us to reload the function
          in the parent lazily when we require it.
        - The AutotuneCache, if enabled, is constructed on each worker per triton config
          and pickled by to us via `CachingAutotuner.save_cache_hook`.
        """
    def multi_kernel(self, *args, **kwargs) -> Any: ...
    def cpp(self, source_code: str): ...
    def cpp_pybinding(self, argtypes: list[str], source_code: str): ...
    def cuda(self, source_code, dst_file_ext, aot_compile: bool = False): ...
    def rocm(self, source_code, dst_file_ext, aot_compile: bool = False): ...
    def halide(self, meta: HalideMeta, source_code: str): ...
    def wait(self, scope: dict[str, Any]) -> None: ...
    def _wait_futures(self, scope: dict[str, Any]) -> None: ...
