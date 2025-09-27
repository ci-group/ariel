import dataclasses
import functools
import queue
import torch
from . import config as config
from .runtime.benchmarking import benchmarker as benchmarker
from .virtualized import V as V
from _typeshed import Incomplete
from collections.abc import Iterable, Sequence
from ctypes import CDLL
from torch._dynamo.device_interface import get_interface_for_device as get_interface_for_device
from torch._dynamo.testing import rand_strided as rand_strided
from torch._inductor import ir as ir
from torch._inductor.codecache import CUDACodeCache as CUDACodeCache, CppCodeCache as CppCodeCache, DLLWrapper as DLLWrapper, PyCodeCache as PyCodeCache, get_hash as get_hash
from torch._inductor.select_algorithm import TritonTemplateCaller as TritonTemplateCaller
from torch._inductor.utils import get_gpu_type as get_gpu_type, get_ld_library_path as get_ld_library_path, is_gpu as is_gpu
from torch._logging import getArtifactLogger as getArtifactLogger
from torch.utils._ordered_set import OrderedSet as OrderedSet
from types import ModuleType
from typing import Any, Callable, IO

CUDA_VISIBLE_DEVICES: str
autotuning_log: Incomplete

class NonzeroWorkspaceNotSupportedError(Exception): ...

class TuningProcess:
    """
    Class to launch and interact with a benchmarking subprocess.
    """
    @staticmethod
    def process_main(read_pipe: IO[bytes], write_pipe: IO[bytes]) -> None:
        """
        Entry point for the child process.
        """
    @staticmethod
    def send(obj: Any, write_pipe: IO[bytes]) -> None: ...
    @staticmethod
    def recv(read_pipe: IO[bytes]) -> Any: ...
    device: Incomplete
    def __init__(self, device: int | None) -> None: ...
    write_pipe: Incomplete
    read_pipe: Incomplete
    selector: Incomplete
    process: Incomplete
    running: bool
    def start(self) -> None:
        """
        Start the benchmarking subprocess.
        """
    def alive(self) -> bool:
        """
        True if the subprocess is still running.
        """
    def put(self, req: Any) -> None:
        """
        Push a work item to the child process.
        """
    def get(self, timeout: float = 120.0) -> Any:
        """
        Get a response from the child process. Raises TimeoutError on timeout;
        raises EOFError if the subprocess crashes.
        """
    def shutdown(self, wait: bool = True) -> None:
        """
        Signal the child process to shut down gracefully.
        """
    def wait(self) -> None:
        """
        Wait for the child process to exit.
        """
    def close(self) -> None:
        """
        Close resources.
        """
    def kill(self) -> None:
        """
        Send a SIGKILL to the child process.
        """

class TuningProcessPool:
    """
    Maintains a pool of TuningProcesses to benchmark kernels in parallel
    across devices. By default, we create one TuningProcess per device and
    set the sub-process environment to make only that device visible.
    """
    processes: Incomplete
    process_queue: queue.Queue[TuningProcess]
    executor: Incomplete
    def __init__(self) -> None:
        """
        Start the child processes.
        """
    @staticmethod
    def get_device_list() -> Sequence[int | None]:
        """
        Gather the list of devices to be used in the pool.
        """
    def shutdown(self) -> None:
        """
        Signal all child processes to exit.
        """
    def target(self, choice: TritonTemplateCaller) -> float:
        """
        Entry point for the thread-pool helper threads: Wait for an open TuningProcess,
        remove it from the queue, execute the benchmark in that subprocess, and return
        the TuningProcess to the queue.
        """
    def benchmark(self, choices: list[TritonTemplateCaller]) -> dict[TritonTemplateCaller, float]:
        """
        Benchmark each choice in a separate process.
        """
LayoutOrBuffer = ir.Layout | ir.Buffer

@dataclasses.dataclass
class TensorMeta:
    device: torch.device
    dtype: torch.dtype
    sizes: torch._prims_common.ShapeType
    strides: torch._prims_common.StrideType
    offset: int
    name: str | None = ...
    @classmethod
    def from_irnodes(cls, irnodes: LayoutOrBuffer | Sequence[LayoutOrBuffer]) -> TensorMeta | list[TensorMeta]: ...
    def to_tensor(self) -> torch.Tensor: ...

@dataclasses.dataclass
class BenchmarkRequest:
    """
    Only handle triton template benchmark for now. The extern kernel benchmark
    can be done inside the same process since they usually don't cause crash.

    Important: Instances of this class and subclasses have to be serializable
    across process boundaries. Do not put CUDA Tensors in here!
    """
    kernel_name = ...
    input_tensor_meta = ...
    output_tensor_meta = ...
    extra_args = ...
    def __init__(self, kernel_name: str, input_tensor_meta: TensorMeta | list[TensorMeta], output_tensor_meta: TensorMeta | list[TensorMeta], extra_args: Iterable[Any]) -> None: ...
    def make_run_fn(self, *input_tensors: torch.Tensor, out: torch.Tensor) -> Callable[[], None]: ...
    def cleanup_run_fn(self) -> None: ...
    def do_bench(self, fn, *input_tensors: torch.Tensor, out: torch.Tensor | None = None) -> float: ...
    def benchmark(self, *input_tensors: torch.Tensor, out: torch.Tensor | None = None) -> float: ...

class _TestBenchmarkRequest(BenchmarkRequest):
    """
    Supports unit testing. Defined in this file instead of the test file so the
    TuningProcess sub-process can unpickle these objects.
    """
    result: Incomplete
    device: Incomplete
    sleep: Incomplete
    exc: Incomplete
    crash: Incomplete
    def __init__(self, result: float = 0.0, device: int | None = None, sleep: float | None = None, exc: Exception | None = None, crash: bool = False) -> None: ...
    def benchmark(self, *input_tensors: torch.Tensor, out: torch.Tensor | None = None) -> float: ...

class GPUDeviceBenchmarkMixin:
    def do_bench(self, fn, *input_tensors: torch.Tensor, out: torch.Tensor | None = None) -> float: ...

class CPUDeviceBenchmarkMixin:
    def do_bench(self, fn, *input_tensors: torch.Tensor, out: torch.Tensor | None = None) -> float: ...

class TritonBenchmarkRequest(BenchmarkRequest):
    module_path: Incomplete
    module_cache_key: Incomplete
    num_stages: Incomplete
    num_warps: Incomplete
    num_consumer_groups: Incomplete
    num_buffers_warp_spec: Incomplete
    matrix_instr_nonkdim: Incomplete
    waves_per_eu: Incomplete
    kpack: Incomplete
    def __init__(self, kernel_name: str, input_tensor_meta: TensorMeta | list[TensorMeta], output_tensor_meta: TensorMeta | list[TensorMeta], extra_args: Iterable[Any], module_path: str, module_cache_key: str, num_stages: int, num_warps: int, num_consumer_groups: int = 0, num_buffers_warp_spec: int = 0, matrix_instr_nonkdim: int = 0, waves_per_eu: int = 0, kpack: int = 0) -> None: ...
    def make_run_fn(self, *input_tensors: torch.Tensor, out: torch.Tensor) -> Callable[[], None]: ...
    def precompile(self) -> None: ...
    def __str__(self) -> str: ...

class TritonGPUBenchmarkRequest(GPUDeviceBenchmarkMixin, TritonBenchmarkRequest): ...
class TritonCPUBenchmarkRequest(CPUDeviceBenchmarkMixin, TritonBenchmarkRequest): ...

class CUDABenchmarkRequest(GPUDeviceBenchmarkMixin, BenchmarkRequest):
    """
    A class to handle CUDA (CUTLASS) benchmark requests. This class is for
    managing the lifecycle of a CUDA kernel benchmark, including compiling
    the source code, managing workspace memory, and executing the kernel.

    Important: Instances of this class have to be serializable across
    process boundaries. Do not put CUDA Tensors in here!
    """
    source_code: Incomplete
    workspace_size: int
    workspace: torch.Tensor | None
    DLL: DLLWrapper | None
    _workspace_size_updated: bool
    hash_key: str
    source_file: str
    def __init__(self, kernel_name: str, input_tensor_meta: TensorMeta | list[TensorMeta], output_tensor_meta: TensorMeta | list[TensorMeta], extra_args: Iterable[Any], source_code: str) -> None: ...
    def precompile(self) -> None:
        """
        Precompile the CUDA source code to populate the CUDACodeCache.
        This may happen in a separate thread pool.
        """
    def make_run_fn(self, *input_tensors: torch.Tensor, out: torch.Tensor) -> Callable[[], None]:
        """
        Create a function to run the CUDA kernel with the given input and output tensors.
        """
    def update_workspace_size(self) -> None: ...
    def ensure_dll_loaded(self) -> None: ...
    def cleanup_run_fn(self) -> None: ...
    def __str__(self) -> str: ...

class CppBenchmarkRequest(CPUDeviceBenchmarkMixin, BenchmarkRequest):
    source_code: Incomplete
    hash_key: Incomplete
    DLL: CDLL | ModuleType | None
    def __init__(self, kernel_name: str, input_tensor_meta: TensorMeta | list[TensorMeta], output_tensor_meta: TensorMeta | list[TensorMeta], extra_args: Iterable[Any], source_code: str) -> None: ...
    def precompile(self) -> None: ...
    def make_run_fn(self, *input_tensors: torch.Tensor, out: torch.Tensor) -> Callable[[], None]: ...
    def cleanup_run_fn(self) -> None: ...
    def __str__(self) -> str: ...

@functools.cache
def get_tuning_process_pool() -> TuningProcessPool: ...
def benchmark_in_sub_process(choices: list[TritonTemplateCaller]) -> dict[TritonTemplateCaller, float]:
    """
    Do benchmarking in a subprocess and return the perf number (latency).
    """
