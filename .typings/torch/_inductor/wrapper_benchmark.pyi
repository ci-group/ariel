import argparse
import dataclasses
import torch
from .runtime.benchmarking import benchmarker as benchmarker
from .runtime.runtime_utils import create_bandwidth_info_str as create_bandwidth_info_str, get_num_bytes as get_num_bytes
from _typeshed import Incomplete
from torch.autograd import DeviceType as DeviceType
from torch.utils._ordered_set import OrderedSet as OrderedSet
from types import ModuleType
from typing import Any, Protocol

class BenchmarkCallableType(Protocol):
    def __call__(self, times: int, repeat: int) -> float: ...

_kernel_category_choices: Incomplete

def get_kernel_category_by_source_code(src_code: str) -> str:
    """
    Similar to get_kernel_category but use the source code. Call this API
    if we have not compile the src_code to module yet.
    """
def get_kernel_category(kernel_mod: ModuleType) -> str:
    """
    Given the module defining a triton kernel, return the category of the kernel.
    Category can be one of:
    - pointwise
    - reduction
    - persistent_reduction

    Currently we simply decide the category depending on what decorator is imported
    by the kernel.
    """
def get_triton_kernel(mod: ModuleType): ...
def benchmark_all_kernels(benchmark_name: str, benchmark_all_configs: dict[Any, Any] | None) -> None:
    """
    An experimental API used only when config.benchmark_kernel is true.

    Run the kernel benchmarks for all the kernels cached in PyCodeCache.
    Used in the compiled modules.

    Put this method here rather than codegen it for convenience since its implementation
    does not change based on different graph modules being compiled.
    """

@dataclasses.dataclass
class ProfileEvent:
    category: str
    key: str
    self_device_time_ms: float
    count: float

def parse_profile_event_list(benchmark_name: str, event_list: torch.autograd.profiler_util.EventList, wall_time_ms: float, nruns: int, device_name: str) -> None: ...
def perf_profile(wall_time_ms: float, times: int, repeat: int, benchmark_name: str, benchmark_compiled_module_fn: BenchmarkCallableType) -> None: ...
def ncu_analyzer(benchmark_name: str, benchmark_compiled_module_fn: BenchmarkCallableType, args: argparse.Namespace) -> None: ...
def collect_memory_snapshot(benchmark_compiled_module_fn: BenchmarkCallableType) -> None: ...
@torch.compiler.disable
def compiled_module_main(benchmark_name: str, benchmark_compiled_module_fn: BenchmarkCallableType) -> None:
    """
    This is the function called in __main__ block of a compiled module.
    """
