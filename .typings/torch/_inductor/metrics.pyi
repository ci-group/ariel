import dataclasses
from _typeshed import Incomplete
from dataclasses import dataclass
from functools import lru_cache
from torch._inductor import config as config
from torch._inductor.scheduler import BaseSchedulerNode as BaseSchedulerNode
from torch._inductor.utils import get_benchmark_name as get_benchmark_name
from torch.utils._ordered_set import OrderedSet as OrderedSet
from typing import Callable

generated_kernel_count: int
generated_cpp_vec_kernel_count: int
num_bytes_accessed: int
nodes_num_elem: list[tuple[BaseSchedulerNode, int]]
node_runtimes: list[tuple[BaseSchedulerNode, float]]
ir_nodes_pre_fusion: int
cpp_to_dtype_count: int

@dataclasses.dataclass
class CppOuterLoopFusedCount:
    inner_kernel_number: int
    local_buffer_number: int = ...

cpp_outer_loop_fused_inner_counts: list[CppOuterLoopFusedCount]
num_comprehensive_padding: int
num_matches_for_scatter_upon_const_tensor: int
num_loop_reordering: int
parallel_reduction_count: int

def reset() -> None: ...

@dataclass
class CachedMetricsDeltas:
    """
    The subset of metrics we want update across cache hits, e.g., the
    FxGraphCache.
    """
    generated_kernel_count: int
    generated_cpp_vec_kernel_count: int
    ir_nodes_pre_fusion: int
    cpp_to_dtype_count: int
    num_bytes_accessed: int
    num_matches_for_scatter_upon_const_tensor: int

def get_metric_fields() -> list[str]: ...

class CachedMetricsHelper:
    """
    A helper class to help calculate and apply counter deltas for those
    metrics we want to save with cache entries (e.g., FxGraphCache) and
    apply on a cache hit.
    """
    cached_metrics: Incomplete
    def __init__(self) -> None: ...
    def get_deltas(self) -> CachedMetricsDeltas: ...
    @staticmethod
    def apply_deltas(delta: CachedMetricsDeltas) -> None: ...

REGISTERED_METRIC_TABLES: dict[str, MetricTable]

@dataclass
class MetricTable:
    table_name: str
    column_names: list[str]
    num_rows_added: int = ...
    def add_row(self, row_fn: Callable[[], dict[str, str | float | None]]) -> None: ...
    def output_filename(self) -> str: ...
    def write_header(self) -> None: ...
    def _write_row(self, row: list[str]) -> None: ...
    @staticmethod
    def register_table(name: str, column_names: list[str]) -> None: ...

def _parse_kernel_fn_code(kernel_module_code: str) -> str:
    """
    The kernel_module_code is the python module that contains kernel function code.
    kernel function is the proper triton kernel function annotated with
    @triton.jit
    """
def _parse_kernel_line_of_code(proper_kernel_fn_code: str) -> int:
    """
    Return the line of code for the kernel excluding the decorators.
    """
def _parse_size_hints(kernel_module_code: str, kernel_category: str) -> str | None: ...
def _parse_reduction_hint(kernel_category: str, kernel_module_code: str) -> str | None: ...
def _count_pattern(proper_kernel_fn_code: str, pattern: str) -> int: ...
def _count_args(proper_kernel_fn_code: str) -> int: ...
def _parse_proper_kernel_fn_code(kernel_fn_code: str) -> str:
    """
    Skip decorators.
    """
def _parse_numel(proper_kernel_fn_code: str, numel_arg_name: str) -> int | None: ...
def _parse_kernel_args_num_gb(kernel_fn_code: str, kernel_category: str) -> float | None:
    """
    inductor meta looks like:
        inductor_meta={... 'mutated_arg_names': [], 'no_x_dim': False, 'kernel_num_gb': 2.0},
    """
def log_kernel_metadata(kernel_name: str, kernel_path: str, kernel_module_code: str) -> None:
    """
    An utility to log kernel metadata. We may parse metadata from kernel source code here.

    It's fine to parse the generated kernel code here since the logging is
    disabled by default. It would hurt compilation time.
    """
def purge_old_log_files() -> None:
    """
    Purge the old log file at the beginning when the benchmark script runs.
    Should do it in the parent process rather than the child processes running
    each individual model.
    """
def enabled_metric_tables() -> OrderedSet[str]: ...
@lru_cache
def enabled_metric_tables_impl(config_str: str) -> OrderedSet[str]: ...
def is_metric_table_enabled(name: str) -> bool: ...
def get_metric_table(name: str) -> MetricTable: ...
