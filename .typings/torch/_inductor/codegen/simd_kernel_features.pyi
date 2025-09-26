import collections
import dataclasses
import functools
import sympy
import torch
import typing
from ...utils._ordered_set import OrderedSet as OrderedSet
from ...utils._sympy.functions import FloorDiv as FloorDiv, ModularIndexing as ModularIndexing
from ...utils._sympy.symbol import SymT as SymT, make_symbol as make_symbol
from ..dependencies import Dep as Dep, MemoryDep as MemoryDep, extract_loop_body_with_args as extract_loop_body_with_args
from ..runtime.hints import ReductionHint as ReductionHint
from ..scheduler import SchedulerNode as SchedulerNode
from ..utils import cache_on_self as cache_on_self
from ..virtualized import V as V
from _typeshed import Incomplete
from collections.abc import Iterable, Sequence
from torch._inductor.tiling_utils import CoalesceVarAnalysis as CoalesceVarAnalysis
from typing import Any

class NodeScheduleMarker:
    @staticmethod
    def only_nodes(it: Iterable[NodeScheduleEntry]) -> Iterable[SchedulerNode]: ...
    @staticmethod
    def is_reduction() -> bool: ...
NodeScheduleEntry = SchedulerNode | type[NodeScheduleMarker]

class DisableReduction(NodeScheduleMarker):
    """
    Marker to invoke `kernel.disable_reduction()`.  This closes a
    reduction loop and allows for pointwise ops to occur on the output
    of a reduction.
    """

class EnableReduction(NodeScheduleMarker):
    """
    Marker to end a DisableReduction block.
    """
    @staticmethod
    def filter(node_schedule: list[NodeScheduleEntry]) -> Iterable[SchedulerNode]:
        """
        Get the nodes from node_schedule skipping those in a
        DisableReduction block.
        """

class SIMDKernelFeatures:
    """
    An ordered schedule of nodes that will become a single kernel.
    """
    node_schedule: Incomplete
    numel: sympy.Expr
    reduction_numel: sympy.Expr
    _stats_cache: dict[tuple[sympy.Expr, ...], MemoryStats]
    coalesce_analysis: Incomplete
    def __init__(self, node_schedule: list[NodeScheduleEntry], numel: sympy.Expr, reduction_numel: sympy.Expr = ..., coalesce_analysis: CoalesceVarAnalysis | None = None) -> None: ...
    @cache_on_self
    def is_reduction(self) -> bool: ...
    @cache_on_self
    def scheduler_nodes(self) -> Iterable[SchedulerNode]: ...
    def reduction_nodes(self) -> list[SchedulerNode]: ...
    @cache_on_self
    def buf_accesses(self) -> dict[str, list[Dep]]:
        """only needed for config.benchmark_kernel"""
    @cache_on_self
    def op_counts(self) -> collections.Counter[str]: ...
    def contains_op(self, op_name: str) -> bool:
        """True if V.ops.{op_name} is used in node_schedule"""
    def get_mutations(self) -> OrderedSet[str]: ...
    @cache_on_self
    def select_index_dtype(self) -> torch.dtype: ...
    @cache_on_self
    def get_reduction_hint(self) -> ReductionHint: ...
    @cache_on_self
    def buffer_read_counts(self) -> dict[str, int]:
        """Counts how many times each buffer is read within the kernel"""
    def has_non_contiguous_pw_in_reduction_kernel(self) -> bool: ...
    @staticmethod
    def reduction_hint(node: Any) -> ReductionHint: ...
    def memory_stats(self, groups_dict: dict[str, sympy.Expr] | None = None) -> MemoryStats:
        """Analysis to generate features that can be used in heuristics"""

class MemoryEstimator:
    """
    Estimate various properties of the kernel for use in heuristics.
    We simulate the memory effects of CSE/buffer elimination in codegen.
    """
    kernel_sizes: tuple[sympy.Expr, ...]
    outside_loop: MemoryEstimate
    loops: list[MemoryEstimate]
    persistent: MemoryEstimate
    symbols: list[sympy.Symbol]
    features: Incomplete
    inside_reduction: Incomplete
    store_buffer_names: OrderedSet[str]
    must_keep_buffers: OrderedSet[str]
    num_reductions_dims: int
    groups: Incomplete
    def __init__(self, features: SIMDKernelFeatures, groups: Sequence[sympy.Expr]) -> None: ...
    def simulate_codegen(self) -> None: ...
    def remove_kernel_local(self) -> None: ...
    def scope(self, dep: MemoryDep) -> MemoryEstimate:
        """Determine how a read/write should be categorized"""
    def has_reduction_var(self, index: sympy.Expr) -> bool: ...
    def set_ranges(self, *lengths: list[list[sympy.Expr]]) -> list[list[sympy.Expr]]: ...
    @staticmethod
    def make_flat_range(sym: sympy.Symbol, numel: sympy.Expr, lengths: list[sympy.Expr]) -> list[sympy.Expr]: ...

@dataclasses.dataclass
class MemoryEstimate:
    """Tracks the memory usage of a single loop in the generated kernel"""
    reads: dict[str, OrderedSet[MemoryDep]] = dataclasses.field(default_factory=functools.partial(collections.defaultdict, OrderedSet))
    writes: dict[str, OrderedSet[MemoryDep]] = dataclasses.field(default_factory=functools.partial(collections.defaultdict, OrderedSet))
    def remove(self, name: str) -> None: ...
    def __bool__(self) -> bool: ...
    def __repr__(self) -> str: ...

@dataclasses.dataclass
class StatsForDim:
    """Memory usage stats for a block dimension in the generated kernel (different from user dimensions)"""
    count_per_thread_contiguous: int = ...
    count_per_thread_broadcast: int = ...
    count_per_thread_non_contiguous: int = ...
    bytes_per_thread_contiguous: int = ...
    bytes_per_thread_broadcast: int = ...
    bytes_per_thread_non_contiguous: int = ...
    bytes_contiguous_or_broadcast: sympy.Expr = ...
    bytes_non_contiguous: sympy.Expr = ...
    def __add__(self, other: typing.Self) -> StatsForDim: ...
    @property
    def count_per_thread(self) -> int: ...
    @property
    def bytes_per_thread(self) -> int: ...
    @property
    def bytes(self) -> sympy.Expr: ...
    @property
    def contiguous_score(self) -> float: ...

@dataclasses.dataclass
class StatsForLoop:
    """Memory usage stats for single loop in the generated kernel"""
    count_per_thread: int = ...
    bytes_per_thread: int = ...
    def __add__(self, other: typing.Self) -> StatsForLoop: ...

@dataclasses.dataclass
class StatsForReadsOrWrites:
    """Memory usage stats that are collected for reads/writes/both"""
    dim: list[StatsForDim]
    loop: list[StatsForLoop]
    bytes_contiguous_or_broadcast: sympy.Expr = ...
    bytes_non_contiguous: sympy.Expr = ...
    def __add__(self, other: typing.Self) -> StatsForReadsOrWrites: ...
    @property
    def count_per_thread(self) -> int: ...
    @property
    def bytes_per_thread(self) -> int: ...
    @property
    def bytes(self) -> sympy.Expr: ...
    @classmethod
    def compute(cls, loop_deps: list[dict[str, OrderedSet[MemoryDep]]], index_symbols: list[sympy.Symbol]) -> typing.Self: ...

@dataclasses.dataclass
class StatsForKernelType:
    """Memory usage stats that are collected for both persistent and looped kernels"""
    reads: StatsForReadsOrWrites
    writes: StatsForReadsOrWrites
    memory: StatsForReadsOrWrites
    @classmethod
    def compute(cls, loops: list[MemoryEstimate], estimator: MemoryEstimator) -> typing.Self: ...

@dataclasses.dataclass
class MemoryStats:
    """Memory usage stats collected for each generated kernel"""
    persistent: StatsForKernelType
    looped: StatsForKernelType
    def get(self, persistent: bool) -> StatsForKernelType: ...
    @classmethod
    def compute(cls, estimator: MemoryEstimator) -> typing.Self: ...
