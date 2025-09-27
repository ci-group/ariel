import abc
import dataclasses
import sympy
import torch
from .. import config as config
from ..utils import CachedMethod as CachedMethod, IndentedBuffer as IndentedBuffer, _align as _align, align as align, cache_on_self as cache_on_self
from ..virtualized import V as V
from .wrapper import AllocateLine as AllocateLine, BufferLike as BufferLike, FreeIfNotReusedLine as FreeIfNotReusedLine, MemoryPlanningLine as MemoryPlanningLine, NullLine as NullLine, ReuseLine as ReuseLine
from _typeshed import Incomplete
from collections.abc import Iterable
from torch.utils._ordered_set import OrderedSet as OrderedSet
from typing import Any, Protocol

@dataclasses.dataclass
class LiveRange:
    """
    A range where a given tensor is live.  Begin and end are both counters
    representing points in the program of grouped memory operations.
    Begin is inclusive, end is exclusive.

    Invariant: begin <= end
    """
    begin: float
    end: float
    def contains(self, other: LiveRange):
        """Is other entirely within self"""
    def join(self, other: LiveRange):
        """Combine two ranges using a union operation"""
    def __len__(self) -> int: ...

class LiveRanges:
    """
    A collection of LiveRange regions, allowing for non-contiguous
    live regions.

    Invariant: LiveRanges.ranges is in sorted order and non-overlapping
    """
    ranges: Incomplete
    def __init__(self, ranges: Iterable[LiveRange]) -> None: ...
    def overlaps(self, other: LiveRanges):
        """Check if any pair of ranges in self and other overlap"""
    @property
    def begin(self): ...
    @property
    def end(self): ...
    def __repr__(self) -> str: ...

class AllocationTreeNode:
    """
    Abstract base class for nodes in allocation pool.
    """
    def allocate(self, block: Allocation, is_last: bool) -> bool:
        """
        Try to assign block to a memory location in this bool.  Return True if
        an assignment was made.
        """
    def get_live_ranges(self) -> LiveRanges:
        """Aggregate LiveRanges for all objects below this in tree"""
    def get_size_hint(self) -> int:
        """Number of bytes used for example inputs"""
    def get_symbolic_size(self) -> sympy.Expr:
        """Number of bytes needed at runtime"""
    def finalize(self, pool, offset) -> AllocationTreeNode:
        """Called after all allocations have been made"""
    def is_empty(self): ...

@dataclasses.dataclass
class Allocation(AllocationTreeNode):
    """
    Represents memory allocated to a given node in the allocation pool.
    """
    node: BufferLike
    live_range: LiveRange
    size_hint: int
    symbolic_size: sympy.Expr
    allocated: bool = ...
    pool: AllocationPool | None = ...
    offset: sympy.Expr | None = ...
    @property
    def device(self): ...
    def get_live_ranges(self): ...
    def get_size_hint(self): ...
    def get_symbolic_size(self): ...
    def mark_allocated(self) -> None: ...
    def finalize(self, pool, offset): ...
    def codegen_alloc_from_pool(self, wrapper): ...
    def __repr__(self) -> str: ...

@dataclasses.dataclass
class Empty(AllocationTreeNode):
    """
    Placeholder to represent empty space in the allocation pool.
    Only exists to get the size_hint correct in parent nodes.
    """
    size_hint: int
    def get_live_ranges(self): ...
    def get_size_hint(self): ...
    def get_symbolic_size(self): ...
    def is_empty(self): ...

class MemorySplitProtocol(Protocol):
    get_live_ranges: CachedMethod[[], LiveRanges]
    get_size_hint: CachedMethod[[], int]
    get_symbolic_size: CachedMethod[[], sympy.Expr]
    def _allocate(self, block: Allocation, is_last: bool) -> bool: ...

class ClearCacheOnAllocateMixin(MemorySplitProtocol, metaclass=abc.ABCMeta):
    """
    Helper to assist in caching get_live_ranges, get_size_hint, and
    get_symbolic_size.
    """
    def allocate(self, block: Allocation, is_last: bool): ...
    def clear_cache(self) -> None: ...

@dataclasses.dataclass
class TemporalSplit(ClearCacheOnAllocateMixin, AllocationTreeNode):
    """
    Contains a list of allocations not overlapping in LiveRanges.

    Invariant: no pair (a,b) in self.allocations will have:
         a.get_live_ranges().overlaps(b.get_live_ranges())
    """
    allocations: list[AllocationTreeNode]
    def _allocate(self, block: Allocation, is_last: bool): ...
    @cache_on_self
    def get_live_ranges(self) -> LiveRanges: ...
    @cache_on_self
    def get_size_hint(self) -> int: ...
    @cache_on_self
    def get_symbolic_size(self) -> sympy.Expr: ...
    def is_empty(self): ...
    def finalize(self, pool, offset): ...

@dataclasses.dataclass
class SpatialSplit(ClearCacheOnAllocateMixin, AllocationTreeNode):
    """
    Contains two allocations, left and right, that do not overlap in space.
    Right will be allocated immediately after left in memory.
    """
    left: TemporalSplit
    right: TemporalSplit
    @staticmethod
    def create(left, extra_space): ...
    def _allocate(self, block: Allocation, is_last: bool): ...
    @cache_on_self
    def get_live_ranges(self): ...
    @cache_on_self
    def get_size_hint(self) -> int: ...
    @cache_on_self
    def get_symbolic_size(self) -> sympy.Expr: ...
    def finalize(self, pool, offset): ...

@dataclasses.dataclass
class AllocationPool:
    """
    Represents a pool of allocations that will be generated by a single
    call to torch.empty.
    """
    device: torch.device
    root: TemporalSplit
    can_expand: bool = ...
    restrict_live_range: LiveRange | None = ...
    name: str | None = ...
    names_to_del: list[str] = dataclasses.field(default_factory=list)
    creation_cache: dict[str, str] = dataclasses.field(default_factory=dict)
    def allocate(self, block: Allocation, is_last: bool): ...
    def allocate_at_end(self, block): ...
    def finalize(self, name) -> None: ...
    def codegen_create(self, wrapper, code: IndentedBuffer): ...
    def codegen_destroy(self, wrapper, code: IndentedBuffer): ...
    def __eq__(self, other): ...
    def __hash__(self): ...

@dataclasses.dataclass
class AllocationPools:
    """
    Collection of many AllocationPool objects grouped by device.
    """
    device_to_pools: dict[torch.device, list[AllocationPool]] = dataclasses.field(default_factory=dict)
    def get_pools(self, block): ...
    def allocate(self, block: Allocation): ...
    def allocate_output(self, block: Allocation):
        """Outputs get different pools so memory gets freed properly"""
    def finalize(self) -> None:
        """Called at the end of allocation process"""
    def pprint(self) -> None: ...

class BufferGroup:
    """
    Due to inplace reuse an allocated buffer can have many names.
    This tracks these collections of buffers sharing underlying memory.
    """
    node: Incomplete
    names: Incomplete
    is_output: bool
    allocation: Allocation | None
    live_range: Incomplete
    def __init__(self, node: BufferLike) -> None: ...
    def update_usage(self, timestep: int):
        """Expand self.live_range to include timestep"""
    def sym_nbytes(self): ...
    def make_allocation(self) -> None: ...
    def __repr__(self) -> str: ...

@dataclasses.dataclass
class PoolMemoryPlanningLine(MemoryPlanningLine):
    """Abstract base class for {Alloc,Dealloc}FromPoolLine"""
    group: BufferGroup
    timestep: int | None = ...
    @property
    def node(self): ...

@dataclasses.dataclass
class AllocFromPoolLine(PoolMemoryPlanningLine):
    """Similar to AllocationLine, but takes memory from a pool"""
    is_first_pool_usage: bool = ...
    def codegen(self, code: IndentedBuffer): ...

@dataclasses.dataclass
class DeallocFromPoolLine(PoolMemoryPlanningLine):
    """Similar to FreeIfNotReusedLine, but takes memory from a pool"""
    is_last_pool_usage: bool = ...
    def codegen(self, code: IndentedBuffer): ...

@dataclasses.dataclass
class MemoryPlanner:
    """
    Coordination object to run memory planning passes during wrapper
    codegen.
    """
    wrapper: Any
    pools: AllocationPools = dataclasses.field(default_factory=AllocationPools)
    buffer_groups: list[BufferGroup] | None = ...
    def plan(self, lines: list[Any]) -> list[Any]:
        """Call all the memory planning passes in sequence"""
    def drop_removed_buffers(self, lines) -> None:
        """
        Replace any memory planning lines in V.graph.removed_buffers with NullLine
        """
    def compute_buffer_groups(self, lines):
        """
        Populates self.buffer_groups with BufferGroup objects that join
        allocations with common storage (due to inplace reuse) into a
        single object.
        """
    def convert_to_pool_lines(self, lines) -> None:
        """
        Convert AllocateLine/FreeIfNotReusedLine/ReuseLine into their
        pool-based counterparts.
        """
    def compute_live_ranges(self, lines) -> None:
        """Populate every BufferGroup.live_ranges field based on first/last usage"""
    def allocate_groups(self):
        """
        Assign every allocation to a specific location in a specific AllocationPool.
        """
    def mark_first_last_usage(self, lines) -> None:
        """
        Populate the AllocFromPoolLine.is_first_pool_usage and
        DeallocFromPoolLine.is_last_pool_usage fields so that pools
        are created/destroyed.
        """
