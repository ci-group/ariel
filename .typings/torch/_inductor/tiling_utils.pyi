import dataclasses
import sympy
from .virtualized import V as V
from _typeshed import Incomplete
from collections.abc import Iterable as Iterable, Iterator as Iterator
from torch._inductor import config as config
from torch._inductor.dependencies import index_vars_no_squeeze as index_vars_no_squeeze
from torch._inductor.scheduler import FusedSchedulerNode as FusedSchedulerNode, SchedulerNode as SchedulerNode
from torch._inductor.utils import sympy_product as sympy_product, sympy_subs as sympy_subs
from torch.utils._ordered_set import OrderedSet as OrderedSet
from torch.utils._sympy.functions import FloorDiv as FloorDiv, Identity as Identity, ModularIndexing as ModularIndexing
from torch.utils._sympy.solve import try_solve as try_solve
from torch.utils._sympy.symbol import SymT as SymT, symbol_is_type as symbol_is_type
from typing import Callable, Literal, TypeVar, overload

T = TypeVar('T')
U = TypeVar('U')
Split: Incomplete
VarsAndRanges: Incomplete
loop_tiling_log: Incomplete

def solve_for_zero(expr: sympy.Expr) -> sympy.Expr | None:
    """
    Given an expr with a single free symbol, solve for a constant relation that would make
    this expression 0.
    """
def solve_for_tiling(expr: sympy.Expr) -> sympy.Expr | None:
    """
    Giving an expr with a single free symbol, try to find a tiling that would
    make the expression coalesced with respect to that symbol.

    Tiling an expression `x` by `y` means that the expression will now be indexed
    by both the original (x) and by (x * y). So we are looking for a
    multiplicative factor that will make ((x + 1) * y) - (x * y) == 1.

    To simplify things for sympy, we'll try just x * y == 1, check x(1) and x(0).
    """
def find_coalesced_var(index: sympy.Expr, var_ranges: dict[sympy.Expr, int]) -> sympy.Expr | None:
    """
    Try to find the symbol which coalesces this index
    """

@dataclasses.dataclass(frozen=True)
class FusedNormalizedReadsWrites:
    """
    Normalized reads and writes for nodes in the same FusedSchedulerNode.
    """
    index_vars: OrderedSet[sympy.Symbol]
    reduce_vars: OrderedSet[sympy.Symbol]
    reads: dict[sympy.Expr, OrderedSet[str]]
    writes: dict[sympy.Expr, OrderedSet[str]]
    var_ranges: dict[sympy.Symbol, int]

@overload
def get_pw_red_splits(n: SchedulerNode, pointwise_numel: sympy.Expr, red_numel: sympy.Expr, none_if_not_divisible: Literal[True]) -> tuple[VarsAndRanges, VarsAndRanges] | None: ...
@overload
def get_pw_red_splits(n: SchedulerNode, pointwise_numel: sympy.Expr, red_numel: sympy.Expr, none_if_not_divisible: Literal[False] = False) -> tuple[VarsAndRanges, VarsAndRanges]: ...

class NodeSplitGetter:
    """
    Finds a Pointwise, Reduction Split that compatible with all nodes in a SchedulerNode.
    """
    node: Incomplete
    pointwise_numel: sympy.Expr
    red_numel: sympy.Expr
    pw_split_options: dict[int, OrderedSet[Split]]
    reduction_split: Split
    all_node_sizes: OrderedSet[tuple[Split, Split]]
    seen_pw_splits: OrderedSet[Split]
    def __init__(self, node: FusedSchedulerNode | SchedulerNode) -> None: ...
    def get_node_splits(self) -> tuple[Split, Split]:
        """
        Get a compatible pointwise, reduction split of the node
        """
    def try_split(self, pw: Split, red: Split) -> tuple[Split, Split] | None:
        """
        See if this split is compatible, and potentially returning a longer split
        than the input.
        """

zip_equal: Incomplete

def apply_var_mapping(iter_vars: list[sympy.Symbol], red_vars: list[sympy.Symbol], norm_pw_vars: list[sympy.Symbol], norm_red_vars: list[sympy.Symbol], new_ranges: list[list[sympy.Expr]], return_getters_groups: list[list[Callable[[list[sympy.Expr]], sympy.Expr]]]) -> dict[sympy.Symbol, sympy.Expr]:
    """Maps original variables to expressions using normalized variables."""
def extract_normalized_read_writes(node: FusedSchedulerNode | SchedulerNode) -> FusedNormalizedReadsWrites | None:
    """Extracts index variables, reduce variables, read/write expressions, and variable ranges from a fused node."""
def get_score(addr: sympy.Expr, var_ranges: dict[sympy.Symbol, int]) -> int:
    """
    Score addr according to its approximate size
    """
def get_hint(v: sympy.Expr | int) -> int: ...

@dataclasses.dataclass(frozen=True)
class VarTiling:
    """
    Tiling of a var by `tiling_factor` that yields additional coalesced mem accesses by `benefit_score`
    """
    var: sympy.Symbol
    tiling_factor: int
    score: int

@dataclasses.dataclass(frozen=True)
class CoalesceVarAnalysis:
    coalesced_by_var: dict[sympy.Expr, int]
    norm_read_writes: FusedNormalizedReadsWrites
    suggested_split: VarTiling | None = ...

def analyze_memory_coalescing(fused_node: FusedSchedulerNode | SchedulerNode) -> CoalesceVarAnalysis | None:
    """
    Find variables that coalesce the reads and writes and score the total size.

    If uncoalesced memory expressions are found, look for additionally tiling of variables
    which will coalesce memory accesses.

    For instance - for the following expression:

    (32*p0) // 2048

    Tiling p0 by 64 will make this expression coalesced.
    """
