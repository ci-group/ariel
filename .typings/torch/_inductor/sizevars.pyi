import functools
import sympy
from .runtime.runtime_utils import is_power_of_2 as is_power_of_2
from .utils import VarRanges as VarRanges, has_free_symbols as has_free_symbols, sympy_index_symbol as sympy_index_symbol, sympy_index_symbol_with_prefix as sympy_index_symbol_with_prefix, sympy_subs as sympy_subs
from .virtualized import V as V
from _typeshed import Incomplete
from collections.abc import Iterable, Sequence
from sympy import Expr
from torch.fx.experimental.symbolic_shapes import ShapeEnv as ShapeEnv, has_free_unbacked_symbols as has_free_unbacked_symbols
from torch.utils._ordered_set import OrderedSet as OrderedSet
from torch.utils._sympy.functions import FloorDiv as FloorDiv, ModularIndexing as ModularIndexing
from torch.utils._sympy.symbol import SymT as SymT, symbol_is_type as symbol_is_type
from torch.utils._sympy.value_ranges import IntInfinity as IntInfinity, ValueRanges as ValueRanges, bound_sympy as bound_sympy
from typing import Any, Callable

log: Incomplete

def statically_known_true(shape_env: ShapeEnv, expr: sympy.Basic | bool, axioms: tuple[sympy.Expr] | None = None, var_to_range: tuple[tuple[sympy.Symbol, ValueRanges[Any]]] | None = None) -> bool: ...

class SizeVarAllocator:
    shape_env: Incomplete
    var_to_val: Incomplete
    replacements: dict[sympy.Symbol, Expr]
    unbacked_replacements: dict[Expr, Expr] | None
    precomputed_replacements: dict[Expr, sympy.Symbol]
    inv_precomputed_replacements: dict[sympy.Symbol, Expr]
    stride_vars: Incomplete
    simplify_with_ranges: Incomplete
    _simplify_loops: Incomplete
    def __init__(self, shape_env=None) -> None: ...
    def simplify(self, expr: Expr): ...
    def make_simplify_with_ranges_cache(self) -> Callable[[Expr, VarRanges], Expr]:
        """
        self._simplify_with_ranges() can be expensive, cache its results
        """
    def make_simplify_loops_cache(self):
        """
        self._simplify_with_ranges() can be expensive, cache its results
        """
    def _simplify_with_ranges(self, expr: Expr, var_ranges: VarRanges) -> Expr:
        """
        Simplify indexing expression with knowledge of the ranges of
        iteration variables.
        """
    def _simplify_loops_impl(self, index_vars: list[sympy.Symbol], sizes, index_formulas):
        """
        Try to remove as many axis from loop iterations as possible, by:
            1) removing size==1 dimensions
            2) fuse contiguous dimensions into a single loop
            If channel_last = True, we will prevent the last dim fused with other dims
        """
    def statically_known_true(self, expr: sympy.Basic | bool) -> bool:
        """
        Returns true if an expression is always true (symbolically or via guards),
        false otherwise. Never add guards, or throw data dependent errors.
        """
    def statically_known_equals(self, left: Expr | int, right: Expr | int) -> bool:
        """
        Returns a bool indicating if it is sound to optimize as if left and right are equal.
        """
    def statically_known_list_equals(self, left: list[Expr], right: list[Expr]) -> bool:
        """
        Returns a bool indicating if it is sound to optimize as if left and right lists are equal.
        """
    def statically_known_leq(self, left: Expr, right: Expr | int) -> bool:
        """
        Returns a bool indicating if it is sound to optimize as if left is less than or equal to right.
        """
    def statically_known_geq(self, left: Expr, right: Expr | int) -> bool:
        """
        Returns a bool indicating if it is sound to optimize as if left is greater than or equal to right.
        """
    def statically_known_lt(self, left: Expr, right: Expr | int) -> bool:
        """
        Returns a bool indicating if it is sound to optimize as if left is less than right.
        """
    def statically_known_gt(self, left: Expr, right: Expr | int) -> bool:
        """
        Returns a bool indicating if it is sound to optimize as if left is greater than right.
        """
    def statically_known_multiple_of(self, numerator: Expr, denominator: Expr | int) -> bool:
        """
        Return a bool indicating if it is sound to optimize for the numerator being a multiple of the denominator.
        """
    def statically_known_power_of_2(self, expr: Expr) -> bool:
        """
        Returns a bool indicating if x is known to be a power of 2.
        """
    def guard_equals(self, left: Expr, right: Expr) -> Expr: ...
    def guard_leq(self, left: Expr, right: Expr) -> None: ...
    def guard_lt(self, left: Expr, right: Expr) -> None: ...
    def guarded_order(self, seq):
        """
        Return the order of a sequence as a permutation of range(len(seq)) and guard on that order not changing.
        """
    def guard_or_false(self, left): ...
    def guard_or_true(self, left): ...
    def evaluate_expr(self, left: Expr | sympy.logic.boolalg.Boolean, size_oblivious: bool = False, fallback_value: bool | None = None) -> bool: ...
    def evaluate_min(self, left: Expr, right: Expr) -> Expr:
        """return the smaller of left and right, and guard on that choice"""
    def evaluate_max(self, left: Expr, right: Expr) -> Expr:
        """return the larger of left and right, and guard on that choice"""
    def evaluate_static_shape(self, left: Expr | int) -> int: ...
    def evaluate_static_shapes(self, left: Sequence[Expr | int]) -> list[int]: ...
    def remove_precomputed_replacements(self, expr: Expr) -> Expr: ...
    def symbolic_hint(self, expr: Expr | int) -> Expr | int: ...
    def size_hint(self, expr: Expr | int, *, fallback: int | None = None) -> int: ...
    def size_hint_or_throw(self, expr: Expr | int) -> int: ...
    def size_hints(self, exprs: Iterable[Expr | int], *, fallback: int | None = None) -> tuple[int, ...]: ...
    def _lru_cache(self, fn, maxsize=None):
        """
        Wrapper around functools.lru_cache that clears when replacements
        has been invalidated.
        """
    def make_stride_vars_cache(self): ...
    def _stride_vars(self, index: Expr, vars: Sequence[sympy.Symbol], support_vars: Sequence[sympy.Symbol]) -> list[Expr]:
        """Convert an indexing expression back into strides

        NOTE: This is only valid if the index is a standard strided offset
        calculation. e.g. 10 * ModularIndexing(i0 + 1, 1, 2) would give a
        stride of -10 because the index wraps around after the first element

        """
    def _get_unbacked_replacements(self) -> dict[Expr, Expr]:
        """
        This helps with covering unbacked symint cases where you may have two
        expressions: s0 + u0 and u1. And s0 + u0 is known to be equal to u1
        via deferred_runtime_asserts.

        For example in atomically_apply_size_hint, it must return the same size
        hint for both s0 + u0 and u1, but it first needs to know they are equal.
        Then it can substitute s0 + u0 for u1.
        """
    @functools.lru_cache
    def _sub_unbacked_exprs(self, expr: Expr) -> Expr: ...
    def atomically_apply_size_hint(self, expr: Expr | int, *, fallback: int | None = None) -> Expr | int: ...
    def offset_var(self, index: Expr, vars: list[sympy.Symbol]) -> Expr:
        """Extract offset part of an indexing expression"""
    def stride_hints(self, index: Expr, vars: Sequence[sympy.Symbol], support_vars: Sequence[sympy.Symbol] | None = None) -> list[int]: ...
    def stride_order(self, index: Expr, vars: list[sympy.Symbol]) -> list[int]: ...
    def lookup_precomputed_size(self, expr: Expr) -> Expr: ...
    def free_symbols(self) -> OrderedSet[sympy.Symbol]: ...
    def combine_modular_indexing_pairs(self, index: sympy.Expr) -> sympy.Expr:
        """
        A pair of special ModularIndexing can be combined.

        E.g. ModularIndexing(ModularIndexing(x, 1, a), 1, b)
        We can simplify this to ModuleIndexing(x, 1, b), if
        1. x is non negative integer
        2. a and b are positive integers
        3. a is a multiple of b.
        """
    def expand_floor_div(self, index: sympy.Expr) -> bool | tuple[sympy.Expr, sympy.Expr]:
        """
        Expand the FloorDiv to the entire expression so that the expression may
        be simplified.

        E.g., for a 2D contiguous tensor with shape [a, 2 * b], and index variables
        x1, x2, index expression 'x1 * 2b + x2' can be easily combined.
        But index expression 'x1 * b + x2 // 2' can not.
        By expanding the FloorDiv to the entire expression, we get
        '(x1 * 2b + x2) // 2'. This transformation allows us to merge loops
        for the numerator!

        Return false if this optimization can be applied;
        Return the new expression and the denominator otherwise.
        The original expression will be equivalent to 'new_expression // denominator'
        """

def join_dimensions(expr: Expr) -> Expr: ...
def _join_dimensions_cached(expr: Expr) -> Expr:
    """
    ModularIndexing(i0, 1, 32) + 32 * ModularIndexing(i0, 32, 4)
    becomes
    ModularIndexing(i0, 1, 128)
    ModularIndexing(i0, 1, 32) + 32 * FloorDiv(i0, 32)
    becomes i0


    This type of pattern can come from view operations
    """

class SimplifyIndexing(V.WrapperHandler):
    """
    A wrapper around .virtualize.ops that uses var range information to
    simplify ModularIndexing/FloorDiv.
    """
    name: str
    _simplify: Callable[[Expr], Expr]
    def __init__(self, inner, var_ranges: VarRanges) -> None: ...
    def load(self, name: str, index: sympy.Expr): ...
    def store(self, name, index, value, mode=None): ...
    def store_reduction(self, name, index, value): ...
    def index_expr(self, index, dtype): ...
    def check_bounds(self, index, size, lower, upper): ...
