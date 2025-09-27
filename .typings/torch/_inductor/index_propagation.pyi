import sympy
import torch
from .ops_handler import DefaultHandler as DefaultHandler
from .sizevars import statically_known_true as statically_known_true
from .utils import generate_assert as generate_assert
from .virtualized import V as V
from _typeshed import Incomplete
from collections.abc import Sequence
from dataclasses import dataclass
from torch._prims_common import dtype_to_type as dtype_to_type, is_integer_dtype as is_integer_dtype
from torch.utils._sympy.functions import FloorDiv as FloorDiv, ModularIndexing as ModularIndexing, Where as Where
from torch.utils._sympy.value_ranges import ValueRanges as ValueRanges, bound_sympy as bound_sympy
from typing import Any, Literal, overload
from typing_extensions import TypeAlias

_ExprType: Incomplete

def _is_constant(val: _ExprType): ...
def upper_bound(val: _ExprType): ...

@dataclass
class TypedExpr:
    """A SymPy expression with associated type"""
    expr: _ExprType
    dtype: torch.dtype
    def is_constant(self): ...
    def __post_init__(self) -> None: ...

class SymPyOps:
    """An ops handler where all IR values are SymPy expressions

    When a value cannot be represented as a SymPy expression, the method is
    either not defined, or returns NotImplemented

    """
    @staticmethod
    def identity(value: Any) -> Any: ...
    @staticmethod
    def constant(value: int | float | bool, dtype: torch.dtype) -> TypedExpr: ...
    @staticmethod
    def index_expr(value: sympy.Expr | int, dtype: torch.dtype) -> TypedExpr: ...
    @staticmethod
    def to_dtype(value: TypedExpr, dtype: torch.dtype, src_dtype: torch.dtype | None = None, use_compute_types: bool = False) -> TypedExpr: ...
    @staticmethod
    def abs(x: TypedExpr) -> TypedExpr: ...
    @staticmethod
    def square(x: TypedExpr) -> TypedExpr: ...
    @staticmethod
    def add(x: TypedExpr, y: TypedExpr) -> TypedExpr: ...
    @staticmethod
    def sub(x: TypedExpr, y: TypedExpr) -> TypedExpr: ...
    @staticmethod
    def mul(x: TypedExpr, y: TypedExpr) -> TypedExpr: ...
    @staticmethod
    def neg(x: TypedExpr) -> TypedExpr: ...
    @staticmethod
    def floordiv(x: TypedExpr, y: TypedExpr) -> TypedExpr: ...
    @staticmethod
    def mod(x: TypedExpr, y: TypedExpr) -> TypedExpr | None: ...
    @staticmethod
    def remainder(x: TypedExpr, y: TypedExpr) -> TypedExpr | None: ...
    @staticmethod
    def minimum(x: TypedExpr, y: TypedExpr) -> TypedExpr: ...
    @staticmethod
    def maximum(x: TypedExpr, y: TypedExpr) -> TypedExpr: ...

@dataclass
class IndexPropVar:
    value: Any
    is_symbolic: bool = ...
    @staticmethod
    def new_symbolic(expr: TypedExpr) -> IndexPropVar: ...
    def __post_init__(self) -> None: ...

IndexPropResult: TypeAlias

class IndexPropagation(DefaultHandler):
    """Ops wrapper that tries to propagate constant and index_expr values through the computation.

    This aims to maximize the compile time simplification possible, and convert
    indirect indexing from arange into normal static indexing.

    """
    _inner: Incomplete
    shape_env: Incomplete
    var_to_range: Incomplete
    indirect_var_ranges: Incomplete
    axioms: Incomplete
    def __init__(self, inner: Any, iter_ranges: dict[sympy.Symbol, sympy.Expr], indirect_var_ranges: dict[sympy.Symbol, sympy.Expr]) -> None: ...
    def materialize_expr(self, expr: sympy.Expr, dtype: torch.dtype) -> Any: ...
    def unwrap(self, a: Any | IndexPropVar) -> Any: ...
    def wrap(self, a) -> IndexPropResult: ...
    @overload
    def fallback(self, name: Literal['indirect_indexing'], args: Sequence[Any], kwargs: dict[str, Any]) -> IndexPropVar: ...
    @overload
    def fallback(self, name: str, args: Sequence[Any], kwargs: dict[str, Any]) -> IndexPropResult: ...
    def propagate_sympy(self, name: str, args: Sequence[Any], kwargs: dict[str, Any]) -> IndexPropResult: ...
    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any: ...
    def statically_true(self, e):
        """
        Given some iter_ranges, return a function that given an expression, returns whether
        it is true or false using value ranges, guard knowledge and runtime_asserts.

        FIXME I think this may not be entirely right, as we may not be able to use all runtime_asserts
              If this is an issue, just use guards in `self.axioms`.

              The proper way of handling this would be to have a global shape_env that adds
              runtime_asserts as they happen in the code. Then, it should be used in SimplifyIndexing
              to perform wrap_expr and in CSEProxy.check_bounds to elide upper / lower bounds also
              for indirect_indexing
        """
    def indirect_indexing(self, index: Any | IndexPropVar, size: Any, check: bool = True, wrap_neg: bool = True) -> Any: ...
