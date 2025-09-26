import dataclasses
import sympy
import torch
from .ops_handler import DefaultHandler as DefaultHandler
from .virtualized import StoreMode as StoreMode, V as V
from _typeshed import Incomplete
from torch._inductor import config as config
from torch._inductor.dtype_propagation import DtypePropagationOpsHandler as DtypePropagationOpsHandler
from torch._inductor.index_propagation import SymPyOps as SymPyOps, TypedExpr as TypedExpr
from torch._inductor.scheduler import SchedulerNode as SchedulerNode
from typing import Any

def construct_symbol(count: int, dtype: torch.dtype) -> sympy.Symbol: ...

class PreservesZeros(SymPyOps, DefaultHandler):
    """
    For prologue kernels where the loads are masked, does the final store of this kernel preserve
    the zeros.
    """
    count: Incomplete
    store_preserves_zeros: bool | None
    dtype_prop: Incomplete
    def __init__(self) -> None: ...
    def load(self, name: str, index: sympy.Expr) -> TypedExpr: ...
    def store(self, name: str, index: sympy.Expr, value: TypedExpr, mode: StoreMode = None) -> None: ...
    def indirect_indexing(self, *args: Any, **kwargs: Any) -> sympy.Expr: ...
    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any: ...

def prologue_preserves_zero_mask(prologue: SchedulerNode) -> bool:
    """
    Does this prologue preserve zero masks
    """

@dataclasses.dataclass
class DTypeContainer:
    dtype: torch.dtype
    is_scalar: bool = ...

class RecordLowPrecisionOps(DefaultHandler):
    disallow_fp32_ops: Incomplete
    low_precision_numeric_op: bool
    dtype_prop: Incomplete
    non_numeric_ops: Incomplete
    def __init__(self, disallow_fp32_ops: bool = False) -> None: ...
    def load(self, name: str, index: sympy.Expr) -> DTypeContainer: ...
    @staticmethod
    def store(name: str, index: sympy.Expr, value: TypedExpr, mode: StoreMode = None) -> None: ...
    def check_bounds(self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool) -> None: ...
    @staticmethod
    def indirect_indexing(*args: Any, **kwargs: Any) -> sympy.Expr: ...
    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any: ...

def low_prec_float(dtype: torch.dtype) -> bool: ...
def can_codegen_without_upcasts(prologue: SchedulerNode, disallow_fp32_ops: bool = False) -> bool:
    """
    Can this prologue be run without `upcast_to_fp32` while preserving numerics.

    This is only true if the node only contains dtype conversions, indexing, and other non-arithmetic operators.

    If disallow_fp32_ops is True, then we also disallow ops that are explicitly computed in fp32 or fp64.
    """
