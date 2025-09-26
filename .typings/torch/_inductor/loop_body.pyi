import collections
import functools
import sympy
import torch
import torch.fx
from . import config as config, dependencies as dependencies
from .codegen.common import index_prevent_reordering as index_prevent_reordering
from .ops_handler import DefaultHandler as DefaultHandler, OpsHandler as OpsHandler, WrapperHandler as WrapperHandler
from .utils import cache_on_self as cache_on_self, reduction_num_outputs as reduction_num_outputs, sympy_index_symbol_with_prefix as sympy_index_symbol_with_prefix, sympy_subs as sympy_subs
from .virtualized import V as V, ops as ops
from _typeshed import Incomplete
from enum import Enum
from torch._dynamo.utils import identity as identity
from torch.fx.proxy import Scope as Scope, TracerBase as TracerBase
from torch.utils._sympy.symbol import SymT as SymT
from typing import Any, Callable, NamedTuple, TypeVar

T = TypeVar('T')

class InterpreterShim(torch.fx.Interpreter):
    @staticmethod
    @functools.cache
    def _dummy_gm(): ...
    module: Incomplete
    graph: Incomplete
    submodules: Incomplete
    extra_traceback: bool
    fetch_attr: Incomplete
    current_node: Incomplete
    def __init__(self, graph, submodules) -> None: ...
    def run_node(self, n: torch.fx.Node) -> Any: ...
    def run(self, *args, **kwargs): ...

class LightTracer(TracerBase):
    graph: Incomplete
    scope: Incomplete
    module_stack: Incomplete
    node_name_to_scope: Incomplete
    def __init__(self) -> None: ...

class MemoryEntry(NamedTuple):
    index_name: str
    buffer_name: str | None
    mode: str | None

class MemoryUsageType(Enum):
    LOAD = ...
    LOAD_SEED = ...
    STORE = ...
    STORE_REDUCTION = ...
    INDEX_EXPR = ...
    CHECK_BOUNDS = ...
    BUCKETIZE = ...

class LoopBody:
    """
    Captures the body of a Loops subclass into an FX graph.  Persists any
    indexing simplifications and makes it easier to analyze loop bodies.
    """
    indexing_exprs: dict[str, sympy.Expr]
    indexing_exprs_name: dict[sympy.Expr, str]
    submodules: dict[str, Any]
    subblocks: dict[str, LoopBodyBlock]
    indirect_vars: list[sympy.Symbol]
    indirect_var_ranges: dict[sympy.Symbol, sympy.Expr]
    root_block: LoopBodyBlock
    memory_usage: dict[MemoryUsageType, list[MemoryEntry]]
    op_counts: collections.Counter[str]
    sizes: Incomplete
    iter_vars: Incomplete
    reduce_vars: Incomplete
    var_ranges: Incomplete
    indexing: Incomplete
    def __init__(self, fn, args, var_ranges, iter_vars, reduce_vars) -> None: ...
    def _init_with_tracing(self, fn, args) -> None:
        """Do an FX trace of an arbitrary callable to construct self"""
    def _init_with_copy(self, other: LoopBody, args):
        """
        _init_with_tracing() is slow, so this is a fast path in the case
        where we are just reordering/merging/splitting the args of an
        existing LoopBody.
        """
    def has_op(self, name: str): ...
    def merge_loops(self) -> LoopBody:
        """
        Merge both iteration and reduction loops and return a new LoopBody.
        """
    def reorder_iter_loops(self, new_order) -> LoopBody:
        """
        Reorder iteration loops and return a new LoopBody.
        """
    @property
    def vars(self): ...
    @cache_on_self
    def get_nodes(self): ...
    @cache_on_self
    def bounds(self): ...
    def get_read_expr(self, buffer_name): ...
    def get_write_expr(self, buffer_name): ...
    def get_read_exprs(self): ...
    def get_all_read_expr(self, buffer_name): ...
    def get_write_exprs(self): ...
    def get_all_write_expr(self, buffer_name): ...
    def debug_str(self): ...
    def is_memory_copy(self) -> bool:
        """
        True of this contains only a single loads and store.
        Note, this could involve a layout change.
        """
    __repr__ = debug_str
    def add_index_expr(self, expr: sympy.Expr, mtype: MemoryUsageType, buffer_name: str | None = None, mode: str | None = None): ...
    def add_submodule(self, block, prefix):
        """Not actually for nn.Modules, but subblocks in generated code are mapped to FX call_module opcodes"""
    def add_indirect(self, size): ...
    def replace_indirect(self, old, new) -> None:
        """Swap in a variable used in indirect indexing"""
    def get_index(self, name): ...
    def indexing_from_args(self, indices): ...
    def __call__(self, *indices): ...
    def bind_set_indirect_shim(self, var, size, check, wrap_neg): ...
    def bind_scan_shim(self, combine_fn): ...
    def bind_masked_shim(self, name): ...

class LoopBodyBlock:
    """
    Captures the body of a Loops subclass into an FX graph.
    In normal cases there will be a 1:1 mapping between LoopBody and
    LoopBodyBlock, however in the case of ops.masked() the masked out
    operations will manifest as an extra LoopBodyBlock.
    """
    body: Incomplete
    graph: Incomplete
    def __init__(self, body: LoopBody, fn: Callable[..., Any], args: list[Any]) -> None: ...
    def __call__(self): ...
    def debug_str(self, name: str = 'block'): ...
    def contains_only_ops(self, allowed_ops) -> bool: ...
    def clone(self, body: LoopBody):
        """Shallow copy with a new parent LoopBody"""

class CountOps(DefaultHandler):
    _inner: Incomplete
    _counts: Incomplete
    def __init__(self, inner: OpsHandler[Any], counts: collections.Counter[str]) -> None: ...
    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any: ...

class CaptureIndexing(WrapperHandler):
    name: str
    body: Incomplete
    tracer: Incomplete
    def __init__(self, inner: OpsHandler[Any], body: LoopBody, tracer: LightTracer) -> None: ...
    def _add_index(self, expr: sympy.Expr, mtype: MemoryUsageType, **kwargs: Any): ...
    def _simplify(self, expr: sympy.Expr) -> sympy.Expr: ...
    def load(self, name: str, index: sympy.Expr): ...
    def load_seed(self, name: str, index: int): ...
    def store(self, name, index, value, mode=None): ...
    def store_reduction(self, name, index, value): ...
    def reduction(self, dtype, src_dtype, reduction_type, value): ...
    def index_expr(self, index, dtype): ...
    def check_bounds(self, index, size, lower, upper): ...
    def bucketize(self, values: T, boundaries: tuple[str, sympy.Expr, sympy.Expr, sympy.Expr], boundary_indices: T, indexing_dtype: torch.dtype, right: bool, sorter: tuple[str, sympy.Expr] | None = None, sorter_indices: T | None = None) -> T:
        """
        See [Note: Inductor bucketize op]
        """
    def masked(self, mask_proxy, masked_body: Callable[..., Any], other_proxy):
        """
        Recursively capture the masked out body in another LoopBodyBlock
        """
    def scan(self, dtype_proxy, combine_fn: Callable[[tuple[Any, ...], tuple[Any, ...]], tuple[Any, ...]], value_proxy): ...
    def sort(self, dtypes, values, stable, descending): ...
    def frexp(self, value_proxy): ...
    def indirect_indexing(self, index_proxy, size, check: bool = True, wrap_neg: bool = True):
        """
        Flow data from tensors into indexing formulas.
        Introduce a call_module to update the indexing.
        """
    def output(self, *result) -> None: ...
