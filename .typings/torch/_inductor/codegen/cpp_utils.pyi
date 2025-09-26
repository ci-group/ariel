import sympy
import torch
import types
from .. import ir as ir
from ..dependencies import Dep as Dep
from ..loop_body import LoopBody as LoopBody
from ..scheduler import BaseSchedulerNode as BaseSchedulerNode, SchedulerBuffer as SchedulerBuffer
from ..utils import IndentedBuffer as IndentedBuffer, sympy_index_symbol_with_prefix as sympy_index_symbol_with_prefix, sympy_subs as sympy_subs
from ..virtualized import OpsValue as OpsValue, V as V, ops as ops
from .common import CSEVariable as CSEVariable, Kernel as Kernel, KernelArgs as KernelArgs, OptimizationContext as OptimizationContext
from _typeshed import Incomplete
from torch._prims_common import is_integer_dtype as is_integer_dtype
from torch.utils._ordered_set import OrderedSet as OrderedSet
from torch.utils._sympy.printers import CppPrinter as _CppPrinter
from torch.utils._sympy.symbol import SymT as SymT, symbol_is_type as symbol_is_type
from torch.utils._sympy.value_ranges import ValueRanges as ValueRanges
from typing import Any, Callable, NamedTuple

DTYPE_TO_CPP: Incomplete
DTYPE_TO_ATEN: Incomplete
DEVICE_TO_ATEN: Incomplete
LAYOUT_TO_ATEN: Incomplete
DEVICE_TO_INT: Incomplete
_IS_WINDOWS: Incomplete
INDEX_TYPE: str

class GemmBlocking(NamedTuple):
    block_m: Incomplete
    block_n: Incomplete
    block_k: Incomplete

def get_promote_dtype(args): ...
def promote_args(new_args): ...

class CppCSEVariable(CSEVariable):
    is_vec: bool
    dependent_itervars: Incomplete
    def __init__(self, name, bounds: ValueRanges[Any], dtype: torch.dtype | None = None) -> None: ...
    def __repr__(self) -> str: ...
    def update_on_args(self, name, args, kwargs) -> None: ...
    def _set_dependent_itervars(self, index: sympy.Expr):
        """
        Set the relevant itervars for this variable based on the `index` expression.
        This includes the itervars directly used in the `index` as well as relevant itervars
        of other cse variables used in the `index`.
        """
    def depends_on(self, itervar: sympy.Symbol): ...

class CppPrinter(_CppPrinter):
    def doprint(self, expr, *, simplify: bool = True, p: bool = True): ...

cexpr: Incomplete

def cexpr_index(index): ...
def value_to_cpp(value, cpp_type): ...
def rewrite_index_for_function(localize_buffer_handler: LocalizeBufferHandler, index: sympy.Expr, global_buf_name: str): ...
def rewrite_index_for_nodes(localize_buffer_handler: LocalizeBufferHandler, index: sympy.Expr, global_buf_name: str): ...

class LocalizeBufferHandler(V.WrapperHandler):
    global_to_local: Incomplete
    rewrite_index: Incomplete
    def __init__(self, inner, global_to_local: dict[str, ir.Buffer], rewrite_index: Callable[[LocalizeBufferHandler, sympy.Expr, str], sympy.Expr]) -> None: ...
    def localize(self, name: str, index: sympy.Expr): ...
    def load(self, name: str, index: sympy.Expr): ...
    def store(self, name, index, value, mode=None): ...
    def store_reduction(self, name, index, value): ...

class LocalBufferContext:
    """
    This class creates a context that helps to generate code involving Inductor IR with
    function local buffers. These buffers are constructed during the codegen process and
    are used to store intermediate results such as local accumulators. We do not want to
    add them to `V.graph` since they are not global and we do not want to add them as
    function arguments either. So we patch the codegen processes under this scope to support
    these buffers without exposure to the outside world.
    """
    kernel_args: Incomplete
    exit_stack: Incomplete
    local_buffers: dict[str, ir.Buffer]
    global_buffers: dict[str, ir.Buffer]
    global_to_local: dict[str, ir.Buffer]
    removed_buffers: OrderedSet[str]
    def __init__(self, kernel_args: KernelArgs) -> None: ...
    def __enter__(self): ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: types.TracebackType | None) -> None: ...
    def add_local_buffer(self, local_buffer: ir.Buffer, global_buffers: list[ir.Buffer] | None = None): ...
    def localize_function(self, fn: Callable[..., Any], rewrite_index: Callable[[LocalizeBufferHandler, sympy.Expr, str], sympy.Expr] = ...): ...
    def localize_nodes(self, nodes: list[ir.IRNode], rewrite_index: Callable[[LocalizeBufferHandler, sympy.Expr, str], sympy.Expr] = ...) -> list[ir.IRNode]:
        """
        Given `local_buf` and `global_buf` registered in current `LocalBufferContext`
        though the method of `add_local_buffer`, localizes the `global_buf` to `local_buf`
        for the given `nodes` and returns a new list of IR nodes that work on `local_buf`
        instead of `global_buf`, i.e., all the loads and stores are redirected to
        `local_buf`. This helps the fused loops to work on smaller-sized local buffers
        for better data locality.

        The the data access of `local_buf` is assumed to be contiguous with the
        same order as the `global_buf`.
        """

def unify_mask_base_type(buffer: IndentedBuffer, vars: tuple[CSEVariable, ...], dtype=...):
    """
    Given list of cse variables,
    Cast each to new mask base dtype and return casted cse variable.
    """
def may_unify_binary_op_mask_type(a, b):
    """
    Given two cse variables, when dtype is bool, unify them to the same mask dtype and return casted cse variable.
    """
def codegen_rand(offset, code, rand_function, dst_dtype=...): ...
def get_gemm_template_output_and_compute_dtype(input_dtype): ...
def create_epilogue_with_attr(input_buffer, attr, **kwargs): ...
def _get_loop_body(fn_list): ...
def _get_dtype_from_loopbodies(loop_bodies): ...
def template_fusion_with_epilogues_supported(template: BaseSchedulerNode, epilogues: list[BaseSchedulerNode]) -> tuple[bool, bool]: ...
