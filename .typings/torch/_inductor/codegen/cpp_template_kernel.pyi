import sympy
from .. import config as config, cpp_builder as cpp_builder, ir as ir
from ..autotune_process import CppBenchmarkRequest as CppBenchmarkRequest
from ..loop_body import LoopBody as LoopBody
from ..select_algorithm import PartialRender as PartialRender
from ..utils import sympy_index_symbol as sympy_index_symbol, sympy_index_symbol_with_prefix as sympy_index_symbol_with_prefix
from ..virtualized import V as V
from .common import REMOVED as REMOVED
from .cpp import CppKernel as CppKernel, CppKernelProxy as CppKernelProxy, KernelGroup as KernelGroup
from .cpp_utils import DTYPE_TO_CPP as DTYPE_TO_CPP, LocalBufferContext as LocalBufferContext, cexpr_index as cexpr_index
from _typeshed import Incomplete
from torch._inductor.utils import do_bench_using_profiling as do_bench_using_profiling
from torch.utils._ordered_set import OrderedSet as OrderedSet
from torch.utils._sympy.symbol import SymT as SymT
from typing import Any, Callable

def parse_expr_with_index_symbols(expr): ...
def wrap_with_tensorbox(node) -> ir.TensorBox: ...

class CppTemplateKernel(CppKernel):
    kernel_name: Incomplete
    render_hooks: Incomplete
    local_buffers: Incomplete
    def __init__(self, kernel_name, num_threads) -> None: ...
    def render(self, template, **kwargs): ...
    def def_kernel(self, inputs: dict[str, ir.Buffer], outputs: dict[str, ir.Buffer], aliases: dict[str, str] | None = None, function_name: str = '', extra_sizevars: list[sympy.Expr] | None = None, placeholder: str = '<DEF_KERNEL>') -> str: ...
    def call_kernel(self, name: str, node: ir.CppTemplateBuffer): ...
    def dtype(self, node: ir.Buffer) -> str: ...
    def acc_dtype(self, node: ir.Buffer) -> str: ...
    def size(self, node: ir.Buffer, dim: int) -> str: ...
    def stride(self, node: ir.Buffer, dim: int) -> str: ...
    def index(self, node: ir.Buffer, indices: list[Any]) -> str: ...
    def slice_nd(self, node, ranges: list[tuple[Any, Any]]) -> ir.ReinterpretView:
        """
        Slice the given node with a list of ranges (start and end) corresponding to its dims.
        The dim is not sliced if the corresponding range is empty.
        """
    def select(self, node, dim: int, idx: int) -> ir.ReinterpretView: ...
    def view(self, node, sizes: list[Any]) -> ir.View: ...
    def permute(self, node, dims): ...
    def maybe_codegen_profile(self) -> str: ...
    def unroll_pragma(self, unroll): ...
    def define_buffer(self, name, sizes: list[Any], dtype=...) -> str:
        """Define kernel local buffer"""
    def define_stack_allocated_buffer(self, name, sizes: list[Any], dtype=...) -> str:
        """Define stack-allocated buffer"""
    def reinit_buffer_if_null(self, name):
        """Reinit the previously defined local buffer if it is null"""
    def release_buffer(self, name):
        """Codegen the code to release the ownership of a local buffer to others"""
    def store_pointwise_nodes(self, dst: ir.Buffer, nodes: list[ir.IRNode], offsets: list[sympy.Expr] | None = None, reindexers: list[Callable[[list[Any]], list[Any]] | None] | None = None) -> str: ...
    def store_grouped_gemm_pointwise_nodes(self, dst: tuple[ir.Buffer], nodes: list[ir.IRNode], offsets: list[sympy.Expr], reindexers: list[Callable[[list[Any]], list[Any]] | None], output_names: list[str]) -> str: ...
    def store_output(self, dst: ir.Buffer, src: ir.Buffer, orig_src: ir.Buffer | None = None, epilogue_nodes: list[ir.IRNode] | None = None, offsets: list[Any] | None = None, reindexers: list[Callable[[list[Any]], list[Any]] | None] | None = None):
        """
        Store the `src` buffer to the `dst` buffer. The size of `src` and `dst` should match.
        If `epilogue_nodes` is provided, the `src` buffer is firstly computed with the epilogues
        before stored to `dst`. The `epilogues_nodes` are all pointwise.

        Notes:
        1. `src` and `dst` buffer could be the same buffer in which case we are doing in-place compute
           and stores. In case `epilogue_nodes` are not provided, we do nothing.
        2. The `epilogue_nodes`, if exist, have computations on `src` before storing to `dst` but since
           they come form the original Inductor IR, they might need to be adjusted before working with
           `src` and `dst` as outlined below:
           a) `src` or `dst` buffer could be a sub-slice of the ranges the `epilogue_nodes`work on.
              In this case, the `offsets` could be provided to adjust the indices passed to
              `epilogue_nodes` during codegen and the data ranges are also configured according to
              the sizes of `src` and `dst`.
           b) `dst` might be indexed in a different way as the `epilogue_nodes`, hence a `reindexer` is
              needed on the indices to `epilogue_nodes` to match the indexing of `dst`.
           c) If `src` is local, we need to add a local buffer for it and localize the `orig_src` buffer
              in `epilogue_nodes` with `src`.
        """
    def store_outputs(self, dst: tuple[ir.Buffer], src: tuple[ir.IRNode], orig_src: tuple[ir.IRNode] | None = None, epilogue_nodes: list[ir.IRNode] | None = None, offsets: list[Any] | None = None, reindexers: list[Callable[[list[Any]], list[Any]] | None] | None = None, multi_output_buffers: tuple[ir.MultiOutput] | None = None): ...
    def check_bounds(self, expr, size, lower, upper) -> None: ...

class CppTemplateCaller(ir.ChoiceCaller):
    """
    CppTemplateCaller

    This class represents a caller for CPP template kernels. It is a subclass of ir.ChoiceCaller.
    Attributes:
        name (str): The name of the caller.
        category (str): The category of the caller.
        bmreq (CppBenchmarkRequest): The benchmark request for the caller.
        template_buffer (ir.CppTemplateBuffer): The template buffer for the caller.
    """
    category: Incomplete
    make_kernel_render: Incomplete
    bmreq: Incomplete
    template: Incomplete
    info_kwargs: Incomplete
    def __init__(self, name: str, category: str, input_nodes: list[ir.Buffer], layout: ir.Layout, make_kernel_render: Callable[[ir.CppTemplateBuffer, bool, list[ir.IRNode] | None], str], bmreq: CppBenchmarkRequest, template: CppTemplate, info_kwargs: dict[str, ir.PrimitiveInfoType | list[ir.PrimitiveInfoType]] | None = None) -> None: ...
    def precompile(self) -> None: ...
    def benchmark(self, *args, out) -> float: ...
    def hash_key(self) -> str: ...
    def info_dict(self) -> dict[str, ir.PrimitiveInfoType | list[ir.PrimitiveInfoType]]: ...
    def output_node(self) -> ir.TensorBox: ...
