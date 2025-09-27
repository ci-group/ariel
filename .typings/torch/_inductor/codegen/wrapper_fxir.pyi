import dataclasses
import sympy
import torch
from .. import config as config, ir as ir
from ..utils import LineContext as LineContext, convert_shape_to_symint as convert_shape_to_symint, convert_to_symint as convert_to_symint
from .common import CodegenSymbol as CodegenSymbol, FileBackedGraphModule as FileBackedGraphModule, WorkspaceArg as WorkspaceArg, WorkspaceZeroMode as WorkspaceZeroMode
from .wrapper import AllocateLine as AllocateLine, BufferLike as BufferLike, CommBufferAllocateLine as CommBufferAllocateLine, CommBufferFreeLine as CommBufferFreeLine, CommentLine as CommentLine, EnterDeviceContextManagerLine as EnterDeviceContextManagerLine, EnterSubgraphLine as EnterSubgraphLine, ExitDeviceContextManagerLine as ExitDeviceContextManagerLine, ExitSubgraphLine as ExitSubgraphLine, ExternKernelAllocLine as ExternKernelAllocLine, ExternKernelOutLine as ExternKernelOutLine, FreeIfNotReusedLine as FreeIfNotReusedLine, FreeLine as FreeLine, KernelCallLine as KernelCallLine, KernelDefinitionLine as KernelDefinitionLine, Line as Line, MultiOutputLine as MultiOutputLine, NullLine as NullLine, PythonWrapperCodegen as PythonWrapperCodegen, ReinterpretLine as ReinterpretLine, ReuseLine as ReuseLine, SymbolicCallArg as SymbolicCallArg, SymbolicCallArgLine as SymbolicCallArgLine, WrapperLine as WrapperLine
from _typeshed import Incomplete
from collections import Counter
from torch._higher_order_ops.triton_kernel_wrap import TraceableTritonKernelWrapper as TraceableTritonKernelWrapper, tracing_triton_hopifier_singleton as tracing_triton_hopifier_singleton, triton_kernel_wrapper_mutation as triton_kernel_wrapper_mutation
from torch._inductor.codecache import PyCodeCache as PyCodeCache
from torch._inductor.runtime.triton_heuristics import CachingAutotuner as CachingAutotuner
from torch._inductor.select_algorithm import extern_kernels as extern_kernels
from torch._inductor.utils import sympy_product as sympy_product
from torch._inductor.virtualized import V as V
from torch._library.triton import wrap_triton as wrap_triton
from torch.fx import GraphModule as GraphModule
from torch.utils._sympy.functions import FloorDiv as FloorDiv
from typing import Any, Callable

aten: Incomplete
log: Incomplete

@dataclasses.dataclass
class SymbolBuffer(CodegenSymbol):
    """
    Represents a sympy.Symbol graph input.
    """
    symbol: sympy.Symbol
    def get_name(self) -> str: ...
    def get_example(self) -> torch.Tensor | sympy.Symbol: ...
CodegenBuffer = BufferLike | SymbolBuffer

@dataclasses.dataclass
class TritonKernel:
    """
    Stores metadata about Triton kernels for use in FX.
    """
    tuner: CachingAutotuner
    wrapped: TraceableTritonKernelWrapper

class WrapperFxCodegen(PythonWrapperCodegen):
    """
    Backend to generate wrapper code as an FX IR graph.
    """
    supports_caching: bool
    def _generate(self, is_inference: bool) -> tuple[FileBackedGraphModule, None]: ...
    def compile_graph(self, gm: GraphModule) -> Callable[..., Any]:
        """
        Converts the graph module into a runnable function. The default implementation
        is simply an interpreter calling kernels in eager mode. Derived backends can
        override this to do further compilation.
        """
    @classmethod
    def create(cls, is_subgraph: bool, subgraph_name: str | None, parent_wrapper: PythonWrapperCodegen | None, partition_signatures: ir.GraphPartitionSignature | None = None) -> WrapperFxCodegen: ...

@dataclasses.dataclass
class FxConverter:
    """
    Generates FX IR from Wrapper IR. As each instance is only meant to be used once, the
    input and output code are stored as attributes.
    """
    lines: list[Line]
    prologue: str = ...
    gm = ...
    buffer_to_node: dict[str | None, torch.fx.Node] = ...
    kernels: dict[str, TritonKernel] = ...
    _unique_symbol_ids: Counter[str] = ...
    def __post_init__(self) -> None: ...
    def _import_kernel(self, code: str, kernel_name: str) -> CachingAutotuner:
        """
        Imports a kernel from source, possibly autotuning block parameters.
        """
    def _fake_tensor(self, size: tuple[Any, ...], stride: tuple[Any, ...], dtype: torch.dtype | None = None, device: torch.device | None = None) -> torch.Tensor: ...
    def _create_meta_from_buffer(self, node: torch.fx.Node, buffer: CodegenBuffer) -> None: ...
    def _create_as_strided(self, input_node: torch.fx.Node, size: tuple[Any, ...], stride: tuple[Any, ...], offset: int | sympy.Expr) -> torch.fx.Node: ...
    def _record_allocation(self, buffer: CodegenBuffer, node: torch.fx.Node) -> None:
        """
        Updates the symbol table to record that an Inductor buffer maps to the result of
        an FX node.
        """
    def _free(self, buffer: CodegenBuffer | ir.TorchBindObject) -> None:
        """
        Removes the buffer from the symbol table.
        """
    def _lookup_args(self, args: tuple[Any, ...]) -> tuple[Any, ...]:
        """
        Maps call args back to FX nodes.
        """
    def _get_buffer(self, node: ir.IRNode) -> CodegenBuffer:
        """
        Extract buffer data from an IR node.
        """
    def _generate_graph_inputs(self) -> None:
        """
        Converts graph inputs to FX placeholders.
        """
    def _generate_buffer(self, node: ir.IRNode) -> torch.fx.Node | None:
        """
        Generates FX IR for transformations on a buffer, such as ReinterpretView.
        Does nothing if no such transformations are present.
        """
    def _generate_output(self) -> None:
        """
        Generate FX IR for graph outputs.
        """
    def generate(self) -> torch.fx.GraphModule:
        """
        Main entrypoint for FX codegen.
        """
    def _generate_allocate(self, line: WrapperLine) -> None: ...
    def _generate_comment(self, line: WrapperLine) -> None: ...
    def _generate_enter_device_context_manager(self, line: WrapperLine) -> None: ...
    def _generate_exit_device_context_manager(self, line: WrapperLine) -> None: ...
    def _generate_enter_subgraph(self, line: WrapperLine) -> None: ...
    def _generate_exit_subgraph(self, line: WrapperLine) -> None: ...
    def _generate_free(self, line: WrapperLine) -> None: ...
    def _generate_free_if_not_reused(self, line: WrapperLine) -> None: ...
    def _generate_line_context(self, line: WrapperLine) -> None: ...
    def _generate_reinterpret(self, line: WrapperLine) -> None: ...
    def _generate_reinterpret_helper(self, input_buffer: BufferLike, result_buffer: BufferLike, layout: ir.Layout) -> None: ...
    def _generate_reuse(self, line: WrapperLine) -> None: ...
    def _generate_multi_output(self, line: WrapperLine) -> None: ...
    def _generate_null(self, line: WrapperLine) -> None: ...
    def _generate_comm_buffer_allocate(self, line: WrapperLine) -> None: ...
    def _generate_comm_buffer_free(self, line: WrapperLine) -> None: ...
    def _generate_triton_call(self, line: WrapperLine) -> None: ...
    def _generate_extern_kernel_alloc(self, line: WrapperLine) -> None: ...
    def _generate_extern_kernel_out(self, line: WrapperLine) -> None: ...
    def _generate_extern_kernel_common(self, kernel: ir.ExternKernel, out_ir_node: ir.IRNode) -> None:
        """
        Generates FX IR from either ExternKernelAlloc or ExternKernelOut.
        """
    def _generate_kernel_call(self, line: WrapperLine) -> None: ...
    def _generate_kernel_definition(self, line: WrapperLine) -> None: ...
    def _generate_symbolic_call_arg(self, line: WrapperLine) -> None: ...
