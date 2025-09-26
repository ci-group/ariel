from ...ir import Buffer as Buffer, ChoiceCaller as ChoiceCaller, IRNode as IRNode, Layout as Layout, PrimitiveInfoType as PrimitiveInfoType, TensorBox as TensorBox
from ...virtualized import V as V
from ..common import Kernel as Kernel, OpOverrides as OpOverrides, WorkspaceArg as WorkspaceArg, WorkspaceZeroMode as WorkspaceZeroMode
from ..cpp_utils import CppPrinter as CppPrinter
from .rocm_benchmark_request import ROCmBenchmarkRequest as ROCmBenchmarkRequest
from .rocm_template_buffer import ROCmTemplateBuffer as ROCmTemplateBuffer
from .rocm_utils import DTYPE_TO_ROCM_TYPE as DTYPE_TO_ROCM_TYPE
from _typeshed import Incomplete
from collections.abc import Sequence
from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu as CppWrapperCpu
from torch._inductor.codegen.rocm.rocm_template import ArgInfo as ArgInfo, ROCmTemplate as ROCmTemplate
from torch._inductor.utils import do_bench_using_profiling as do_bench_using_profiling
from typing import Any, Callable

log: Incomplete
cexpr: Incomplete

def _normalize_idx(index: int, total_length: int) -> int: ...

class ROCmKernel(Kernel):
    """
    Baseclass for ROCm based Kernels
    """
    overrides = OpOverrides

class ROCmTemplateKernel(ROCmKernel):
    """
    Template kernels defined by ROCm in C++.
    """
    _EXTRA_CPP_ARGS: str
    kernel_name: Incomplete
    named_nodes: dict[str, IRNode]
    runtime_arg_info: Incomplete
    runtime_arg_values: Incomplete
    def __init__(self, kernel_name: str, runtime_arg_info: list['ArgInfo'], runtime_arg_values: list[Any]) -> None:
        """
        Initializes a new instance of the ROCmTemplateKernel class.

        Args:
            kernel_name (str): The name of the kernel.
        """
    def get_signature(self): ...
    signature: Incomplete
    def def_kernel(self, inputs: list[IRNode], outputs: list[IRNode], size_args: list[str], names_str: str = '', input_reorder: list[int] | None = None) -> str:
        """
        Hook called from template code to generate function definition and
        needed args.

        Args:
            inputs: List of input IRNodes
            outputs: List of output IRNodes
            names_str: Comma separated list of input + output argument names.
            input_reorder: The actual order of input nodes.
                           e.g. The template might have input argument defined as [X, W, Bias],
                           and the actual input passed into this template could be [Bias, X, W].
                           In this case, the `input_reorder` would be [2, 0, 1].
        """
    def call_kernel(self, name: str, node: ROCmTemplateBuffer) -> None:
        """
        Generates code to call the kernel through V.graph.wrapper_code.
        used from within torch._inductor.wrapper.PythonWrapperCodegen

        name: Name of kernel function.
        node: The ROCmTemplateBuffer node which contains information about the kernel, it's fused epilogue nodes
        as well as all required inputs and outputs.
        """

class ROCmTemplateCaller(ChoiceCaller):
    """
    ROCmTemplateCaller

    This class represents a caller for ROCm template kernels. It is a subclass of ChoiceCaller.
    Attributes:
        name (str): The name of the caller.
        category (str): The category of the caller.
        bmreq (ROCmBenchmarkRequest): The benchmark request for the caller.
        template_buffer (ROCmTemplateBuffer): The template buffer for the caller.
    """
    category: Incomplete
    make_kernel_render: Incomplete
    bmreq: Incomplete
    template: Incomplete
    info_kwargs: Incomplete
    def __init__(self, name: str, category: str, input_nodes: list[Buffer], layout: Layout, make_kernel_render: Callable[[ROCmTemplateBuffer, Sequence[IRNode] | None], str], bmreq: ROCmBenchmarkRequest, template: ROCmTemplate, info_kwargs: dict[str, PrimitiveInfoType | list[PrimitiveInfoType]] | None) -> None: ...
    def precompile(self) -> None: ...
    def benchmark(self, *args, out) -> float: ...
    def __str__(self) -> str: ...
    def call_name(self) -> str: ...
    def hash_key(self) -> str: ...
    def info_dict(self) -> dict[str, PrimitiveInfoType | list[PrimitiveInfoType]]:
        """Information returned here is logged to the autotune log file when that is enabled."""
    def output_node(self) -> TensorBox: ...
