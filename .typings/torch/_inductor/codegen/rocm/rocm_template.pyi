from ...autotune_process import TensorMeta as TensorMeta
from ...ir import Buffer as Buffer, IRNode as IRNode, Layout as Layout
from ...utils import IndentedBuffer as IndentedBuffer, unique as unique
from ...virtualized import V as V
from ..common import KernelTemplate as KernelTemplate
from .rocm_benchmark_request import ROCmBenchmarkRequest as ROCmBenchmarkRequest
from .rocm_kernel import ROCmTemplateCaller as ROCmTemplateCaller, ROCmTemplateKernel as ROCmTemplateKernel
from .rocm_template_buffer import ROCmTemplateBuffer as ROCmTemplateBuffer
from .rocm_utils import DTYPE_TO_ROCM_TYPE as DTYPE_TO_ROCM_TYPE
from _typeshed import Incomplete
from dataclasses import dataclass
from typing import Any

log: Incomplete

@dataclass(frozen=True)
class ArgInfo:
    name: str
    ty: str

class ROCmTemplate(KernelTemplate):
    index_counter: Incomplete
    gfx9_threads_per_warp: int
    input_nodes: Incomplete
    output_node: Buffer
    input_reorder: Incomplete
    layout: Incomplete
    def __init__(self, name: str, input_nodes: list[Buffer], layout: Layout, input_reorder: list[int] | None = None) -> None:
        """

        Baseclass for ROCm C++ Templates, derived from KernelTemplate. Not to be instantiated directly.

        Args:
            name (str): The name of the ROCmTemplate object.
            input_nodes (List[IRNode]): A list of input IRNodes.
            layout (Layout): The layout of the output buffer / tensor.
            input_reorder (Optional[List[int]]): An optional list that specifies the order of the input nodes.

        """
    def generate(self, **kwargs) -> ROCmTemplateCaller:
        """
        Generates the ROCm template caller object for the given GEMM template and operation. This ROCmTemplateCaller
        may be used to call and benchmark the generated ROCm kernel in a standalone manner to enable Autotuning.

        Args:
            kwargs: Additional keyword arguments.

        Returns:
            A ROCmTemplateCaller object representing the generated ROCm template caller.
        """
    def header(self) -> IndentedBuffer: ...
    def globals(self) -> IndentedBuffer: ...
    def render(self, **kwargs) -> str: ...
    def get_runtime_arg_info(self) -> list[ArgInfo]: ...
    def get_runtime_arg_values(self, **kwargs) -> list[Any]: ...
