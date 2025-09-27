from ...autotune_process import CUDABenchmarkRequest as CUDABenchmarkRequest, TensorMeta as TensorMeta
from ...ir import Buffer as Buffer, CUDATemplateBuffer as CUDATemplateBuffer, IRNode as IRNode, Layout as Layout
from ...scheduler import BaseSchedulerNode as BaseSchedulerNode
from ...utils import IndentedBuffer as IndentedBuffer, unique as unique
from ...virtualized import V as V
from ..common import KernelTemplate as KernelTemplate
from .cuda_kernel import CUDATemplateCaller as CUDATemplateCaller, CUDATemplateKernel as CUDATemplateKernel
from .cutlass_utils import DTYPE_TO_CUTLASS_TYPE as DTYPE_TO_CUTLASS_TYPE
from _typeshed import Incomplete
from dataclasses import dataclass
from torch._inductor.utils import Placeholder as Placeholder
from torch._logging import getArtifactLogger as getArtifactLogger
from typing import Any
from typing_extensions import override

GemmOperation = Any
autotuning_log: Incomplete

@dataclass(frozen=True)
class ArgInfo:
    name: str
    ty: str

class CUDATemplate(KernelTemplate):
    index_counter: Incomplete
    input_nodes: Incomplete
    output_node: Buffer
    input_reorder: Incomplete
    layout: Incomplete
    def __init__(self, name: str, input_nodes: list[Buffer], layout: Layout, input_reorder: list[int] | None = None) -> None:
        """

        Baseclass for CUDA C++ Templates, derived from KernelTemplate. Not to be instantiated directly.

        Args:
            name (str): The name of the CUDATemplate object.
            input_nodes (List[IRNode]): A list of input IRNodes.
            layout (Layout): The layout of the output buffer / tensor.
            input_reorder (Optional[List[int]]): An optional list that specifies the order of the input nodes.

        """
    @staticmethod
    def supports_epilogue_fusion(op: GemmOperation) -> bool: ...
    def generate(self, description, **kwargs) -> CUDATemplateCaller:
        """
        Generates the CUDA template caller object for the given GEMM template and operation. This CUDATemplateCaller
        may be used to call and benchmark the generated CUDA kernel in a standalone manner to enable Autotuning.

        Args:
            kwargs: Additional keyword arguments.

        Returns:
            A CUDATemplateCaller object representing the generated CUDA template caller.
        """
    def header(self) -> IndentedBuffer: ...
    def globals(self) -> IndentedBuffer: ...
    def render(self, **kwargs) -> str: ...
    def get_runtime_arg_info(self) -> list[ArgInfo]: ...
    def get_runtime_arg_values(self, **kwargs) -> list[Any]: ...

class CUTLASSTemplate(CUDATemplate):
    """
    CUTLASSTemplate is a class that provides a template for generating CUTLASS Templates. Used as a baseclass for the
    CUTLASSGemmTemplate, providing functionality that might also be relevant for non-GEMM CUTLASS Kernels.
    """
    def header(self) -> IndentedBuffer: ...
    def globals(self) -> IndentedBuffer: ...
    def cute_int(self, int_str: str, var_name: str) -> str: ...
    _DTYPE_TO_CUTLASS: Incomplete
    _DTYPE_TO_CUTLASS_SPARSE_META: Incomplete
    def cutlass_type_cast(self, node: IRNode, ptr: str) -> str: ...
    def cutlass_sparse_meta_type_cast(self, node: IRNode, ptr: str) -> str: ...
    @override
    def get_runtime_arg_info(self) -> list[ArgInfo]: ...
    @override
    def get_runtime_arg_values(self, **kwargs) -> list[Any]:
        """
        Helper method to retrieve runtime args from generate kwargs
        """
