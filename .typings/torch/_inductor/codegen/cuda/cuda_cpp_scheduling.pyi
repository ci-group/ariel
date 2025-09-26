from ... import config as config
from ...._dynamo.utils import counters as counters
from ...codecache import code_hash as code_hash, get_path as get_path
from ...ir import Buffer as Buffer, CUDATemplateBuffer as CUDATemplateBuffer, ComputedBuffer as ComputedBuffer, Pointwise as Pointwise
from ...scheduler import BaseSchedulerNode as BaseSchedulerNode, BaseScheduling as BaseScheduling, FusedSchedulerNode as FusedSchedulerNode, SchedulerNode as SchedulerNode, WhyNoFuse as WhyNoFuse
from ...utils import get_fused_kernel_name as get_fused_kernel_name, get_kernel_metadata as get_kernel_metadata, sympy_product as sympy_product
from ...virtualized import V as V
from ..common import BackendFeature as BackendFeature, IndentedBuffer as IndentedBuffer
from _typeshed import Incomplete
from collections.abc import Sequence
from torch._inductor.codegen.cuda.cutlass_python_evt import CutlassEVTCodegen as CutlassEVTCodegen, MockCutlassHandler as MockCutlassHandler
from torch._inductor.utils import Placeholder as Placeholder
from torch.utils._ordered_set import OrderedSet as OrderedSet

log: Incomplete

class WhyNoFuseNames(WhyNoFuse):
    name1: Incomplete
    name2: Incomplete
    def __init__(self, name1: str, name2: str) -> None: ...

class CUDACPPScheduling(BaseScheduling):
    """
    Partial Scheduling implementation for CUDA C++ Kernels.
    This class is intended to be used in combination with TritonScheduling,
    and delegated to by CUDACombinedScheduling.

    It handles fusion decisions and CUDA C++ specific template code generation.
    """
    @classmethod
    def get_backend_features(cls, device) -> OrderedSet[BackendFeature]: ...
    def group_fn(self, sizes): ...
    @staticmethod
    def is_cuda_cpp_template(node: BaseSchedulerNode) -> bool: ...
    def is_cuda_cpp_fused_template(self, node: BaseSchedulerNode) -> bool: ...
    def can_fuse_vertical(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> bool: ...
    def define_kernel(self, src_code: str, node_schedule) -> str: ...
    def codegen_template(self, template_node: BaseSchedulerNode, epilogue_nodes: Sequence[BaseSchedulerNode], prologue_nodes: Sequence[BaseSchedulerNode]):
        """
        Codegen a CUDA template, possibly with fused epilogues
        """
    @staticmethod
    def _unwrap_epilogue_nodes(fused_node: FusedSchedulerNode) -> list[BaseSchedulerNode]: ...
    def _can_fuse_epilogue_impl(self, cuda_template_buffer: CUDATemplateBuffer, existing_epilogue_nodes: list[BaseSchedulerNode], node_to_fuse: BaseSchedulerNode) -> bool:
        """
        Check if the given node can be fused with the epilogue. At the moment, Kernels
        support fusion with Pointwise operations, wrapped in (named) ComputedBuffer nodes.

        Args:
            cuda_template_buffer : A CUDATemplateBuffer object representing the CUDA template and it's result buffer
            existing_epilogue_nodes : List[SchedulerNode]: The list of already fused epilogue nodes.
            node_to_fuse: The SchedulerNode node to be checked if it can be fused with the epilogue.
        Returns:
        - bool: True if the given node can be fused with the epilogue, False otherwise.

        """
