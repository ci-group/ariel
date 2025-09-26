from ... import config as config
from ...codecache import code_hash as code_hash, get_path as get_path
from ...scheduler import BaseSchedulerNode as BaseSchedulerNode, BaseScheduling as BaseScheduling, SchedulerNode as SchedulerNode
from ...utils import get_fused_kernel_name as get_fused_kernel_name, get_kernel_metadata as get_kernel_metadata, sympy_product as sympy_product
from ...virtualized import V as V
from ..common import IndentedBuffer as IndentedBuffer
from .rocm_template_buffer import ROCmTemplateBuffer as ROCmTemplateBuffer
from _typeshed import Incomplete
from collections.abc import Sequence

log: Incomplete

class ROCmCPPScheduling(BaseScheduling):
    """
    Partial Scheduling implementation for ROCm C++ Kernels.
    This class is intended to be used in combination with TritonScheduling,
    and delegated to by CUDACombinedScheduling.

    It handles fusion decisions and ROCm C++ specific template code generation.
    """
    def group_fn(self, sizes): ...
    @staticmethod
    def is_rocm_cpp_template(node: BaseSchedulerNode) -> bool: ...
    def can_fuse_vertical(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> bool: ...
    def define_kernel(self, src_code: str, node_schedule) -> str: ...
    def codegen_template(self, template_node: BaseSchedulerNode, epilogue_nodes: Sequence[BaseSchedulerNode], prologue_nodes: Sequence[BaseSchedulerNode]):
        """
        Codegen a ROCm template, possibly with fused epilogues
        """
