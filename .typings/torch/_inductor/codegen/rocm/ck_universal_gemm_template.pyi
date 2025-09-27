from ...utils import IndentedBuffer as IndentedBuffer, is_dynamic as is_dynamic, try_import_ck_lib as try_import_ck_lib
from _typeshed import Incomplete
from torch._inductor import config as config
from torch._inductor.codegen.cpp_utils import DTYPE_TO_CPP as DTYPE_TO_CPP
from torch._inductor.codegen.rocm.ck_template import CKTemplate as CKTemplate
from torch._inductor.codegen.rocm.compile_command import rocm_compile_command as rocm_compile_command
from torch._inductor.codegen.rocm.rocm_kernel import ROCmTemplateKernel as ROCmTemplateKernel
from torch._inductor.ir import Buffer as Buffer, Layout as Layout
from torch._inductor.runtime.runtime_utils import next_power_of_2 as next_power_of_2
from typing import NamedTuple

_: Incomplete
gen_ops_library: Incomplete
gen_ops_preselected: Incomplete
CKGemmOperation: Incomplete
log: Incomplete

class InductorROCmOp(NamedTuple):
    op: Incomplete
    kBatch: Incomplete

padding_lookup: Incomplete

def is_static_int(number): ...
def torch_layout_to_ck_layout(torch_layout): ...

class CKGemmTemplate(CKTemplate):
    gemm_template: str
    standalone_runner_template: str
    alpha: Incomplete
    beta: Incomplete
    is_batched: Incomplete
    def __init__(self, input_nodes: list[Buffer], layout: Layout, alpha: float, beta: float, input_reorder: list[int] | None = None) -> None: ...
    def header(self) -> IndentedBuffer: ...
    def globals(self) -> IndentedBuffer: ...
    def inline_utils(self): ...
    def _has_padding(self, dimension, gemm_specialization): ...
    def filter_op(self, op_info: InductorROCmOp):
        """
        Determines whether a given op definition is suitable for the current
        input / output of the operation that this template implements.

        Filter is based on inputs' dtype, layout and statically inferred size.

        Returns None if the op is not suitable, otherwise returns the op to be used.
        """
    def _check_num_k_loops(self, op, kBatch): ...
    def _prefetch_stages(self, op, a_dtype_size, b_dtype_size, warp_size: int = 64): ...
    def emit_ck_instance(self, op: CKGemmOperation): ...
    output_node: Incomplete
    def render(self, kernel: ROCmTemplateKernel, op: CKGemmOperation, **kwargs) -> str:
        """
        The primary entry point for the code rendering process used in this template.
        """
    def _is_rcr_f16(self): ...
    def _get_kBatch(self, op): ...
    def gen_ops(self) -> list[InductorROCmOp]:
        """
        Creates a list of `CKGemmOperation` instances that match the GEMM operation this template represents.
        The instances are guaranteed to have the correct layout, dtype and dimension padding for the GEMM input arguments.

        An instance may invalidate the GEMM configuration at runtime.
        Such instances will be assigned +inf runtime by the autotune process.
        """
    @staticmethod
    def add_ck_gemm_choices(choices, layout, input_nodes, alpha: int = 1, beta: int = 0, input_reorder=None) -> None:
        """
        Add Composable Kernel Universal GEMM instance choices to the auto-tuning list.
        """
    def size_args(self): ...
