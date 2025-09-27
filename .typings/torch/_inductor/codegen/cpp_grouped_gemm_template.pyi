from .. import config as config, ir as ir
from ..._dynamo.utils import counters as counters
from ..kernel.mm_common import mm_args as mm_args
from ..select_algorithm import ChoiceCaller as ChoiceCaller, DataProcessorTemplateWrapper as DataProcessorTemplateWrapper
from ..utils import parallel_num_threads as parallel_num_threads
from ..virtualized import V as V
from .cpp import get_export_declaration as get_export_declaration
from .cpp_gemm_template import CppGemmTemplate as CppGemmTemplate, expand_bias as expand_bias, gen_2d_view_of_epilogue_buf as gen_2d_view_of_epilogue_buf, prune_tensors as prune_tensors, transpose_w as transpose_w
from .cpp_micro_gemm import CppMicroGemmAMX as CppMicroGemmAMX, create_micro_gemm as create_micro_gemm
from .cpp_template_kernel import CppTemplateKernel as CppTemplateKernel
from .cpp_utils import DTYPE_TO_CPP as DTYPE_TO_CPP, GemmBlocking as GemmBlocking, create_epilogue_with_attr as create_epilogue_with_attr, get_gemm_template_output_and_compute_dtype as get_gemm_template_output_and_compute_dtype
from _typeshed import Incomplete
from torch.utils._ordered_set import OrderedSet as OrderedSet
from typing import Callable

log: Incomplete
GEMM_TEMPLATE: str

def get_deduplicated_act(act_mapping: dict[int, ir.IRNode]) -> list[ir.IRNode]: ...

class CppGroupedGemmTemplate(CppGemmTemplate):
    act_mapping: Incomplete
    gemm_grouped_num: Incomplete
    output_node: list[ir.Buffer]
    def __init__(self, input_nodes: list[ir.IRNode], layout: ir.Layout, num_threads: int, register_blocking: GemmBlocking, beta: int = 1, alpha: int = 1, has_bias: bool = False, epilogue_creator: Callable[[ir.Buffer], ir.Pointwise] | None = None, act_mapping: dict[int, ir.IRNode] | None = None, gemm_grouped_num: int = 1) -> None:
        """
        Template for Group of GEMMs:
        * Each GEMM has the same dimensions (m, n, k) and the same leading dimensions (lda, ldb, ldc)
          for their A, B, and C matrices.
        * Each GEMM has distinct or shared activations, has distinct weight, has unique bias or no bias, has distinct epilogues.
        * In the current implementation, the outputs of all GEMMs are accumulated using pointwise epilogues.
          This behavior can be extended in the future if needed.
        """
    @classmethod
    def add_choices(cls, choices: list[ChoiceCaller], layout: ir.Layout, input_nodes: list[ir.IRNode], beta: int = 1, alpha: int = 1, has_bias: tuple[bool, ...] = (False, False), trans_w: bool = False, input_indices: list[int] | None = None, epilogue_creator: Callable[[ir.Buffer], ir.Pointwise] | None = None, act_mapping: dict[int, ir.IRNode] | None = None) -> DataProcessorTemplateWrapper: ...
    def render(self, kernel: CppTemplateKernel, template_buffer_node: ir.CppTemplateBuffer | None = None, flag_template_buffer_has_other_users: bool | None = None, epilogue_nodes: list[ir.IRNode] | None = None, **kwargs) -> str: ...
