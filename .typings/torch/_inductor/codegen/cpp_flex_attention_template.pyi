from .. import ir as ir
from ...utils._ordered_set import OrderedSet as OrderedSet
from ..ir import TensorBox as TensorBox
from ..select_algorithm import DataProcessorTemplateWrapper as DataProcessorTemplateWrapper
from ..utils import parallel_num_threads as parallel_num_threads
from ..virtualized import V as V
from .cpp_template import CppTemplate as CppTemplate
from .cpp_utils import GemmBlocking as GemmBlocking
from _typeshed import Incomplete

log: Incomplete
SOFTMAX_FUSIONS: str
BRGEMM_PACK_FUNCTIONS: str
MICRO_GEMM_TEMPLATE: str
ALLOCATE_BUFFER: str
FLEX_ATTENTION_TEMPLATE: str

class CppFlexAttentionTemplate(CppTemplate):
    scale: Incomplete
    score_mod: Incomplete
    mask_mod: Incomplete
    score_buf_name: Incomplete
    mask_buf_name: Incomplete
    score_buf_idx: Incomplete
    mask_buf_idx: Incomplete
    kv_block_size: Incomplete
    q_block_size: Incomplete
    has_other_buffer: Incomplete
    no_full_kv_block: Incomplete
    other_buffer_input_offset: int
    fake_buffers: Incomplete
    len_score_other: Incomplete
    len_mask_other: Incomplete
    kernel_input_name_to_buffer: Incomplete
    block_vars: Incomplete
    extra_sizevars: Incomplete
    other_buf_start_idx: int
    score_mod_other_buffers: Incomplete
    mask_mod_other_buffers: Incomplete
    other_ptr_data: Incomplete
    def __init__(self, input_nodes, layout: ir.Layout, scale, score_mod, mask_mod, kv_block_size, q_block_size, has_other_buffer, no_full_kv_block, fake_buffers, len_score_other, len_mask_other, kernel_input_name_to_buffer, block_vars) -> None: ...
    def update_kernel_args(self, kernel_args): ...
    def generate_other_buffer(self, buf_list, start_offset, len_attr, kernel_args): ...
    def modification(self, subgraph_buffer, output_name, output_idx): ...
    @staticmethod
    def add_choices(choices, input_nodes, layout, scale, score_mod, mask_mod, kv_block_size, q_block_size, has_other_buffer, no_full_kv_block, fake_buffers, len_score_other, len_mask_other, kernel_input_name_to_buffer, block_vars): ...
    def apply_score_mod(self, score, b, h, q_idx, kv_idx): ...
    accumulate_dtype: Incomplete
    input_dtype: Incomplete
    def render(self, kernel, template_buffer_node: ir.CppTemplateBuffer | None = None, epilogue_nodes: list[ir.IRNode] | None = None, **kwargs) -> str: ...
    def codegen_softmax_fusion(self, kernel_name: str): ...
    def codegen_brgemm_pack_function(self, kernel_name: str): ...
    def codegen_allocate_buffer(self, buffer_name: str, buffer_dtype, buffer_size): ...
    def micro_gemm_define(self, kernel_name: str): ...
    def codegen_micro_gemm(self, kernel_name: str): ...
