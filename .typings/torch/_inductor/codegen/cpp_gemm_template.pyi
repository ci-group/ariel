import torch
from .. import config as config, ir as ir
from ..._dynamo.utils import counters as counters
from ..kernel.mm_common import mm_args as mm_args
from ..select_algorithm import DataProcessorTemplateWrapper as DataProcessorTemplateWrapper
from ..utils import has_free_symbols as has_free_symbols, is_same_mkldnn_tensor as is_same_mkldnn_tensor, is_same_tensor as is_same_tensor, parallel_num_threads as parallel_num_threads
from ..virtualized import V as V, ops as ops
from .cpp import get_export_declaration as get_export_declaration
from .cpp_micro_gemm import CppMicroBrgemm as CppMicroBrgemm, CppMicroGemm as CppMicroGemm, CppMicroGemmAMX as CppMicroGemmAMX, CppMicroGemmFP32Vec as CppMicroGemmFP32Vec, LayoutType as LayoutType, create_micro_gemm as create_micro_gemm, is_int8_woq_gemm_small_m_dim_corner_case as is_int8_woq_gemm_small_m_dim_corner_case
from .cpp_template import CppTemplate as CppTemplate
from .cpp_template_kernel import CppTemplateKernel as CppTemplateKernel
from .cpp_utils import DTYPE_TO_CPP as DTYPE_TO_CPP, GemmBlocking as GemmBlocking, create_epilogue_with_attr as create_epilogue_with_attr, get_gemm_template_output_and_compute_dtype as get_gemm_template_output_and_compute_dtype
from _typeshed import Incomplete
from torch.utils._ordered_set import OrderedSet as OrderedSet
from typing import Any, Callable, TypeVar

log: Incomplete
GEMM_TEMPLATE_INIT_BLOCKING_BASIC_BLOCK: str
GEMM_TEMPLATE_INIT_BLOCKING_EXTENDED: str
GEMM_TEMPLATE_MULTI_THREADS_PARAMS: str
GEMM_TEMPLATE_SINGLE_THREAD_PARAMS: str
GEMM_TEMPLATE_M_LOOP_PARAMS: str
GEMM_TEMPLATE_N_LOOP_PARAMS: str
GEMM_TEMPLATE_MICROKERNEL_DEF: str
GEMM_TEMPLATE_STUB_DEF: str
GEMM_TEMPLATE: str
SMALL_M_GEMM_TEMPLATE: str

def _is_int8_gemm(inputs): ...
def get_padded_n(n, block_n): ...
_T = TypeVar('_T', ir.IRNode, torch.Tensor)

def transpose_w(W: _T, trans_w: bool) -> _T:
    """
    Transpose W based on the trans_w flag.
    """
def expand_bias(B: _T | None, X: _T) -> _T | None:
    """
    Expand Bias to the same size of X.
    """
def prune_tensors(input_nodes: list[ir.IRNode], new_input_nodes: list[ir.IRNode]):
    """
    Prune unused tensors from `V.graph` since the GEMM Template use new packed weight.
    """
def gen_2d_view_of_epilogue_buf(Y: ir.Buffer, template_buffer: ir.Buffer, epilogue_nodes: list[ir.IRNode], reindexers: list[Callable[[list[Any]], list[Any]] | None], default_reindexers: list[Callable[[list[Any]], list[Any]] | None]) -> tuple[ir.Buffer | ir.ReinterpretView, list[Callable[[list[Any]], list[Any]] | None]]:
    """
    The dimension and the indexing could be different between the GEMM output, i.e. `template_buffer`, which is
    2D with MxN) and the output from the template after epilogues, i.e. `Y`. In the GEMM template code,
    we are not aware of the dimension and the indexing of the epilogues and always work on 2D tiles according to
    the indexing of the GEMM output.
    In this function, we return a 2D buffer (`Y_2d`) according to GEMM output (reinterpreted from `Y` if needed) and
    build a reindexer that converts the indexing of `Y` into `Y_2d`.
    """

class CppGemmTemplate(CppTemplate):
    """
    GEMM Template for Inductor CPP Backend.
    """
    beta: Incomplete
    alpha: Incomplete
    has_bias: Incomplete
    register_blocking: Incomplete
    padded_n: Incomplete
    is_dynamic_M: Incomplete
    should_block_weights: Incomplete
    thread_blocking: Incomplete
    cache_blocking: Incomplete
    def __init__(self, input_nodes, layout: ir.Layout, num_threads: int, register_blocking: GemmBlocking, beta: int = 1, alpha: int = 1, has_bias: bool = False, epilogue_creator: Callable[[ir.Buffer], ir.Pointwise] | None = None, should_block_weights: bool = True, name: str = 'packed_gemm') -> None: ...
    def make_thread_blocking_cache(self): ...
    def _thread_blocking(self, num_threads: int) -> GemmBlocking:
        """
        NOTE [Thread blocking in Cpp GEMM]
        We use simple heuristics to decide the thread blocking:
        1. Make sure all threads are occupied as much as possible.
        2. For (m, n) blocks, favor more square-sized thread blocks for better data reuse.
        3. If (m, n) blocks cannot occupy all the threads, we consider k-slicing.
        TODO(jgong5): allow tuning various blocking options
        """
    def make_cache_blocking_cache(self): ...
    def _cache_blocking(self, num_threads: int) -> GemmBlocking: ...
    def log_blockings(self): ...
    def maybe_k_slicing(self): ...
    @classmethod
    def add_choices(cls, choices, layout, input_nodes, beta: int = 1, alpha: int = 1, has_bias: bool = False, trans_w: bool = False, input_indices=None, epilogue_creator: Callable[[ir.Buffer], ir.Pointwise] | None = None, act_mapping: dict[int, ir.IRNode] | None = None):
        """
        Add choices for the GEMM template.
        """
    @staticmethod
    def get_padded_size(n, block_n, k, should_block_weight): ...
    @classmethod
    def prep_weight(cls, inputs, layout: ir.Layout, micro_gemm: CppMicroGemm, should_block_weight: bool, use_int8_fast_compensation_path: bool = False, skip_int8_compensation: bool = False):
        """
        NOTE Weight prep consists of 2 separate steps:
        1. Blocking the weight tensor into a 3D shape: [n//block_n, k, block_n]
           This is always done if the weight tensor is constant, i.e. for all GEMM and some BMM.
           For BMM, we also block non-contiguous weight tensors, since they would be reshaped anyway.
           This assumes that blocked, contiguous weights will be more efficient for the GEMM kernel,
           and is worth the overhead of reshape and blocking.

           This blocking includes additional padding, when n is not a multiple of block_n.
           This padding allows a more efficient microkernel implementation. For BMM, this is only done
           if reshape would happen anyway, i.e.  if the weight tensor is constant, is not contiguous,
           or is using AMX VNNI layout.
        2. Packing the weight tensor into a VNNI-friendly shape. For constant input,
           this is done at the same time as the weight blocking.

        At compile time, the constant weight tensors are blocked and packed. For non-constant tensors (e.g. BMM)
        which will be blocked (non-contiguous or VNNI-layout tensors), the weight tensor is blocked and packed at runtime.

        CppBmmTemplate overrides the methods get_padded_size, and block_weight in order to accommodate
        an additional dimension for the batch size and to determine if the weight tensor should be blocked.
        """
    @staticmethod
    def check_if_block_weight(W, micro_gemm): ...
    @classmethod
    def block_weight(cls, W, new_size, padding): ...
    @classmethod
    def pack_vnni_weight(cls, W, micro_gemm, new_size): ...
    def get_default_reindexers(self, epilogue_nodes): ...
    def get_options(self, kernel: CppTemplateKernel, template_buffer_node: ir.CppTemplateBuffer | None = None, flag_template_buffer_has_other_users: bool | None = None, epilogue_nodes: list[ir.IRNode] | None = None) -> dict[str, Any]: ...
    def is_int8_woq_gemm_small_m_dim(self, X: ir.ReinterpretView, W: ir.ReinterpretView, N, K, micro_gemm):
        """Use SMALL_M_GEMM_TEMPLATE"""
    render_options: Incomplete
    def render(self, kernel: CppTemplateKernel, template_buffer_node: ir.CppTemplateBuffer | None = None, flag_template_buffer_has_other_users: bool | None = None, epilogue_nodes: list[ir.IRNode] | None = None, **kwargs) -> str: ...
    def codegen_blocks(self, num_threads, N, K, micro_gemm, is_dynamic_M, kernel, GemmOut, config, L1_cache_size, L2_cache_size, X, W): ...
    def codegen_microkernel_def(self): ...
    def codegen_gemm_stub_def(self): ...
    def codegen_multi_threads_params(self): ...
    def codegen_single_thread_params(self, is_dynamic_M): ...
    def codegen_m_loop_params(self): ...
    def codegen_n_loop_params(self): ...
    @classmethod
    def is_woq_int4(cls): ...
    @classmethod
    def q_group_size(cls) -> None: ...

class CppWoqInt4GemmTemplateMeta(type):
    def __getitem__(cls, q_group_size): ...

class CppWoqInt4GemmTemplate(metaclass=CppWoqInt4GemmTemplateMeta): ...
