from .. import ir as ir
from ..select_algorithm import PartialRender as PartialRender
from ..virtualized import V as V
from .common import ArgName as ArgName
from .cpp_gemm_template import CppGemmTemplate as CppGemmTemplate, GEMM_TEMPLATE as GEMM_TEMPLATE
from .cpp_micro_gemm import LayoutType as LayoutType
from .cpp_template_kernel import CppTemplateKernel as CppTemplateKernel
from .cpp_utils import DTYPE_TO_CPP as DTYPE_TO_CPP, GemmBlocking as GemmBlocking
from _typeshed import Incomplete
from typing import Any, Callable

GEMM_SINGLE_THREAD_MM_STUB: str
GEMM_THREADED_MM_STUB: str
BMM_TEMPLATE: str

class CppBmmTemplate(CppGemmTemplate):
    b_index: Incomplete
    def __init__(self, input_nodes, layout: ir.Layout, num_threads: int, register_blocking: GemmBlocking, beta: int = 1, alpha: int = 1, has_bias: bool = False, epilogue_creator: Callable[[ir.Buffer], ir.Pointwise] | None = None, should_block_weights: bool = False, name: str = 'bmm') -> None:
        """
        In order to simplify the implementation and increase code reuse, the BMM template implements
        two versions of the GEMM kernel: a single-threaded version and a multi-threaded version.
        GEMM kernels are called in a loop over the batch dimension, with single-threaded GEMM calls
        for all but the last (B % num_threads), which are handled by the multi-threaded GEMM kernel.

        We use an extra sizevar `b_index` to index the batch dimension, which we pass into the GEMM
        template as a sympy.Symbol. This allows us to slice the 3D batch tensors in the GEMM template
        without any changes to the GEMM template itself.
        """
    @staticmethod
    def get_padded_size(n, block_n, k, should_block_weight): ...
    @staticmethod
    def check_if_block_weight(W, micro_gemm): ...
    def get_gemm_function_call(self, kernel: CppTemplateKernel, function_name: str, placeholder: str, b_index: str) -> str:
        """
        Similar to 'def_kernel' in cpp_template_kernel, but instead of generating a function definition,
        generate a function call for the GEMM kernel.
        Args:
            placeholder: The string to replace the function call with
            b_index: The index for slicing the 3D batch tensors
        """
    def get_default_reindexers(self, epilogue_nodes): ...
    def get_options(self, kernel: CppTemplateKernel, template_buffer_node: ir.CppTemplateBuffer | None = None, flag_template_buffer_has_other_users: bool | None = None, epilogue_nodes: list[ir.IRNode] | None = None, **kwargs) -> dict[str, Any]: ...
    render_options: Incomplete
    def render(self, kernel: CppTemplateKernel, template_buffer_node: ir.CppTemplateBuffer | None = None, flag_template_buffer_has_other_users: bool | None = None, epilogue_nodes: list[ir.IRNode] | None = None, **kwargs) -> str: ...
    def codegen_single_thread_gemm(self): ...
    def codegen_multi_thread_gemm(self): ...
    def codegen_gemm_stub_def(self): ...
