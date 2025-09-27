from . import config as config, lowering as lowering
from .codegen.cpp_gemm_template import CppGemmTemplate as CppGemmTemplate, CppWoqInt4GemmTemplate as CppWoqInt4GemmTemplate
from .codegen.cpp_utils import create_epilogue_with_attr as create_epilogue_with_attr
from .lowering import expand as expand, register_lowering as register_lowering
from .mkldnn_ir import WeightInt4PackMatmul as WeightInt4PackMatmul
from .select_algorithm import ExternKernelChoice as ExternKernelChoice, autotune_select_algorithm as autotune_select_algorithm, realize_inputs as realize_inputs
from .utils import use_aten_gemm_kernels as use_aten_gemm_kernels, use_cpp_gemm_template as use_cpp_gemm_template
from .virtualized import V as V
from _typeshed import Incomplete
from torch._inductor.kernel.mm_common import mm_args as mm_args

log: Incomplete
aten__weight_int8pack_mm: Incomplete
aten__weight_int4pack_mm_cpu: Incomplete
quantized: Incomplete
_quantized: Incomplete
aten: Incomplete

def register_quantized_ops() -> None: ...
def register_woq_mm_ops() -> None: ...
