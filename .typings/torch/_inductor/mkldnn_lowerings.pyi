import torch
from . import config as config, ir as ir
from .codegen.cpp_gemm_template import CppGemmTemplate as CppGemmTemplate
from .codegen.cpp_grouped_gemm_template import CppGroupedGemmTemplate as CppGroupedGemmTemplate
from .codegen.cpp_utils import create_epilogue_with_attr as create_epilogue_with_attr
from .ir import TensorBox as TensorBox
from .lowering import add as add, add_needs_realized_inputs as add_needs_realized_inputs, aten as aten, permute as permute, register_lowering as register_lowering, to_dtype as to_dtype, view as view
from .select_algorithm import ChoiceCaller as ChoiceCaller, ExternKernelChoice as ExternKernelChoice, autotune_select_algorithm as autotune_select_algorithm
from .utils import use_aten_gemm_kernels as use_aten_gemm_kernels, use_cpp_gemm_template as use_cpp_gemm_template
from .virtualized import OpsValue as OpsValue, V as V, ops as ops
from torch._inductor.kernel.mm_common import mm_args as mm_args

def create_int8_compensation(W_tensor: torch.Tensor, packed_weight: ir.TensorBox, x_scale: ir.TensorBox, x_zp: ir.TensorBox, w_scale: ir.TensorBox) -> tuple[bool, ir.TensorBox, ir.TensorBox | None]: ...
def codegen_int8_gemm_template_compensation(use_int8_fast_compensation_path: bool, input: OpsValue, _weight_compo: OpsValue, _x_scale: OpsValue | None, _x_zp: OpsValue | None, _w_scale: OpsValue | None, _x_w_scale: OpsValue | None) -> OpsValue: ...
def grouped_gemm_lowering(x: TensorBox, w: list[TensorBox], b: list[TensorBox], attr=None, scalars=None, algorithm=None, layout=None): ...
def register_onednn_fusion_ops(): ...
