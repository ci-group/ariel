from ..cutlass_utils import torch_dtype_to_cutlass_type as torch_dtype_to_cutlass_type, try_import_cutlass as try_import_cutlass
from _typeshed import Incomplete
from cutlass.backend.evt.ir.tensor import Tensor as CutlassTensor
from cutlass_library import DataType as DataType, EpilogueScheduleType as EpilogueScheduleType, TileDescription as TileDescription
from sympy import Expr as Expr
from torch._inductor.codegen.cuda import cuda_env as cuda_env
from torch._inductor.ir import ComputedBuffer as ComputedBuffer, InputBuffer as InputBuffer, is_contiguous_strides_for_shape as is_contiguous_strides_for_shape
from torch._inductor.utils import IndentedBuffer as IndentedBuffer
from torch.utils._ordered_set import OrderedSet as OrderedSet
from typing import Any, Callable

EpilogueFunctor = Any
Buffer = ComputedBuffer | InputBuffer
CutlassTupleType = Any
CutlassVisitorType = Any
CutlassArgType = Any
_CUTLASS_C_DTYPES: Incomplete

def create_example_tensors(var_name_to_buffer_name: dict[str, str], name_to_buffer: dict[str, Buffer], size_hint_fn: Callable[[Expr | int], int]) -> dict[str, CutlassTensor]: ...
def trace(fn_src: str, example_tensors: dict[str, CutlassTensor], accum_type: DataType, output_type: DataType, tile_description: TileDescription, epilogue_schedule: EpilogueScheduleType, name_to_buffer: dict[str, Buffer], size_hint_fn: Callable[[Expr | int], int], **kwargs: dict[str, Any]) -> tuple[str, str, str]: ...
def _trace(fn_src: str, example_tensors: dict[str, CutlassTensor], cc: int, **kwargs: Any) -> EpilogueFunctor: ...
def _render_argument_type(epilogue_functor: EpilogueFunctor, name_to_buffer: dict[str, Buffer], size_hint_fn: Callable[[Expr | int], int]) -> str: ...
def _get_arg_from_node(arg_ty: type, node: Buffer, size_hint_fn: Callable[[Expr | int], int]) -> str: ...
