import torch
import torch._C._onnx as _C_onnx
from _typeshed import Incomplete
from collections.abc import Sequence
from torch import _C as _C
from torch.onnx import _constants as _constants, _type_utils as _type_utils, errors as errors, utils as utils
from torch.onnx._globals import GLOBALS as GLOBALS
from torch.onnx._internal import jit_utils as jit_utils
from torch.types import Number as Number
from typing import Any, Callable, NoReturn, TypeVar as _TypeVar
from typing_extensions import Concatenate as _Concatenate, ParamSpec as _ParamSpec

_T = _TypeVar('_T')
_U = _TypeVar('_U')
_P = _ParamSpec('_P')
_ValueDescriptor: Incomplete

def _parse_arg(value, desc: _ValueDescriptor, arg_name: str | None = None, node_name: str | None = None): ...
def _node_get(node: _C.Node, key: str):
    """Gets attributes of a node which is polymorphic over return type."""
def _is_onnx_constant(value: _C.Value):
    """Whether a Value is an ONNX constant."""
def _maybe_get_const(value: _C.Value | torch.Tensor | Number | Sequence | None, descriptor: _ValueDescriptor): ...
def _maybe_get_scalar(value): ...
def _get_const(value, desc, arg_name): ...
def _unpack_list(list_value: _C.Value) -> list[_C.Value]: ...
def _unpack_tuple(tuple_value: _C.Value) -> tuple[_C.Value, ...]: ...
def _unpack_quantized_tensor(tuple_value: _C.Value) -> tuple[_C.Value, ...]:
    """Unpacks a quantized tensor into a tuple of tensor and scale/zero_point.
    Args:
        tuple_value: A tuple of tensor, scale, zero_point, and optionally axis.
    Returns:
        A tuple of tensor, scale, zero_point, and optionally axis.
    """
def _is_packed_list(list_value: Any) -> bool: ...
def parse_args(*arg_descriptors: _ValueDescriptor) -> Callable[[Callable[_Concatenate[_U, _P], _T]], Callable[_Concatenate[_U, _P], _T]]:
    '''A decorator which converts args from torch._C.Value to built-in types.

    For example:

    ```
    @parse_args(\'v\', \'i\', \'fs\')
    foo(g, a, b, c):
        assert isinstance(a, torch._C.Value)
        assert isinstance(b, int)
        assert isinstance(c, list)
        assert isinstance(c[0], float)
    ```

    Args:
        arg_descriptors: list of str, where each element is
            a string that specifies the type to convert to. Valid descriptors:
            "v": no conversion, keep torch._C.Value.
            "i": int
            "is": list of int
            "f": float
            "fs": list of float
            "b": bool
            "s": str
            "t": torch.Tensor
            "none": the variable is unused
    '''
def quantized_args(*arg_q_descriptors: bool, scale: float | None = None, zero_point: int | None = None, quantize_output: bool = True) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """A decorator which extends support for quantized version of the base operator.

    Quantization is detected by examining the arguments that are annotated by
    `arg_q_descriptors`.

    If quantization is detected, the base operator symbolic function will be wrapped with
    argument de-quantization and output quantization.

    Otherwise, only the base symbolic function will be invoked.

    For example:

    ```
    @quantized_args(True, False)
    def foo(g, x, y):
        return x + y
    ```

    is equivalent to

    ```
    def q_foo(g, x, y):
        if is_quantized_tensor(x):
            x = dequantize(x)
            out = foo(g, x, y)
            return quantize(out)
        else:
            return foo(g, x, y)
    ```

    Args:
        arg_q_descriptors: A sequence of bool, where each element represents if the
          argument is QTensor for quantized version of this operator. It defaults
          to False for unspecified (variable length) arguments.
        scale: Quantized output scale. If None, derive from
          the first quantized input scale.
        zero_point: Quantized output zero point. If None,
          derive from the first quantized input zero point.
        quantize_output: If True, quantize the output of the base operator. Default is True
    """
def _scalar(x: Any) -> Number | None:
    """Convert a scalar tensor into a Python value."""
def _if_scalar_type_as(self, tensor):
    """
    Convert self into the same type of tensor, as necessary.
    We only support implicit casting for scalars, so we never
    actually need to insert an ONNX cast operator here; just
    fix up the scalar.
    """
def _is_none(x: Any) -> bool: ...
def _is_value(x: Any) -> bool: ...
def _is_constant(value: Any) -> bool: ...
def _is_tensor(x: _C.Value) -> bool: ...
def _as_list_type(jit_type: _C.JitType) -> _C.ListType | None: ...
def _is_list(x: _C.Value) -> bool: ...
def _is_tensor_list(x: _C.Value) -> bool: ...
def _is_scalar_list(x: _C.Value) -> bool:
    """Checks if x is a scalar list, for example: List[float], List[int].

    Besides checking the type is ListType, we also check if the data type is
    a valid ONNX data type.
    """
def _is_tuple_construct(x: _C.Value) -> bool: ...
def is_complex_value(x: _C.Value) -> bool: ...
def _get_tensor_rank(x: _C.Value) -> int | None: ...
def _get_tensor_sizes(x: _C.Value, allow_nonstatic: bool = True): ...
def _get_tensor_dim_size(x: _C.Value, dim: int) -> int | None: ...
def _get_dim_for_cross(x: _C.Value, dim: int | None): ...
def _unimplemented(op: str, msg: str, value: _C.Value | None = None) -> None: ...
def _onnx_unsupported(op_name: str, value: _C.Value | None = None) -> NoReturn: ...
def _onnx_opset_unsupported(op_name: str, current_opset: int, supported_opset: int, value: _C.Value | None = None) -> NoReturn: ...
def _onnx_opset_unsupported_detailed(op_name: str, current_opset: int, supported_opset: int, reason: str, value: _C.Value | None = None) -> NoReturn: ...
def _block_list_in_opset(name: str): ...
def _try_get_scalar_type(*args) -> _type_utils.JitScalarType | None: ...
def _type_promote_from_values(*args) -> _type_utils.JitScalarType: ...
def _maybe_cast_to_type(g: jit_utils.GraphContext, value, jit_type: _type_utils.JitScalarType): ...
def _select_helper(g: jit_utils.GraphContext, self, dim, index, apply_reshape: bool = True): ...
def _slice_helper(g: jit_utils.GraphContext, input, axes, starts, ends, steps=None): ...
def _is_fp(value) -> bool: ...
def _is_bool(value) -> bool: ...
def _generate_wrapped_number(g: jit_utils.GraphContext, scalar):
    '''Creates a wrapped number based on https://github.com/pytorch/pytorch/issues/9515.

    A Tensor is a considered a "wrapped number" if it is
    auto-wrapped from a C++ or Python number type. Integer types are
    wrapped as 0-dim int64 tensors and floating-point types are
    wrapped as 0-dim double tensors.

    The input to this function is constant value. If the data type
    is a floating point type, it is converted to a 0-dim double
    tensor, else it is converted to a 0-dim tensor of its original type
    '''
def _sort_helper(g: jit_utils.GraphContext, input, dim, decending: bool = True, out=None): ...
def _topk_helper(g: jit_utils.GraphContext, input, k, dim, largest: bool = True, sorted: bool = False, out=None): ...
def _lt_helper(g: jit_utils.GraphContext, input, other): ...
def _interpolate_warning(interpolate_mode) -> None: ...
def _unsqueeze_helper(g: jit_utils.GraphContext, input, axes_i): ...
def _squeeze_helper(g: jit_utils.GraphContext, input, axes_i): ...
def _reducesum_helper(g: jit_utils.GraphContext, input, axes_i=None, keepdims_i: int = 1, noop_with_empty_axes_i: int = 0): ...
def _interpolate_size_to_scales(g: jit_utils.GraphContext, input, output_size, dim): ...
def _interpolate_get_scales_if_available(g: jit_utils.GraphContext, scales): ...
def _get_interpolate_attributes(g: jit_utils.GraphContext, mode, args): ...
def _interpolate_get_scales(g: jit_utils.GraphContext, scale_factor, dim): ...
def _interpolate_get_scales_and_mode(g: jit_utils.GraphContext, input, size, scale_factor, mode, align_corners): ...
def _argmin_argmax_helper(g: jit_utils.GraphContext, input: torch._C.Value, dim: torch._C.Value, keepdim: bool, op_name: str): ...
def _interpolate_helper(name, dim, interpolate_mode): ...
def __interpolate_helper(g: jit_utils.GraphContext, input, size, scale_factor, mode, align_corners, recompute_scale_factor): ...
def _unbind_helper(g: jit_utils.GraphContext, self, dim, _outputs): ...
def _scatter_helper(g: jit_utils.GraphContext, self, dim, index, src): ...
def _repeat_interleave_split_helper(g: jit_utils.GraphContext, self, reps, dim): ...
def _repeat_interleave_single_value_repeat_helper(g: jit_utils.GraphContext, self, repeats, dim): ...
def _arange_cast_helper(g: jit_utils.GraphContext, end, start=None, step=None, dtype=None) -> tuple[_type_utils.JitScalarType, _C.Value | None, _C.Value | None, _C.Value | None]: ...
def _arange_helper(g: jit_utils.GraphContext, *args): ...
def _size_helper(g: jit_utils.GraphContext, self, dim): ...
def _index_fill_reshape_helper(g: jit_utils.GraphContext, self, dim, index): ...
def _reshape_helper(g: jit_utils.GraphContext, input, shape, allowzero: int = 0): ...
def _batchnorm_helper(g: jit_utils.GraphContext, input, weight, bias, running_mean, running_var): ...
def _avgpool_helper(tuple_fn: Callable[[Any], Sequence[int]], padding: int | Sequence[int], kernel_size, stride, divisor_override, name) -> tuple[int, ...]: ...
def check_training_mode(op_train_mode: int, op_name: str) -> None:
    """Warns the user if the model's training mode and the export mode do not agree."""
def _flatten_helper(g: jit_utils.GraphContext, input, start_dim, end_dim, dim): ...
def _is_split_static(split_size_or_sizes, _outputs): ...
def _optional_input_placeholder_tensor(g): ...
def _handle_reduce_dim_none(g: jit_utils.GraphContext, self, op_name): ...
def dequantize_helper(g: jit_utils.GraphContext, qtensor: _C.Value, qdtype: _C_onnx.TensorProtoDataType | None = None) -> tuple[_C.Value, _C.Value, _C.Value, _C.Value | None]:
    """Appends to graph `g` ONNX nodes that dequantizes `qtensor` into `tensor`.

    Args:
        g: Graph, the ONNX IR graph that is under construction.
        qtensor: torch._C.Value, either a tuple of (quantized_tensor, scale, zero_point)
            for per tensor quantization, or
            (quantized_tensor, scale, zero_point, axis) for per channel quantization,
            representing the quantized tensor.
        qdtype: torch.onnx.TensorProtoDataType default None, if not None, represents the
            data type of quantized tensor. It must be either
            torch.onnx.TensorProtoDataType.UINT8 or torch.onnx.TensorProtoDataType.INT8.
    """
def quantize_helper(g: jit_utils.GraphContext, tensor: _C.Value, scale: _C.Value, zero_point: _C.Value, axis: _C.Value | None = None) -> _C.Value:
    """Appends to graph `g` ONNX nodes that quantizes `tensor` based on `scale`, `zero_point` and `axis`.

    Args:
        g: Graph, the ONNX IR graph that is under construction.
        tensor: torch._C.Value, representing the tensor to be quantized.
        scale: torch._C.Value, quantized scale.
        zero_point: torch._C.Value, quantized zero point.
        axis: Optional[torch._C.Value] default None, if None, represents per tensor quantization.
            Otherwise, represents per channel quantization, along given axis.

    Returns:
        A TupleConstruct storing information of the quantized tensor.
    """
def requantize_bias_helper(g: jit_utils.GraphContext, bias, input_scale, weight_scale, axis=None):
    """In PyTorch, bias is float and is quantized to int32 implicitly inside the quantized ATen op kernel.
    In ONNX we need to make the quantization explicit because operators expect all of their inputs to be quantized.
    Since int32 is not a supported output type by ONNX operator `QuantizeLinear`, quantization is exported using
    regular operators.
    """
def args_have_same_dtype(args): ...
def _op_with_optional_float_cast(g: jit_utils.GraphContext, op_name, *args, **kwargs):
    '''Some PyTorch operators (e.g., Clip/Min/ReLU/Pad) are super set of ONNX in terms of data types.
    This function maximizes the exportability of PyTorch-ONNX by allowing ONNX-unsupported PyTorch
    operator data type. For example, `Cast<int>(Clip<float>(Cast<float>(INPUT)))` can be used to mimic
    `Clip<int>(INPUT)` (opset version < 12).

    Args:
        g (torch._C.Graph): graph to write the ONNX representation into.
        op_name (str): operator name in ONNX.
        *args (tuple): operands to the operator.
        **kwargs (dict): attributes to the operator along with "opset_before" (optional, None by default)
            indicating the smallest opset version to trigger such casting behavior and "target_float_t"
            (optional, torch.onnx.JitScalarType.FLOAT by default) indicating the data type of internal operator.

    Returns:
        Optional[torch._C.Value, Tuple[torch._C.Value, ...]]: output(s) of the operator.
    '''
def _maybe_cast_reduce_op_input(g: jit_utils.GraphContext, self): ...
def _apply_params(*args, **kwargs):
    """Returns a decorator that calls the decorated (higher-order) function with the given parameters."""
def _reduce_op_symbolic_helper(onnx_op_name, allow_multi_dim_support: bool = True): ...
def _overload_by_arg_count(fn): ...
def _reduce_with_dtype_helper(onnx_op: str, name: str, allow_multi_dim_support: bool = True): ...
def _max_helper(g: jit_utils.GraphContext, self, dim_or_y=None, keepdim=None): ...
def _min_helper(g: jit_utils.GraphContext, self, dim_or_y=None, keepdim=None): ...
def _numel_helper(g: jit_utils.GraphContext, self): ...
def _var_mean_helper(g: jit_utils.GraphContext, input, dim, correction, keepdim): ...
def _embedding_bag_helper(g: jit_utils.GraphContext, embedding_matrix, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx): ...
def _linalg_vector_norm_helper(g: jit_utils.GraphContext, self: torch._C.Value, ord: float, dim: Sequence[int] | None, keepdim: bool, dtype: torch._C.Value): ...

cast_pytorch_to_onnx: Incomplete
scalar_name_to_pytorch: Incomplete
scalar_type_to_pytorch_type: Incomplete
pytorch_name_to_type: Incomplete
scalar_type_to_onnx: Incomplete
_quantized_ops: set[int]
