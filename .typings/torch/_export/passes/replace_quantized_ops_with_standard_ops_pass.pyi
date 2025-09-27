import torch
from _typeshed import Incomplete
from torch._ops import OpOverload as OpOverload
from torch.ao.quantization.fx._decomposed import dequantize_per_channel as dequantize_per_channel, dequantize_per_tensor as dequantize_per_tensor, quantize_per_tensor as quantize_per_tensor
from torch.ao.quantization.utils import calculate_qmin_qmax as calculate_qmin_qmax
from torch.fx.graph_module import _assign_attr as _assign_attr

log: Incomplete
_INPUT_Q_DTYPE: torch.dtype | torch.fx.Node | None
_SCALE: float | torch.fx.Node | None
_ZERO_POINT: float | torch.fx.Node | None

def int_to_valid_dtype(val: int) -> torch.dtype: ...
def fx_enum_to_dtype(gm: torch.fx.GraphModule, val: int) -> torch.fx.Node: ...
def insert_quantized_node(gm: torch.fx.GraphModule, val_node: torch.fx.Node, scale_node: float | torch.fx.Node, zero_point_node: float | torch.fx.Node, qmin_node: float | int | torch.fx.Node, qmax_node: float | int | torch.fx.Node, dtype_node: torch.dtype | torch.fx.Node, qscheme: torch.qscheme | None) -> torch.fx.Node: ...
def get_dequantized(val: torch.Tensor, scale: float | torch.Tensor, zero_point: float | torch.Tensor, qmin: float | int, qmax: float | int, dtype: torch.dtype, axis: int | None, qscheme: torch.qscheme | None) -> torch.Tensor: ...
def insert_dequantized_node(gm: torch.fx.GraphModule, val_node: torch.fx.Node, scale_node: float | torch.fx.Node, zero_point_node: float | torch.fx.Node, qmin_node: float | int | torch.fx.Node, qmax_node: float | int | torch.fx.Node, dtype_node: torch.dtype | torch.fx.Node, axis_node: int | torch.fx.Node | None, qscheme: torch.qscheme | None) -> torch.fx.Node: ...
def get_qmin_qmax(dtype: torch.dtype) -> tuple[int | float, int | float]: ...
def insert_qmin_qmax_node(gm: torch.fx.GraphModule, dtype_node: torch.dtype | torch.fx.Node) -> tuple[torch.fx.Node, torch.fx.Node]: ...
def get_script_object(gm: torch.nn.Module, node: torch.fx.Node) -> torch._C.ScriptObject: ...
def insert_weight_and_bias_get_attr_node_from_get_attr_to_scriptobject(gm: torch.fx.GraphModule, param_node: torch.fx.Node) -> tuple[torch.fx.Node, torch.fx.Node | None]:
    """Directly inline tensor from a get_attr fx node."""
def insert_weight_and_bias_get_attr_node_from_get_attr_to_qtensor(gm: torch.fx.GraphModule, get_attr_to_weight_node: torch.fx.Node, get_attr_to_bias_node: torch.fx.Node | None) -> tuple[torch.fx.Node, torch.fx.Node | None]: ...
def insert_weight_and_bias_get_attr_node(gm: torch.fx.GraphModule, w_qtensor: torch.Tensor, b_qtensor: torch.Tensor | None, w_attr_name: str, b_attr_name: str) -> tuple[torch.fx.Node, torch.fx.Node | None]: ...
def get_tensor_from_qtensor(qtensor: torch.Tensor, dequant: bool = True) -> torch.Tensor: ...
def insert_fused_activation_node(gm: torch.fx.GraphModule, opname: str, fx_node: torch.fx.Node) -> torch.fx.Node: ...
def _conv1d_op_with_squeeze(inp: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, stride: list[int], padding: list[int], dilation: list[int], groups: int) -> torch.Tensor: ...
def _transform_conv_with_packedparam(gm: torch.fx.GraphModule, node: torch.fx.Node):
    """Conv specfic transformation function."""
def _transform_linear_with_packedparam(gm: torch.fx.GraphModule, node: torch.fx.Node):
    """Linear specfic transformation function."""
def _transform_op_where_last_two_arguments_are_scale_and_zero_point(gm: torch.fx.GraphModule, node: torch.fx.Node):
    """
    This transformation function can be used for function where the last two
    parameters are scale and zero point. Additionally, the function's parameters
    do not need any unpacking.
    """
def _transform_scalar_arithmetic(gm: torch.fx.GraphModule, node: torch.fx.Node):
    """Transform scalar overload for basic arithmetic."""
def _transform_prepacked_op(gm: torch.fx.GraphModule, node: torch.fx.Node):
    """
    Transformation for functions under prepacked namespace, where they share
    the same handling logic that [...]OpContext contains all parameters.
    """
def _transform_batch_norm(gm: torch.fx.GraphModule, node: torch.fx.Node): ...
def fx_transform_quantized_op_to_standard_op(gm: torch.fx.GraphModule, node: torch.fx.Node) -> torch.fx.Node: ...
def replace_quantized_ops_with_standard_ops(gm: torch.fx.GraphModule):
    """
    Replace legacy quantized ops (aten.quantize_per_tensor, quantized.conv) with
    PT2 ops (quantize_decomposed.quantize_per_tensor, aten.conv).

    Before:    x || -> aten.q        || -> quantized.conv2d     || -> quantized.linear    || -> aten.dq || -> y

    After:     x || -> qd.q -> qd.dq || -> aten.conv2d -> qd.q -> qd.dq || aten.linear -> qd.q -> qd.dq || -> y

    (qd == quantized_decomposed library, q = quantize, dq = dequantize)
                                          ^
                                          |
                getattr(w), getattr(b) from Conv2dParamPrepack

    During each iteration, the transformation spits out the transformed operator, its quantized output,
    and its dequantized value together. We did this because dequantization need to use the
    scale and zero point parameters from the quantization to recover the approximate original value. After each
    iteration, the new dequantization node will be used as the input to the next node (e.g., dq2 -> linear).

    For operators like conv2d and linear, their weights and bias are packed in a quantized format in the ScriptObject.
    During the transformation, we unpack those objects, get their dequantized tensor, populate those
    as attributes to the module, and use getattr to access them.

    One exception in the transformation is conv_prepack and linear_prepack. Those calls pack
    weight and bias constant tensors into ScriptObject, which are then used by subsequent conv2d or linear calls.
    During transformation, we directly skip transforming conv_prepack or linear_prepack. We check whether ScriptObject to the
    quantized::conv2d or linear is from conv_prepack or linear_prepack. If it is, we then inline those parameters
    to the operator by converting them to a getattr fx.node.

    For prepacked::conv2d_clamp_run and prepacked::linear_clamp_run, we directly convert them to aten.conv2d and aten.linear
    without the need of doing de/quantization.

    Three global variables defined are _INPUT_Q_DTYPE, _SCALE, _ZERO_POINT. _INPUT_Q_DTYPE determines the de/quantization
    data type, which is the same across the entire program, but it only shows up in the very first quantization
    call. _SCALE and _ZERO_POINT are used only when operators do not have those specified. E.g., mul.Scalar.
    """
