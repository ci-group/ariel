import sympy
from .ir import ExternKernelAlloc as ExternKernelAlloc, FixedLayout as FixedLayout, FlexibleLayout as FlexibleLayout, Layout as Layout, MultiOutput as MultiOutput, MultiOutputLayout as MultiOutputLayout, MutationOutput as MutationOutput, NoneLayout as NoneLayout, TensorBox as TensorBox, get_device_type as get_device_type, ir_node_to_tensor as ir_node_to_tensor, is_contiguous_storage_and_layout as is_contiguous_storage_and_layout, may_convert_to_optional as may_convert_to_optional
from .utils import SUPPORTED_MKLDNN_DEVICES as SUPPORTED_MKLDNN_DEVICES, convert_shape_to_inductor as convert_shape_to_inductor, pad_listlike as pad_listlike
from .virtualized import V as V
from _typeshed import Incomplete
from collections.abc import Sequence
from torch._prims_common import make_channels_last_strides_for as make_channels_last_strides_for
from torch.utils._ordered_set import OrderedSet as OrderedSet
from typing import Any

def _prepare_convolution_fusion_create(cls, x: TensorBox, weight: TensorBox, bias: TensorBox, padding: Sequence[int], stride: Sequence[int], dilation: Sequence[int], groups: int, transposed: bool = False, output_padding: Sequence[int] | None = None, quantize_args: list['TensorBox'] | None = None, other: TensorBox | None = None):
    """
    This function is a helper function to prepare inputs, layout and constant args
    for convolution post-op fusion's create function, including deciding the output
    layout (channels first or channels last), realizing inputs and make them etc. The
    function only supports the CPU/XPU device since conv post-op fusion kernel is only
    supported on CPU/XPU right now.
    """
def _prepare_linear_fusion_create(cls, x: TensorBox, weight: TensorBox, bias: TensorBox, quantize_args: list['TensorBox'] | None = None, other: TensorBox | None = None, binary_sum: bool = False):
    """
    This function is a helper function to prepare inputs, layout and constant args
    for linear post-op fusion's create function. The function only supports the CPU device
    since linear post-op fusion kernel is only supported on CPU right now.
    """
def _create_output_node(packed): ...

class ConvolutionUnary(ExternKernelAlloc):
    device_type: Incomplete
    def __init__(self, layout, inputs, constant_args=()) -> None: ...
    def codegen(self, wrapper) -> None: ...
    @classmethod
    def create(cls, x: TensorBox, weight: TensorBox, bias: TensorBox, padding_: list[int], stride_: list[int], dilation_: list[int], groups: int, attr, scalars: list[Any] | None, algorithm): ...

class ConvolutionBinary(ExternKernelAlloc):
    device_type: Incomplete
    cpp_constant_args: Incomplete
    def __init__(self, layout, inputs, constant_args=(), cpp_constant_args=()) -> None: ...
    def codegen(self, wrapper) -> None: ...
    @classmethod
    def create(cls, x: TensorBox, other: TensorBox, weight: TensorBox, bias: TensorBox, padding_: list[int], stride_: list[int], dilation_: list[int], groups: int, binary_attr: str, binary_alpha: float | None, unary_attr: str | None, unary_scalars: list[Any] | None, unary_algorithm: str | None): ...

class ConvolutionBinaryInplace(ExternKernelAlloc):
    device_type: Incomplete
    mutation_outputs: Incomplete
    def __init__(self, kernel_layout, inputs, constant_args=()) -> None: ...
    def codegen(self, wrapper) -> None: ...
    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]: ...
    @classmethod
    def create(cls, x: TensorBox, other: TensorBox, weight: TensorBox, bias: TensorBox, padding_: list[int], stride_: list[int], dilation_: list[int], groups: int, binary_attr: str, binary_alpha: float | None, unary_attr: str | None, unary_scalars: list[Any] | None, unary_algorithm: str | None): ...

class ConvolutionTransposeUnary(ExternKernelAlloc):
    def __init__(self, layout, inputs, constant_args=()) -> None: ...
    def codegen(self, wrapper) -> None: ...
    @classmethod
    def create(cls, x: TensorBox, weight: TensorBox, bias: TensorBox, padding_: list[int], output_padding_: list[int], stride_: list[int], dilation_: list[int], groups_: int, attr, scalars: list[Any] | None, algorithm): ...

class QConvPointWisePT2E(ExternKernelAlloc):
    has_bias: Incomplete
    def __init__(self, layout, inputs, constant_args=()) -> None:
        """
        if bias is not None
            - inputs = [x, w, b, weight_scale, weight_zp]
            - const_args is: [stride, padding, dilation, groups, x_scale, x_zp, o_scale, o_zp,
              fp32_output, unary_attr, unary_scalars, unary_algorithm]
        else
            - inputs = [x, w, weight_scale, weight_zp]
            - const_args is: [bias, stride, padding, dilation, groups, x_scale, x_zp, o_scale, o_zp,
              fp32_output, unary_attr, unary_scalars, unary_algorithm]
        """
    def codegen(self, wrapper) -> None: ...
    @classmethod
    def create(cls, qx: TensorBox, x_scale: TensorBox, x_zero_point: TensorBox, qw: TensorBox, w_scale: TensorBox, w_zero_point: TensorBox, bias: TensorBox, stride: list[int], padding: list[int], dilation: list[int], groups: int, output_scale: float, output_zero_point: int, output_dtype, attr, scalars, algorithm): ...

class QConvPointWiseBinaryPT2E(ExternKernelAlloc):
    has_bias: Incomplete
    idx_for_inplace_sum: int
    def __init__(self, layout, inputs, constant_args=()) -> None:
        """
        Needs input/weight/output qparams
        if bias is not None
            - inputs = [x, x_scale, x_zp, w,  w_scale, w_zp, accum, b]
            - const_args = [stride, padding, dilation, groups, o_scale, o_zp,
            output_dtype, accum_scale, accum_zp, binary_attr, alpha, unary_attr, unary_scalars, unary_algorithm]
        else
            - inputs = [x, x_scale, x_zp, w,  w_scale, w_zp, accum]
            - const_args [b, stride, padding, dilation, groups, o_scale, o_zp,
             output_dtype, accum_scale, accum_zp, binary_attr, alpha, unary_attr, unary_scalars, unary_algorithm]
        """
    def codegen(self, wrapper) -> None: ...
    def get_mutation_names(self) -> Sequence[str]: ...
    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]: ...
    @classmethod
    def create(cls, qx: TensorBox, x_scale: TensorBox, x_zero_point: TensorBox, qw: TensorBox, w_scale, w_zero_point, qaccum: TensorBox, bias: TensorBox, stride: list[int], padding: list[int], dilation: list[int], groups: int, output_scale: TensorBox, output_zero_point: TensorBox, output_dtype, accum_scale, accum_zero_point, binary_attr, alpha, unary_attr, unary_scalars, unary_algorithm): ...

class MKLPackedLinear(ExternKernelAlloc):
    def __init__(self, layout, inputs, constant_args=()) -> None: ...
    def codegen(self, wrapper) -> None: ...
    @classmethod
    def create(cls, x, packed_w, orig_w, B, batch_size): ...

class LinearUnary(ExternKernelAlloc):
    def __init__(self, layout, inputs, constant_args=()) -> None: ...
    def codegen(self, wrapper) -> None: ...
    @classmethod
    def create(cls, x, w, B, attr, scalars, algorithm): ...
    def apply_constraint(self) -> None: ...

class LinearBinary(ExternKernelAlloc):
    kernel: str
    def __init__(self, layout, inputs, constant_args=()) -> None: ...
    def codegen(self, wrapper) -> None: ...
    @classmethod
    def create(cls, x, y, w, B, attr): ...
    def apply_constraint(self) -> None: ...

class QLinearPointwisePT2E(ExternKernelAlloc):
    has_bias: Incomplete
    def __init__(self, layout, inputs, constant_args=(), has_bias: bool = True) -> None:
        """
        if bias is not None
            - inputs = [x, w, b, weight_scale, weight_zp]
            - const_args is: [x_scale, x_zp, o_scale, o_zp,
              fp32_output, unary_attr, unary_scalars, unary_algorithm]
        else
            - inputs = [x, w, weight_scale, weight_zp]
            - const_args is: [bias, x_scale, x_zp, o_scale, o_zp,
              fp32_output, unary_attr, unary_scalars, unary_algorithm]
        """
    def codegen(self, wrapper) -> None: ...
    @classmethod
    def create(cls, qx: TensorBox, x_scale: TensorBox, x_zero_point: TensorBox, qw: TensorBox, w_scale: TensorBox, w_zero_point: TensorBox, bias: TensorBox, output_scale: float, output_zero_point: int, output_dtype, post_op_name, post_op_args, post_op_algorithm): ...

class QLinearPointwiseBinaryPT2E(ExternKernelAlloc):
    has_bias: Incomplete
    idx_for_inplace_sum: int
    def __init__(self, layout, inputs, constant_args=(), has_bias: bool = True) -> None:
        """
        if bias is not None
            - inputs = [x, w, x_scale, x_zp, weight_scale, weight_zp, x2, bias]
            - const_args is: [o_scale, o_zp,
              fp32_output, binary_attr, alpha, unary_attr, unary_scalars, unary_algorithm]
        else
            - inputs = [x, w, x_scale, x_zp, weight_scale, weight_zp, x2]
            - const_args is: [bias, o_scale, o_zp,
              fp32_output, binary_attr, alpha, unary_attr, unary_scalars, unary_algorithm]
        """
    def codegen(self, wrapper) -> None: ...
    def get_mutation_names(self) -> Sequence[str]: ...
    @classmethod
    def create(cls, qx: TensorBox, x_scale: TensorBox, x_zero_point: TensorBox, qw: TensorBox, w_scale: TensorBox, w_zero_point: TensorBox, other: TensorBox, bias: TensorBox, output_scale: float, output_zero_point: int, output_dtype, other_scale, other_zp, binary_post_op, binary_alpha, unary_post_op, unary_post_op_args, unary_post_op_algorithm): ...

class MkldnnRnnLayer(ExternKernelAlloc):
    def __init__(self, layout, inputs, constant_args=()) -> None: ...
    @classmethod
    def create(cls, x: TensorBox, w0: TensorBox, w1: TensorBox, w2: TensorBox, w3: TensorBox, hx: TensorBox, cx: TensorBox, reverse: bool, batch_sizes: list[int], mode: int, hidden_size: int, num_layers: int, has_biases: bool, bidirectional: bool, batch_first: bool, train: bool): ...
    def codegen(self, wrapper): ...

class WeightInt4PackMatmul(ExternKernelAlloc):
    def __init__(self, layout, inputs, constant_args=()) -> None:
        """
        inputs = [x, w, qGroupSize, qScalesAndZeros]
        constant_args = ()
        """
    def codegen(self, wrapper) -> None: ...
    @classmethod
    def create(cls, x: TensorBox, w: TensorBox, qGroupSize: TensorBox, qScalesAndZeros: TensorBox): ...
