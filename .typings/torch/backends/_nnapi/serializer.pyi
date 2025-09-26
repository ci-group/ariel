import enum
from _typeshed import Incomplete
from typing import NamedTuple

LOG: Incomplete

class NNAPI_OperandCode:
    FLOAT32: int
    INT32: int
    UINT32: int
    TENSOR_FLOAT32: int
    TENSOR_INT32: int
    TENSOR_QUANT8_ASYMM: int
    BOOL: int
    TENSOR_QUANT16_SYMM: int
    TENSOR_FLOAT16: int
    TENSOR_BOOL8: int
    FLOAT16: int
    TENSOR_QUANT8_SYMM_PER_CHANNEL: int
    TENSOR_QUANT16_ASYMM: int

class NNAPI_OperationCode:
    ADD: int
    AVERAGE_POOL_2D: int
    CONCATENATION: int
    CONV_2D: int
    DEPTHWISE_CONV_2D: int
    DEPTH_TO_SPACE: int
    DEQUANTIZE: int
    EMBEDDING_LOOKUP: int
    FLOOR: int
    FULLY_CONNECTED: int
    HASHTABLE_LOOKUP: int
    L2_NORMALIZATION: int
    L2_POOL_2D: int
    LOCAL_RESPONSE_NORMALIZATION: int
    LOGISTIC: int
    LSH_PROJECTION: int
    LSTM: int
    MAX_POOL_2D: int
    MUL: int
    RELU: int
    RELU1: int
    RELU6: int
    RESHAPE: int
    RESIZE_BILINEAR: int
    RNN: int
    SOFTMAX: int
    SPACE_TO_DEPTH: int
    SVDF: int
    TANH: int
    BATCH_TO_SPACE_ND: int
    DIV: int
    MEAN: int
    PAD: int
    SPACE_TO_BATCH_ND: int
    SQUEEZE: int
    STRIDED_SLICE: int
    SUB: int
    TRANSPOSE: int
    ABS: int
    ARGMAX: int
    ARGMIN: int
    AXIS_ALIGNED_BBOX_TRANSFORM: int
    BIDIRECTIONAL_SEQUENCE_LSTM: int
    BIDIRECTIONAL_SEQUENCE_RNN: int
    BOX_WITH_NMS_LIMIT: int
    CAST: int
    CHANNEL_SHUFFLE: int
    DETECTION_POSTPROCESSING: int
    EQUAL: int
    EXP: int
    EXPAND_DIMS: int
    GATHER: int
    GENERATE_PROPOSALS: int
    GREATER: int
    GREATER_EQUAL: int
    GROUPED_CONV_2D: int
    HEATMAP_MAX_KEYPOINT: int
    INSTANCE_NORMALIZATION: int
    LESS: int
    LESS_EQUAL: int
    LOG: int
    LOGICAL_AND: int
    LOGICAL_NOT: int
    LOGICAL_OR: int
    LOG_SOFTMAX: int
    MAXIMUM: int
    MINIMUM: int
    NEG: int
    NOT_EQUAL: int
    PAD_V2: int
    POW: int
    PRELU: int
    QUANTIZE: int
    QUANTIZED_16BIT_LSTM: int
    RANDOM_MULTINOMIAL: int
    REDUCE_ALL: int
    REDUCE_ANY: int
    REDUCE_MAX: int
    REDUCE_MIN: int
    REDUCE_PROD: int
    REDUCE_SUM: int
    ROI_ALIGN: int
    ROI_POOLING: int
    RSQRT: int
    SELECT: int
    SIN: int
    SLICE: int
    SPLIT: int
    SQRT: int
    TILE: int
    TOPK_V2: int
    TRANSPOSE_CONV_2D: int
    UNIDIRECTIONAL_SEQUENCE_LSTM: int
    UNIDIRECTIONAL_SEQUENCE_RNN: int
    RESIZE_NEAREST_NEIGHBOR: int

class NNAPI_FuseCode:
    FUSED_NONE: int
    FUSED_RELU: int
    FUSED_RELU1: int
    FUSED_RELU6: int

class OperandValueSourceType:
    IMMEDIATE: int
    NUMBERED_BUFFER: int
    NUMBERED_MEMORY: int

class TorchScalarTypes(enum.Enum):
    QUINT8 = 13

def approx_equal(lhs, rhs, tolerance: float = 1e-06): ...
def tensor_size(op_type, dims): ...
def change_element(tup, index, value): ...

class ConvPoolArgs2d(NamedTuple):
    """Configuration arguments for a convolution."""
    kernel_h: int
    kernel_w: int
    stride_h: int
    stride_w: int
    pad_t: int
    pad_b: int
    pad_l: int
    pad_r: int
    dilation_h: int
    dilation_w: int
    group: int

class DimOrder(enum.Enum):
    PRESUMED_CONTIGUOUS = 0
    CHANNELS_LAST = 1
    SCALAR_OR_VECTOR = 2
    UNKNOWN_CONSTANT = 999

class Operand(NamedTuple):
    """Represenation of an NNAPI operand."""
    op_type: int
    shape: tuple[int, ...]
    dim_order: DimOrder
    scale: float
    zero_point: int
    def use_nchw(self): ...

def broadcast_shapes(shape1, shape2): ...
def get_conv_pool_shape(image_shape, args, out_ch, transpose): ...
def fix_shape(shape, dim_order): ...
def reverse_map_dim(dim_order, d): ...
def flex_name(op_id, dim): ...

class _NnapiSerializer:
    operands: Incomplete
    values: Incomplete
    operations: Incomplete
    value_data: Incomplete
    operation_args: Incomplete
    inputs: Incomplete
    outputs: Incomplete
    flexible_shape_computation_lines: Incomplete
    modules: Incomplete
    constants: Incomplete
    tensor_sequences: Incomplete
    jitval_operand_map: Incomplete
    cached_immediates: Incomplete
    used_weights: Incomplete
    weight_offset: int
    use_int16_for_qint16: Incomplete
    def __init__(self, config, use_int16_for_qint16: bool = False) -> None: ...
    def get_next_operand_id(self): ...
    def add_tensor_operand(self, jitval, oper): ...
    def add_anonymous_tensor_operand(self, oper): ...
    def torch_tensor_to_operand(self, tensor, dim_order): ...
    def add_tensor_operand_for_input(self, arg_idx, jitval, tensor): ...
    def add_tensor_operand_for_weight(self, tensor, dim_order=...): ...
    def add_immediate_operand(self, code, value, dims): ...
    def add_immediate_int_scalar(self, value): ...
    def add_immediate_float_scalar(self, value): ...
    def add_immediate_bool_scalar(self, value): ...
    def add_immediate_int_vector(self, value): ...
    def has_operand_for_jitval(self, jitval): ...
    def get_tensor_operand_by_jitval(self, jitval): ...
    def get_tensor_operand_by_jitval_fixed_size(self, jitval): ...
    def get_tensor_operand_or_constant(self, jitval, dim_order=...): ...
    def get_tensor_operand_for_weight(self, jitval): ...
    def add_operation(self, opcode, inputs, outputs) -> None: ...
    def add_tensor_sequence(self, jitval, values) -> None: ...
    def add_constant_value(self, jitval, ctype, value) -> None: ...
    def get_constant_value(self, jitval, typekind=None): ...
    def operand_to_template_torchscript(self, op_id, oper, shape=None):
        """Return a TorchScript expression to build a template for a given operand."""
    def forward_operand_shape(self, out_op_id, out_dim, in_op_id, in_dim) -> None: ...
    def compute_operand_shape(self, op_id, dim, expr) -> None: ...
    def transpose_to_nhwc(self, in_id, oper): ...
    def transpose_for_broadcast(self, in0_id, in0_oper, in1_id, in1_oper): ...
    def get_size_arg(self, jitval): ...
    def get_conv_pool_args_2d_from_pack(self, kernel_size, packed_config): ...
    def get_conv_pool_args_2d_from_jit(self, kernel_size, stride, padding, dilation=None, group=None): ...
    def get_conv_pool_args_2d_common(self, kernel_size, strides, paddings, dilations, group_num): ...
    def serialize_model(self, model, inputs, return_shapes=None): ...
    def serialize_values(self): ...
    @staticmethod
    def serialize_ints(ints): ...
    ADDER_MAP: Incomplete
    def add_node(self, node) -> None: ...
    def _identity(self, node) -> None: ...
    def add_getattr(self, node) -> None: ...
    def add_constant_node(self, node) -> None: ...
    def add_list_construct(self, node) -> None: ...
    def add_tuple_construct(self, node) -> None: ...
    def add_unsqueeze(self, node) -> None: ...
    def add_to(self, node) -> None: ...
    def add_reshape(self, node) -> None: ...
    def add_flatten(self, node) -> None: ...
    def add_slice(self, node) -> None: ...
    def add_size(self, node) -> None: ...
    def add_cat(self, node) -> None: ...
    def add_mean(self, node) -> None: ...
    def add_quantize(self, node) -> None: ...
    def add_dequantize(self, node) -> None: ...
    def add_pointwise_simple_unary_op(self, node, opcode) -> None: ...
    def _do_add_binary(self, node, opcode, fuse_code, *, qparams=None) -> None:
        """Helper for pointwise binary broadcast ops with superfluous extra args."""
    def add_pointwise_simple_binary_broadcast_op(self, node, opcode, fuse_code) -> None: ...
    def add_add_sub_op(self, node, opcode, fuse_code) -> None: ...
    def add_qadd(self, node, opcode, fuse_code) -> None: ...
    def add_softmax(self, node) -> None: ...
    def add_hardtanh(self, node) -> None: ...
    def add_prelu_op(self, node) -> None: ...
    def add_pool2d_node(self, node, opcode) -> None: ...
    def add_avg_pool2d(self, node) -> None: ...
    def add_adaptive_avg_pool2d(self, node) -> None: ...
    def add_upsample_nearest2d(self, node) -> None: ...
    def add_addmm(self, node) -> None: ...
    def add_linear(self, node) -> None: ...
    def add_addmm_or_linear(self, node, transpose_weight, jit_input, jit_weight, jit_bias) -> None: ...
    def add_qlinear(self, node) -> None: ...
    def get_optional_bias(self, jit_bias, weight_tensor, transpose: bool = False): ...
    def add_conv2d(self, node): ...
    def add_conv_underscore(self, node): ...
    def add_log_softmax(self, node) -> None: ...
    def add_qconv2d(self, node, fuse_code, transpose: bool = False): ...
    def add_conv2d_common(self, jit_out, out_scale, out_zero_point, jit_image, weight_tensor, bias_id, args, transpose, fuse_code) -> None: ...
    def _handle_conv_pool_flexible_input(self, out_id, jit_image, args, transpose) -> None: ...

def serialize_model(module, inputs, *, config=None, return_shapes=None, use_int16_for_qint16: bool = False):
    """Convert to NNAPI and serialize torchscript module.

    Parameters:
        module: Torchscript module to convert
        inputs: Tensors used to specify input details for NNAPI
        config (optional): Optional config to attach to module
        return_shapes (optional): Specify shape of outputs if
            your module uses runtime flexible shapes to set output
            buffer size for NNAPI
        use_int16_for_qint16 (optional): Use Pytorch int16 to represent NNAPI qint16 values
    """
