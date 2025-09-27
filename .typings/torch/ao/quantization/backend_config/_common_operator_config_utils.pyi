from .backend_config import BackendPatternConfig as BackendPatternConfig, DTypeConfig as DTypeConfig, DTypeWithConstraints as DTypeWithConstraints, ObservationType as ObservationType
from _typeshed import Incomplete
from torch.ao.quantization.fuser_method_mappings import _sequential_wrapper2 as _sequential_wrapper2, fuse_conv_bn as fuse_conv_bn, fuse_conv_bn_relu as fuse_conv_bn_relu, fuse_convtranspose_bn as fuse_convtranspose_bn, fuse_linear_bn as fuse_linear_bn
from typing import Callable, NamedTuple

__all__: list[str]

class _ConvMetadata(NamedTuple):
    root: Incomplete
    transpose: Incomplete
    bn: Incomplete
    reference: Incomplete
    transpose_reference: Incomplete
    fused_conv_relu: Incomplete
    fused_conv_bn: Incomplete
    fused_conv_bn_relu: Incomplete
    qat: Incomplete
    relu_qat: Incomplete
    bn_qat: Incomplete
    bn_relu_qat: Incomplete
    func: Incomplete
    func_transpose: Incomplete

_Conv1dMetadata: Incomplete
_Conv2dMetadata: Incomplete
_Conv3dMetadata: Incomplete
_FIXED_QPARAM_OP_0TO1_CONSTRAINTS: Incomplete
_FIXED_QPARAM_OP_NEG1TO1_CONSTRAINTS: Incomplete
_FIXED_QPARAMS_OP_TO_CONSTRAINTS: dict[Callable | str, DTypeWithConstraints]

def _get_binary_op_configs(dtype_configs: list[DTypeConfig]) -> list[BackendPatternConfig]: ...
def _get_linear_configs(dtype_configs: list[DTypeConfig]) -> list[BackendPatternConfig]:
    """
    Return all configs related to linear modules and ops.
    """
def _get_conv_configs(dtype_configs):
    """
    Return all configs related to conv modules and ops.
    """
def _get_cat_config(dtype_configs: list[DTypeConfig]) -> BackendPatternConfig: ...
def _get_ln_configs(dtype_configs: list[DTypeConfig]) -> list[BackendPatternConfig]: ...
def _get_default_op_configs(dtype_configs: list[DTypeConfig]) -> list[BackendPatternConfig]: ...
def _add_fixed_qparams_to_dtype_configs(dtype_configs: list[DTypeConfig], constraints: DTypeWithConstraints) -> list[DTypeConfig]:
    """
    Return a copy of the list of DTypeConfigs where activations are subject to the specified
    constraints required for fixed qparams ops.

    If the data type doesn't match the one in the constraints, simply leave the corresponding
    DTypeConfig unchanged.

    If `scale_min_lower_bound` or `scale_max_upper_bound` is specified in the activations,
    throw an exception since these settings are incompatible with fixed qparams ops.
    """
def _get_fixed_qparams_op_configs(dtype_configs: list[DTypeConfig]) -> list[BackendPatternConfig]: ...
def _get_share_qparams_op_configs(dtype_configs):
    """Get the operator config for the operators that works for both float and quantized input
    if input is quantized, the output Tensor shares the same quantization parameter
    with input.
    Example operator: avgpool2d, reshape, transpose, maxpool2d
    Example observed operator:
    observer_0 - avgpool2d - observer_0 (same observer instance as input)
    """
def _get_bn_configs(dtype_configs: list[DTypeConfig]) -> list[BackendPatternConfig]:
    """Get configs related to batchnorm."""
def _get_rnn_op_configs(dtype_configs: list[DTypeConfig]) -> list[BackendPatternConfig]: ...
def _get_embedding_op_configs(dtype_configs: list[DTypeConfig]) -> list[BackendPatternConfig]: ...
def _get_tensor_info_op_configs(dtype_configs):
    """
    These ops work on tensors of different dtypes but return non-tensors
    containing information about the input tensor.
    """
