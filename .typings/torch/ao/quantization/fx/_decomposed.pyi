import torch
from _typeshed import Incomplete
from torch._refs import _unsqueeze_multiple as _unsqueeze_multiple
from torch.ao.quantization.utils import determine_qparams as determine_qparams, validate_qmin_qmax as validate_qmin_qmax
from torch.library import Library as Library, impl as impl

quantized_decomposed_lib: Incomplete
_INTEGER_DTYPES: Incomplete
_FLOAT_DTYPES: Incomplete
_DTYPE_TO_QVALUE_BOUNDS: Incomplete

def _quant_min_max_bounds_check(quant_min, quant_max, dtype) -> None: ...
def quantize_per_tensor(input: torch.Tensor, scale: float, zero_point: int, quant_min: int, quant_max: int, dtype: torch.dtype) -> torch.Tensor:
    """Affine quantization for the Tensor using the same quantization parameters to map
    from floating point to quantized values

    Args:
       input (torch.Tensor): original float32 or bfloat16 Tensor
       scale (float): quantization parameter for affine quantization
       zero_point (int): quantization parameter for affine quantization
       quant_min (int): minimum quantized value for output Tensor
       quant_max (int): maximum quantized value for output Tensor
       dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor

    Returns:
       Tensor with requested dtype (e.g. torch.uint8), note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """
def quantize_per_tensor_meta(input: torch.Tensor, scale: float, zero_point: int, quant_min: int, quant_max: int, dtype: torch.dtype) -> torch.Tensor: ...
def quantize_per_tensor_tensor(input: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, quant_min: int, quant_max: int, dtype: torch.dtype) -> torch.Tensor:
    """Affine quantization for the Tensor using the same quantization parameters to map
    from floating point to quantized values
    Same as `quantize_per_tensor` but scale and zero_point are Scalar Tensor instead of
    scalar values
    """
def quantize_per_tensor_tensor_meta(input: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, quant_min: int, quant_max: int, dtype: torch.dtype) -> torch.Tensor: ...
def quantize_per_tensor_tensor2(input: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, quant_min: torch.Tensor, quant_max: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Affine quantization for the Tensor using the same quantization parameters to map
    from floating point to quantized values
    Same as `quantize_per_tensor` but scale and zero_point are Scalar Tensor instead of
    scalar values
    """
def quantize_per_tensor_tensor2_meta(input: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, quant_min: torch.Tensor, quant_max: torch.Tensor, dtype: torch.dtype) -> torch.Tensor: ...
def dequantize_per_tensor(input: torch.Tensor, scale: float, zero_point: int, quant_min: int, quant_max: int, dtype: torch.dtype, *, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """Affine dequantization for the Tensor using the same quantization parameters to map
    from quantized values to floating point values

    Args:
       input (torch.Tensor): Tensor with dtype matching `dtype` argument,
       e.g. (`torch.uint8`), it is a per tensor quantized Tensor if combined with
       quantization parameters in the argument of this function (scale/zero_point)

       scale (float): quantization parameter for affine quantization

       zero_point (int): quantization parameter for affine quantization

       quant_min (int): minimum quantized value for input Tensor (not used in computation,
       reserved for pattern matching)

       quant_max (int): maximum quantized value for input Tensor (not used in computation,
       reserved for pattern matching)

       dtype (torch.dtype): dtype for input Tensor (not used in computation,
       reserved for pattern matching)

       out_dtype (torch.dtype?): optional dtype for output Tensor

    Returns:
       dequantized float32 Tensor
    """
def dequantize_per_tensor_meta(input: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, quant_min: int, quant_max: int, dtype: torch.dtype, *, out_dtype: torch.dtype | None = None) -> torch.Tensor: ...
def dequantize_per_tensor_tensor(input: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, quant_min: int, quant_max: int, dtype: torch.dtype, *, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """Affine dequantization for the Tensor using the same quantization parameters to map
    from quantized values to floating point values
    Same as `dequantize_per_tensor` but scale and zero_point are Scalar Tensor instead of
    scalar values
    """
def dequantize_per_tensor_tensor_meta(input: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, quant_min: int, quant_max: int, dtype: torch.dtype, *, out_dtype: torch.dtype | None = None) -> torch.Tensor: ...
def dequantize_per_tensor_tensor2(input: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, quant_min: torch.Tensor, quant_max: torch.Tensor, dtype: torch.dtype, *, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """Affine dequantization for the Tensor using the same quantization parameters to map
    from quantized values to floating point values
    Same as `dequantize_per_tensor` but scale and zero_point are Scalar Tensor instead of
    scalar values
    """
def dequantize_per_tensor_tensor2_meta(input, scale, zero_point, quant_min, quant_max, dtype, *, out_dtype: torch.dtype | None = None) -> torch.Tensor: ...
def choose_qparams_tensor(input: torch.Tensor, qmin: int, qmax: int, eps: float, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Given an input Tensor, derive the per tensor affine quantization parameter
    (scale and zero_point) for target quantized Tensor from the Tensor

    Args:
       input (torch.Tensor): floating point input Tensor
       quant_min (int): minimum quantized value for target quantized Tensor
       quant_max (int): maximum quantized value for target quantized Tensor
       dtype (torch.dtype): dtype for target quantized Tensor

    Returns:
       scale (float): quantization parameter for the target quantized Tensor
       zero_point (int): quantization parameter for the target quantized Tensor
    """
def choose_qparams_symmetric_tensor(input: torch.Tensor, qmin: int, qmax: int, eps: float, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Given an input Tensor, derive the per tensor affine quantization parameter
    (scale and zero_point) for target quantized Tensor from the Tensor

    Args:
       input (torch.Tensor): floating point input Tensor
       quant_min (int): minimum quantized value for target quantized Tensor
       quant_max (int): maximum quantized value for target quantized Tensor
       dtype (torch.dtype): dtype for target quantized Tensor

    Returns:
       scale (float): quantization parameter for the target quantized Tensor
       zero_point (int): quantization parameter for the target quantized Tensor
    """
def choose_qparams_tensor_meta(input: torch.Tensor, quant_min: int, quant_max: int, eps: float, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]: ...
def choose_qparams_symmetric_tensor_meta(input: torch.Tensor, quant_min: int, quant_max: int, eps: float, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]: ...
def _permute_to_axis_zero(x, axis): ...
def quantize_per_channel(input: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor, axis: int, quant_min: int, quant_max: int, dtype: torch.dtype) -> torch.Tensor:
    """Affine per channel quantization for the Tensor using the same quantization
    parameters for each channel/axis to map from floating point to quantized values

    Args:
       input (torch.Tensor): original float32 or bfloat16 Tensor
       scales (torch.Tensor): a list of scale quantization parameter for
       affine quantization, one per channel
       zero_point (torch.Tensor): a list of zero_point quantization parameter for
       affine quantization, one per channel
       quant_min (int): minimum quantized value for output Tensor
       quant_max (int): maximum quantized value for output Tensor
       dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor

    Returns:
       Tensor with requested dtype (e.g. torch.uint8), note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """
def quantize_per_channel_meta(input: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor, axis: int, quant_min: int, quant_max: int, dtype: torch.dtype) -> torch.Tensor: ...
def dequantize_per_channel(input: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor | None, axis: int, quant_min: int, quant_max: int, dtype: torch.dtype, *, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """Affine per channel dequantization for the Tensor using the same quantization
    parameters for each channel/axis to map from quantized values to floating point values

    Args:
       input (torch.Tensor): Tensor with dtype matching `dtype` argument,
       e.g. (`torch.uint8`), it is a per channel quantized Tensor if combined with
       quantization parameter in the argument of this function (scales/zero_points/axis)

       scales (torch.Tensor): a list of scale quantization parameter for
       affine quantization, one per channel

       zero_points (torch.Tensor): a list of zero_point quantization parameter for
       affine quantization, one per channel

       quant_min (int): minimum quantized value for output Tensor (not used in computation,
       reserved for pattern matching)

       quant_max (int): maximum quantized value for output Tensor (not used in computation,
       reserved for pattern matching)

       dtype (torch.dtype): requested dtype for output Tensor (not used in computation,
       reserved for pattern matching)

       out_dtype (torch.dtype?): optional dtype for output Tensor

    Returns:
       dequantized float32 Tensor
    """
def dequantize_per_channel_meta(input: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor | None, axis: int, quant_min: int, quant_max: int, dtype: torch.dtype, *, out_dtype: torch.dtype | None = None) -> torch.Tensor: ...
def choose_qparams_per_token(input: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose quantization parameters for per token quantization. This means for a N dimension Tensor
    (M1, M2, ...Mn, N), we calculate scales/zero_points for each N elements and quantize
    every N elements with the same quantization parameter. The dimension for scales/zero_points
    will be (M1 * M2 ... * Mn)

    Args:
       input (torch.Tensor): original float32/float16 Tensor
       dtype (torch.dtype): dtype (e.g. torch.uint8) for input Tensor

    Returns:
        scales and zero_points, both float32 Tensors
    """
def choose_qparams_per_token_meta(input: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]: ...
def _choose_qparams_per_token_asymmetric_impl(input: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose quantization parameters for per token quantization. This means for a N dimension Tensor
    (M1, M2, ...Mn, N), we calculate scales/zero_points for each N elements and quantize
    every N elements with the same quantization parameter. The dimension for scales/zero_points
    will be (M1 * M2 ... * Mn)

    Args:
       input (torch.Tensor): original float32/float16 Tensor
       dtype (torch.dtype): dtype (e.g. torch.uint8) for input Tensor

    Returns:
        scales and zero_points, both float32 Tensors
    """
def choose_qparams_per_token_asymmetric(input: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]: ...
def choose_qparams_per_token_asymmetric_meta(input: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]: ...
def _per_token_quant_qparam_dim_check(input, scales, zero_points) -> None: ...
def quantize_per_token(input: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor, quant_min: int, quant_max: int, dtype: torch.dtype):
    """Per token quantization for the Tensor using the quantization parameters to map
    from floating point to quantized values. This means for a N dimension Tensor
    (M1, M2, ...Mn, N), we calculate scales/zero_points for each N elements and quantize
    every N elements with the same quantization parameter. The dimension for scales/zero_points
    will be (M1 * M2 ... * Mn)

    Args:
       input (torch.Tensor): original float32 or bfloat16 Tensor
       scales (float32 torch.Tensor): quantization parameter for per token affine quantization
       zero_points (int32 torch.Tensor): quantization parameter for per token affine quantization
       quant_min (int): minimum quantized value for output Tensor
       quant_max (int): maximum quantized value for output Tensor
       dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor

    Returns:
       Tensor with requested dtype (e.g. torch.uint8), note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """
def quantize_per_token_meta(input: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor, quant_min: int, quant_max: int, dtype: torch.dtype): ...
def dequantize_per_token(input: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor, quant_min: int, quant_max: int, dtype: torch.dtype, output_dtype: torch.dtype = ...):
    """Per token dequantization for the Tensor using the quantization parameters to map
    from floating point to quantized values. This means for a N dimension Tensor
    (M1, M2, ...Mn, N), we calculate scales/zero_points for each N elements and quantize
    every N elements with the same quantization parameter. The dimension for scales/zero_points
    will be (M1 * M2 ... * Mn)

    Args:
       input (torch.Tensor): quantized Tensor (uint8, int8 etc.)
       scales (float64 torch.Tensor): quantization parameter for per token affine quantization
       zero_points (int64 torch.Tensor): quantization parameter for per token affine quantization
       quant_min (int): minimum quantized value for input Tensor
       quant_max (int): maximum quantized value for input Tensor
       dtype (torch.dtype): dtype (e.g. torch.uint8) for input Tensor
       output_dtype (torch.dtype): dtype (e.g. torch.float32) for output Tensor

    Returns:
       dequantized Tensor with dtype `output_dtype`
    """
def dequantize_per_token_meta(input: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor, quant_min: int, quant_max: int, dtype: torch.dtype, output_dtype: torch.dtype = ...): ...
def quantize_per_channel_group(input: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor, quant_min: int, quant_max: int, dtype: torch.dtype, group_size: int = 128): ...
def quantize_per_channel_group_meta(input: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor, quant_min: int, quant_max: int, dtype: torch.dtype, group_size: int = 128):
    """Groupwise quantization within each channel for an 2-d Tensor using the quantization parameters
    to map from floating point to quantized values. This means for each row of a 2-d Tensor
    (M, N), we calculate scales/zero_points for each `group_size` elements
    and quantize every `group_size` elements with the same quantization parameter.
    The dimension for scales/zero_points will be (M * ceil(N, group_size),)

    Args:
       input (torch.Tensor): original float32 or bfloat16 Tensor
       scales (float32 torch.Tensor): quantization parameter for per channel group affine quantization
       zero_points (int32 torch.Tensor): quantization parameter for per channel group affine quantization
       quant_min (int): minimum quantized value for output Tensor
       quant_max (int): maximum quantized value for output Tensor
       dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor

    Returns:
       Tensor with requested dtype (e.g. torch.uint8), note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """
def dequantize_per_channel_group(w_int8: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor | None, quant_min: int, quant_max: int, dtype: torch.dtype, group_size: int = 128, output_dtype: torch.dtype = ...):
    """Groupwise dequantization within each channel for an 2-d Tensor using the quantization parameters
    to map from floating point to quantized values. This means for each row of a 2-d Tensor
    (M, N), we calculate scales/zero_points for each `group_size` elements
    and quantize every `group_size` elements with the same quantization parameter.
    The dimension for scales/zero_points will be (M * ceil(N, group_size),)

    Args:
       input (torch.Tensor): quantized Tensor (uint8/int8 etc.)
       scales (float32 torch.Tensor): quantization parameter for per channel group affine quantization
       zero_points (int32 torch.Tensor): quantization parameter for per channel group affine quantization
       quant_min (int): minimum quantized value for input Tensor
       quant_max (int): maximum quantized value for input Tensor
       dtype (torch.dtype): dtype (e.g. torch.uint8) for input Tensor
       output_dtype (torch.dtype): dtype (e.g. torch.float32) for output Tensor

    Returns:
       dequantized Tensor with dtype `output_dtype`
    """

class FakeQuantPerChannel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scales, zero_points, axis, quant_min, quant_max): ...
    @staticmethod
    def backward(ctx, gy): ...

def fake_quant_per_channel(input: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor, axis: int, quant_min: int, quant_max: int) -> torch.Tensor: ...
def fake_quant_per_channel_meta(input: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor, axis: int, quant_min: int, quant_max: int) -> torch.Tensor: ...
def convert_element_type(input: torch.Tensor, dtype: torch.dtype) -> torch.Tensor: ...
def convert_element_type_meta(input: torch.Tensor, dtype: torch.dtype) -> torch.Tensor: ...
