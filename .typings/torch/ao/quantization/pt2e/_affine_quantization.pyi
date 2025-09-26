import torch
from _typeshed import Incomplete
from torch.ao.quantization.observer import AffineQuantizedObserverBase as AffineQuantizedObserverBase, Granularity as Granularity, MappingType as MappingType, TorchAODType as TorchAODType, ZeroPointDomain as ZeroPointDomain, get_block_size as get_block_size
from typing import Any

ABC: Any
logger: Incomplete
FP8_TYPES: Incomplete
_SUB_BYTE_UINT_BOUNDS: Incomplete
_DTYPE_TO_QVALUE_BOUNDS: dict[torch.dtype | TorchAODType, tuple[int, int]]

def _is_float8_type(dtype: torch.dtype) -> bool: ...
def _get_and_check_qmin_qmax(dtype, quant_min, quant_max):
    """Get quant_min and quant_max args based on dtype and also
    verify that they are within the range of possible quant_min/quant_max
    for dtype
    """
def _get_reduction_params(block_size, input_size):
    """Given block_size and input size find the parameters for reduction:

    Output:
        shape_for_reduction: the shape we use to `view` input to prepare it for reduction
        reduction_dims: the dims we'll do reduction over

    Example::
        Input:
          block_size: (3, 3, 2, 10)
          input_size: (3, 3, 10, 10)

        Output:
          shape_for_reduction: (3, 3, 5, 2, 10)
          reduction_dim: [0, 1, 3, 4]
    """
def _register_custom_op(lib):
    '''This decorator is used to preserve some high level operators for torch.export.export
    while still allow them to be decomposed for inductor path

    requirement: make sure `fn.__name__[1:]` is the operator name you want to register

    NOTE: This should be applied at the top, after all other decorators have been applied
    NOTE: We haven\'t tested the case when `fn` accepts tensor subclass instance as input,
    e.g. uint4 tensor subclass instance, and we\'ll probably need to figure out what would make
    sense for downstream system (like executorch) to accept as well

    Example:
        lib = torch.library.Library("my_namespace\', "FRAGMENT")

        register_custom_op = _register_custom_op(lib)

        @register_custom_op
        def _the_op_that_needs_to_be_preserved(...)
            ...

        # after this, `_the_op_that_needs_to_be_preserved` will be preserved as
        # torch.ops.my_namespace.the_op_that_needs_to_be_preserved operator after
        # torch.export.export / torch._export.export_for_training

    '''

quant_lib: Incomplete
register_custom_op: Incomplete

def choose_qparams_affine_with_min_max(min_val: torch.Tensor, max_val: torch.Tensor, mapping_type: MappingType, block_size: tuple[int, ...], target_dtype: torch.dtype, quant_min: int | None = None, quant_max: int | None = None, eps: float | None = None, scale_dtype: torch.dtype | None = None, zero_point_dtype: torch.dtype | None = None, preserve_zero: bool = True, zero_point_domain: ZeroPointDomain | None = ...) -> tuple[torch.Tensor, torch.Tensor]:
    """A variant of :func:`~torchao.quantization.quant_primitives.choose_qparams_affine`
    operator that pass in min_val and max_val directly instead of deriving these from a single input.
    This is used for observers in static quantization where min_val and max_val may be obtained through
    tracking all the data in calibration data set.

    Args:
      Mostly same as :func:`~torchao.quantization.quant_primitives.choose_qparams_affine`. with one
      difference: instead of passing in `input` Tensor and use that to calculate min_val/max_val
      and then scale/zero_point, we pass in min_val/max_val directly
    """
@register_custom_op
def _choose_qparams_affine(input: torch.Tensor | None, mapping_type: str, block_size: list[int], target_dtype: torch.dtype, quant_min: int | float | bool | None = None, quant_max: int | float | bool | None = None, eps: float | None = None, scale_dtype: torch.dtype | None = None, zero_point_dtype: torch.dtype | None = None, preserve_zero: bool = True, zero_point_domain: str | None = 'INT', min_val: torch.Tensor | None = None, max_val: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """op definition that has compatible signatures with custom op library

    The op does the following:
    1. figure out the dimension for reduction based on block_size
    2. find min_val/max_val based on the dimension for reduction
    3. calculate quantization parameters based on min_val/max_val based on args like `preserve_zero`
       and `zero_point_domain`
    """
def quantize_affine(input: torch.Tensor, block_size: tuple[int, ...], scale: torch.Tensor, zero_point: torch.Tensor | None, output_dtype: torch.dtype, quant_min: int | float | None = None, quant_max: int | float | None = None, zero_point_domain: ZeroPointDomain | None = ...) -> torch.Tensor:
    """
    Args:
      input (torch.Tensor): original float32, float16 or bfloat16 Tensor
      block_size: (Tuple[int, ...]): granularity of quantization,
           this means the size of the tensor elements that's sharing the same qparam
           e.g. when size is the same as the input tensor dimension, we are using per tensor quantization
      scale (float): quantization parameter for affine quantization
      zero_point (int): quantization parameter for affine quantization
      output_dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor
      quant_min (Optional[int]): minimum quantized value for output Tensor, if not specified, it will be derived from dtype
      quant_max (Optional[int]): maximum quantized value for output Tensor, if not specified, it will be derived from dtype
      zero_point_domain (ZeroPointDomain): the domain that zero_point is in, should be either integer or float
        if zero_point is in integer domain, zero point is added to the quantized integer value during
        quantization
        if zero_point is in floating point domain, zero point is subtracted from the floating point (unquantized)
        value during quantization
        default is ZeroPointDomain.INT

    Note:
      How can block_size represent different granularities?
      let's say we have a Tensor of size: (3, 3, 10, 10), here is the table showing how block_size represents different
      granularities:

       granularity type       |     block_size
         per_tensor           |    (3, 3, 10, 10)
         per_axis (axis=0)    |    (1, 3, 10, 10)
         per_axis (axis=1)    |    (3, 1, 10, 10)
     per_group (groupsize=2)  |    (3, 3, 10, 2)
     per_group (groupsize=2) for axis = 3 | (3, 3, 2, 10)


    Output:
      quantized tensor with requested dtype
    """
@register_custom_op
def _quantize_affine(input: torch.Tensor, block_size: list[int], scale: torch.Tensor, zero_point: torch.Tensor | None, output_dtype: torch.dtype, quant_min: int | float | bool | None = None, quant_max: int | float | bool | None = None, zero_point_domain: str | None = ...) -> torch.Tensor:
    """op definition that has compatible signatures with custom op library

    Note:
        zero_point_domain is optional specifies how we quantize the floating point to quantized data:
        INT: quantized_val = (float_val / scale) (integer) + zero_point (integer)
        FLOAT: quantized_val = (float_val - (zero_point (float) - scale * mid_point)) / scale
        None: quantized_val = (float_val / scale) | this is primarily used for floatx quantization
            Where we do not want to round values to nearest integer and instead scale and cast.
    """
def _quantize_affine_no_dtype_cast(input: torch.Tensor, block_size: list[int], scale: torch.Tensor, zero_point: torch.Tensor | None, quant_min: int | float, quant_max: int | float, zero_point_domain: str | None = ...) -> torch.Tensor:
    """
    The op does the following:
    1. figure out the dimension for reduction based on block_size, also reshape the input to align with
       the shape after reduction
    2. quantize the input based on the quantization parameters scale and zero_point and args like zero_point_domain
    3. reshape the quantized result to origianl shape
    """
def dequantize_affine(input: torch.Tensor, block_size: tuple[int, ...], scale: torch.Tensor, zero_point: torch.Tensor | None, input_dtype: torch.dtype, quant_min: int | float | None = None, quant_max: int | float | None = None, zero_point_domain: ZeroPointDomain = ..., *, output_dtype: torch.dtype = ...) -> torch.Tensor:
    """
    Args:
      input (torch.Tensor): quantized tensor, should match the dtype `dtype` argument
      block_size: (List[int]): granularity of quantization,
        this means the size of the tensor elements that's sharing the same qparam
        e.g. when size is the same as the input tensor dimension, we are using per tensor quantization
      scale (Tensor): quantization parameter for affine quantization
      zero_point (Tensor): quantization parameter for affine quantization
      input_dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor
      quant_min (Optional[int]): minimum quantized value for input Tensor
      quant_max (Optional[int]): maximum quantized value for input Tensor
      output_dtype (torch.dtype): dtype for output Tensor, default is fp32
      zero_point_domain (ZeroPointDomain): the domain that zero_point is in, should be either integer or float
        if zero_point is in integer domain, zero point is added to the quantized integer value during
        quantization
        if zero_point is in floating point domain, zero point is subtracted from the floating point (unquantized)
        value during quantization
        default is ZeroPointDomain.INT

    Output:
      dequantized Tensor, with requested dtype or fp32
    """
@register_custom_op
def _dequantize_affine(input: torch.Tensor, block_size: list[int], scale: torch.Tensor, zero_point: torch.Tensor | None, input_dtype: torch.dtype, quant_min: int | float | bool | None = None, quant_max: int | float | bool | None = None, zero_point_domain: str | None = ..., output_dtype: torch.dtype = ...) -> torch.Tensor:
    """op definition that has compatible signatures with custom op library"""
def _dequantize_affine_no_dtype_check(input: torch.Tensor, block_size: list[int], scale: torch.Tensor, zero_point: torch.Tensor | None, quant_min: int | float, quant_max: int | float, zero_point_domain: str | None = ..., output_dtype: torch.dtype = ...) -> torch.Tensor:
    """This function converts AQT tensors to their high precision floating point representation

    The op does the following:
    1. figure out the dimension for reduction based on block_size, also reshape the input to align with
       the shape after reduction
    2. dequantize the input based on the quantization parameters scale and zero_point and args like zero_point_domain
    3. reshape the quantized result to origianl shape and change dtype to the output_dtype
    """

class AffineQuantizedMinMaxObserver(AffineQuantizedObserverBase):
    original_dtype: Incomplete
    block_size: Incomplete
    min_val: Incomplete
    max_val: Incomplete
    def forward(self, input: torch.Tensor): ...
    def calculate_qparams(self) -> tuple[torch.Tensor, torch.Tensor]: ...

class AffineQuantizedMovingAverageMinMaxObserver(AffineQuantizedObserverBase):
    is_dynamic: Incomplete
    averaging_constant: Incomplete
    def __init__(self, mapping_type: MappingType, target_dtype: torch.dtype, granularity: Granularity, averaging_constant: float = 0.01, quant_min: int | None = None, quant_max: int | None = None, eps: float | None = None, is_dynamic: bool = False, scale_dtype: torch.dtype | None = None, zero_point_dtype: torch.dtype | None = None, preserve_zero: bool = True, zero_point_domain: ZeroPointDomain | None = ..., **kwargs) -> None: ...
    original_dtype: Incomplete
    block_size: Incomplete
    min_val: Incomplete
    max_val: Incomplete
    def forward(self, input: torch.Tensor): ...
    def calculate_qparams(self) -> tuple[torch.Tensor, torch.Tensor]: ...

class AffineQuantizedPlaceholderObserver(AffineQuantizedObserverBase):
    is_dynamic: Incomplete
    def __init__(self, mapping_type: MappingType, target_dtype: torch.dtype, granularity: Granularity, quant_min: int | None = None, quant_max: int | None = None, eps: float | None = None, is_dynamic: bool = False, scale_dtype: torch.dtype | None = None, zero_point_dtype: torch.dtype | None = None, preserve_zero: bool = True, zero_point_domain: ZeroPointDomain | None = ..., **kwargs) -> None: ...
    block_size: Incomplete
    original_dtype: Incomplete
    def forward(self, input): ...
    def calculate_qparams(self) -> None: ...
