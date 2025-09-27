import abc
import torch
import torch.nn as nn
from _typeshed import Incomplete
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from torch.fx import Node
from typing import Any

__all__ = ['default_affine_fixed_qparams_observer', 'default_debug_observer', 'default_dynamic_quant_observer', 'default_fixed_qparams_range_0to1_observer', 'default_fixed_qparams_range_neg1to1_observer', 'default_float_qparams_observer', 'default_float_qparams_observer_4bit', 'default_histogram_observer', 'default_observer', 'default_per_channel_weight_observer', 'default_placeholder_observer', 'default_reuse_input_observer', 'default_symmetric_fixed_qparams_observer', 'default_weight_observer', 'get_observer_state_dict', 'load_observer_state_dict', 'per_channel_weight_observer_range_neg_127_to_127', 'weight_observer_range_neg_127_to_127', 'FixedQParamsObserver', 'HistogramObserver', 'MinMaxObserver', 'MovingAverageMinMaxObserver', 'MovingAveragePerChannelMinMaxObserver', 'NoopObserver', 'ObserverBase', 'PerChannelMinMaxObserver', 'PlaceholderObserver', 'RecordingObserver', 'ReuseInputObserver', 'UniformQuantizationObserverBase', 'AffineQuantizedObserverBase', 'Granularity', 'MappingType', 'PerAxis', 'PerBlock', 'PerGroup', 'PerRow', 'PerTensor', 'PerToken', 'TorchAODType', 'ZeroPointDomain', 'get_block_size']

class _PartialWrapper:
    p: Incomplete
    callable_args: Incomplete
    def __init__(self, p) -> None: ...
    def __call__(self, *args, **keywords): ...
    def __repr__(self) -> str: ...
    def with_args(self, **kwargs): ...
    def with_callable_args(self, **kwargs): ...

class ObserverBase(ABC, nn.Module, metaclass=abc.ABCMeta):
    """Base observer Module.
    Any observer implementation should derive from this class.

    Concrete observers should follow the same API. In forward, they will update
    the statistics of the observed Tensor. And they should provide a
    `calculate_qparams` function that computes the quantization parameters given
    the collected statistics.

    Args:
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        is_dynamic: indicator for whether the observer is a placeholder for dynamic quantization
        or static quantization
    """
    dtype: Incomplete
    is_dynamic: Incomplete
    def __init__(self, dtype, is_dynamic: bool = False) -> None: ...
    @abstractmethod
    def forward(self, x): ...
    @abstractmethod
    def calculate_qparams(self, **kwargs): ...
    with_args: Incomplete
    with_callable_args: Incomplete

class UniformQuantizationObserverBase(ObserverBase, metaclass=abc.ABCMeta):
    """Common base for all observers using uniform quantization to calculate
    scale and zero_point.

    Args:
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        qscheme: Quantization scheme to be used.
        reduce_range: Reduces the range of the quantized data type by 1 bit.
                      This is sometimes required to avoid instruction overflow.
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    .. warning::

        :attr:`dtype` can only take ``torch.qint8`` or ``torch.quint8``.
               or `torch.int8` or `torch.uint8`

    .. warning::

        :attr:`qscheme` can only take one of the following options:

        - ``torch.per_tensor_affine``
        - ``torch.per_tensor_symmetric``
        - ``torch.per_channel_affine``
        - ``torch.per_channel_symmetric``
    """
    _version: int
    eps: torch.Tensor
    qscheme: Incomplete
    reduce_range: Incomplete
    has_customized_qrange: Incomplete
    def __init__(self, dtype=..., qscheme=..., reduce_range: bool = False, quant_min=None, quant_max=None, factory_kwargs=None, eps=..., is_dynamic: bool = False, **kwargs) -> None: ...
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None: ...
    @torch.jit.export
    def _validate_qmin_qmax(self, quant_min: int, quant_max: int) -> None:
        """Validates that the user-specified quantization range is properly initialized
        and within the given bound supported by the observer dtype.

        To accommodate lower-bit quantization with respect to the existing torch.qint8 and
        torch.quint8 datatypes, the user can choose to use dynamic quantization range by passing
        in a tuple of initial qmin and qmax values. One use case is these customized qmin and qmax
        values are used to calculate static estimates of the scale and zero point for aggressive lower-bit
        fake quantization. These estimates are compared against parameters learned through backpropagation.
        The related literatures for scale and zero point via backpropagation are as follows:

        Learned Step Size Quantization: https://openreview.net/pdf?id=rkgO66VKDS
        Trained Quantization Thresholds: https://arxiv.org/pdf/1903.08066.pdf
        """
    @torch.jit.export
    def _calculate_qparams(self, min_val: torch.Tensor, max_val: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculates the quantization parameters, given min and max
        value tensors. Works for both per tensor and per channel cases

        Args:
            min_val: Minimum values per channel
            max_val: Maximum values per channel

        Returns:
            scales: Scales tensor of shape (#channels,)
            zero_points: Zero points tensor of shape (#channels,)
        """
    @torch.jit.export
    def reset_min_max_vals(self) -> None: ...
_ObserverBase = UniformQuantizationObserverBase

class MinMaxObserver(UniformQuantizationObserverBase):
    """Observer module for computing the quantization parameters based on the
    running min and max values.

    This observer uses the tensor min/max statistics to compute the quantization
    parameters. The module records the running minimum and maximum of incoming
    tensors, and uses this statistic to compute the quantization parameters.

    Args:
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    Given running min/max as :math:`x_\\text{min}` and :math:`x_\\text{max}`,
    scale :math:`s` and zero point :math:`z` are computed as:

    The running minimum/maximum :math:`x_\\text{min/max}` is computed as:

    .. math::

        \\begin{array}{ll}
        x_\\text{min} &= \\begin{cases}
            \\min(X) & \\text{if~}x_\\text{min} = \\text{None} \\\\\n            \\min\\left(x_\\text{min}, \\min(X)\\right) & \\text{otherwise}
        \\end{cases}\\\\\n        x_\\text{max} &= \\begin{cases}
            \\max(X) & \\text{if~}x_\\text{max} = \\text{None} \\\\\n            \\max\\left(x_\\text{max}, \\max(X)\\right) & \\text{otherwise}
        \\end{cases}\\\\\n        \\end{array}

    where :math:`X` is the observed tensor.

    The scale :math:`s` and zero point :math:`z` are then computed as:

    .. math::

        \\begin{aligned}
            \\text{if Symmetric:}&\\\\\n            &s = 2 \\max(|x_\\text{min}|, x_\\text{max}) /
                \\left( Q_\\text{max} - Q_\\text{min} \\right) \\\\\n            &z = \\begin{cases}
                0 & \\text{if dtype is qint8} \\\\\n                128 & \\text{otherwise}
            \\end{cases}\\\\\n            \\text{Otherwise:}&\\\\\n                &s = \\left( x_\\text{max} - x_\\text{min}  \\right ) /
                    \\left( Q_\\text{max} - Q_\\text{min} \\right ) \\\\\n                &z = Q_\\text{min} - \\text{round}(x_\\text{min} / s)
        \\end{aligned}

    where :math:`Q_\\text{min}` and :math:`Q_\\text{max}` are the minimum and
    maximum of the quantized data type.

    .. warning:: :attr:`dtype` can only take ``torch.qint8`` or ``torch.quint8``.

    .. note:: If the running minimum equals to the running maximum, the scale
              and zero_point are set to 1.0 and 0.
    """
    min_val: torch.Tensor
    max_val: torch.Tensor
    def __init__(self, dtype=..., qscheme=..., reduce_range: bool = False, quant_min=None, quant_max=None, factory_kwargs=None, eps=..., is_dynamic: bool = False, **kwargs) -> None: ...
    def forward(self, x_orig):
        """Records the running minimum and maximum of ``x``."""
    @torch.jit.export
    def calculate_qparams(self):
        """Calculates the quantization parameters."""
    @torch.jit.export
    def extra_repr(self): ...
    @torch.jit.export
    def reset_min_max_vals(self) -> None:
        """Resets the min/max values."""

class MovingAverageMinMaxObserver(MinMaxObserver):
    """Observer module for computing the quantization parameters based on the
    moving average of the min and max values.

    This observer computes the quantization parameters based on the moving
    averages of minimums and maximums of the incoming tensors. The module
    records the average minimum and maximum of incoming tensors, and uses this
    statistic to compute the quantization parameters.

    Args:
        averaging_constant: Averaging constant for min/max.
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    The moving average min/max is computed as follows

    .. math::

        \\begin{array}{ll}
                x_\\text{min} = \\begin{cases}
                    \\min(X) & \\text{if~}x_\\text{min} = \\text{None} \\\\\n                    (1 - c) x_\\text{min} + c \\min(X) & \\text{otherwise}
                \\end{cases}\\\\\n                x_\\text{max} = \\begin{cases}
                    \\max(X) & \\text{if~}x_\\text{max} = \\text{None} \\\\\n                    (1 - c) x_\\text{max} + c \\max(X) & \\text{otherwise}
                \\end{cases}\\\\\n        \\end{array}

    where :math:`x_\\text{min/max}` is the running average min/max, :math:`X` is
    is the incoming tensor, and :math:`c` is the ``averaging_constant``.

    The scale and zero point are then computed as in
    :class:`~torch.ao.quantization.observer.MinMaxObserver`.

    .. note:: Only works with ``torch.per_tensor_affine`` quantization scheme.

    .. note:: If the running minimum equals to the running maximum, the scale
              and zero_point are set to 1.0 and 0.
    """
    averaging_constant: Incomplete
    def __init__(self, averaging_constant: float = 0.01, dtype=..., qscheme=..., reduce_range: bool = False, quant_min=None, quant_max=None, eps=..., is_dynamic: bool = False, **kwargs) -> None: ...
    def forward(self, x_orig): ...

class PerChannelMinMaxObserver(UniformQuantizationObserverBase):
    """Observer module for computing the quantization parameters based on the
    running per channel min and max values.

    This observer uses the tensor min/max statistics to compute the per channel
    quantization parameters. The module records the running minimum and maximum
    of incoming tensors, and uses this statistic to compute the quantization
    parameters.

    Args:
        ch_axis: Channel axis
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    The quantization parameters are computed the same way as in
    :class:`~torch.ao.quantization.observer.MinMaxObserver`, with the difference
    that the running min/max values are stored per channel.
    Scales and zero points are thus computed per channel as well.

    .. note:: If the running minimum equals to the running maximum, the scales
              and zero_points are set to 1.0 and 0.
    """
    min_val: torch.Tensor
    max_val: torch.Tensor
    ch_axis: Incomplete
    def __init__(self, ch_axis: int = 0, dtype=..., qscheme=..., reduce_range: bool = False, quant_min=None, quant_max=None, factory_kwargs=None, eps=..., is_dynamic: bool = False, **kwargs) -> None: ...
    def forward(self, x_orig): ...
    def _forward(self, x_orig): ...
    @torch.jit.export
    def calculate_qparams(self): ...
    def extra_repr(self): ...
    def _load_from_state_dict(self, state_dict: dict[str, Any], prefix: str, local_metadata: dict[str, torch.Tensor], strict: bool, missing_keys: list[str], unexpected_keys: list[str], error_msgs: list[str]): ...
    def _load_from_state_dict_script(self, state_dict: dict[str, Any], prefix: str, local_metadata: dict[str, torch.Tensor], strict: bool, missing_keys: list[str], unexpected_keys: list[str], error_msgs: list[str]): ...
    @torch.jit.export
    def reset_min_max_vals(self) -> None:
        """Resets the min/max values."""

class MovingAveragePerChannelMinMaxObserver(PerChannelMinMaxObserver):
    """Observer module for computing the quantization parameters based on the
    running per channel min and max values.

    This observer uses the tensor min/max statistics to compute the per channel
    quantization parameters. The module records the running minimum and maximum
    of incoming tensors, and uses this statistic to compute the quantization
    parameters.

    Args:
        averaging_constant: Averaging constant for min/max.
        ch_axis: Channel axis
        dtype: Quantized data type
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    The quantization parameters are computed the same way as in
    :class:`~torch.ao.quantization.observer.MovingAverageMinMaxObserver`, with the
    difference that the running min/max values are stored per channel.
    Scales and zero points are thus computed per channel as well.

    .. note:: If the running minimum equals to the running maximum, the scales
              and zero_points are set to 1.0 and 0.
    """
    averaging_constant: Incomplete
    def __init__(self, averaging_constant: float = 0.01, ch_axis: int = 0, dtype=..., qscheme=..., reduce_range: bool = False, quant_min=None, quant_max=None, eps=..., is_dynamic: bool = False, **kwargs) -> None: ...
    def forward(self, x_orig): ...

class HistogramObserver(UniformQuantizationObserverBase):
    """
    The module records the running histogram of tensor values along with
    min/max values. ``calculate_qparams`` will calculate scale and zero_point.

    Args:
        bins: Number of bins to use for the histogram
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    The scale and zero point are computed as follows:

    1. Create the histogram of the incoming inputs.
        The histogram is computed continuously, and the ranges per bin change
        with every new tensor observed.
    2. Search the distribution in the histogram for optimal min/max values.
        The search for the min/max values ensures the minimization of the
        quantization error with respect to the floating point model.
    3. Compute the scale and zero point the same way as in the
        :class:`~torch.ao.quantization.MinMaxObserver`
    """
    histogram: torch.Tensor
    min_val: torch.Tensor
    max_val: torch.Tensor
    bins: Incomplete
    dst_nbins: Incomplete
    upsample_rate: int
    def __init__(self, bins: int = 2048, dtype: torch.dtype = ..., qscheme=..., reduce_range: bool = False, quant_min=None, quant_max=None, factory_kwargs=None, eps=..., is_dynamic: bool = False, **kwargs) -> None: ...
    def _get_norm(self, delta_begin: torch.Tensor, delta_end: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        """
        Compute the norm of the values uniformaly distributed between
        delta_begin and delta_end.
        Currently only L2 norm is supported.

        norm = density * (integral_{begin, end} x^2)
             = density * (end^3 - begin^3) / 3
        """
    def _compute_quantization_error(self, next_start_bin: int, next_end_bin: int):
        """
        Compute the quantization error if we use start_bin to end_bin as the
        min and max to do the quantization.
        """
    def _non_linear_param_search(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Non-linear parameter search.

        An approximation for L2 error minimization for selecting min/max.
        By selecting new min/max, we filter out outliers in input distribution.
        This follows the implementation of NormMinimization::NonlinearQuantizationParamsSearch in
        caffe2/quantization/server/norm_minimization.cc
        """
    def _upscale_histogram(self, histogram: torch.Tensor, orig_min: torch.Tensor, orig_max: torch.Tensor, update_min: torch.Tensor, update_max: torch.Tensor): ...
    def _combine_histograms(self, orig_hist: torch.Tensor, orig_min: torch.Tensor, orig_max: torch.Tensor, update_hist: torch.Tensor, update_min: torch.Tensor, update_max: torch.Tensor) -> torch.Tensor: ...
    def reset_histogram(self, x: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor) -> None: ...
    def forward(self, x_orig: torch.Tensor) -> torch.Tensor: ...
    @torch.jit.export
    def calculate_qparams(self): ...
    def _save_to_state_dict(self, destination, prefix, keep_vars) -> None: ...
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None: ...
    def extra_repr(self): ...

class FixedQParamsObserver(ObserverBase):
    """
    Observer that simulates quantize and dequantize with fixed
    quantization parameters in training time. Only per tensor
    quantization is supported.

    Args:
        `scale` (float): fixed scale for the observer
        `zero_point` (int): fixed zero point for the observer
        `dtype`, `qscheme`, `quant_min`, `quant_max`
    """
    scale: torch.Tensor
    zero_point: torch.Tensor
    quant_min: Incomplete
    quant_max: Incomplete
    dtype: Incomplete
    qscheme: Incomplete
    def __init__(self, scale, zero_point, dtype=..., qscheme=..., quant_min: int = 0, quant_max: int = 255, is_dynamic: bool = False, **kwargs) -> None: ...
    def forward(self, X): ...
    @torch.jit.export
    def calculate_qparams(self): ...

class PlaceholderObserver(ObserverBase):
    """
    Observer that doesn't do anything and just passes its configuration to the
    quantized module's ``.from_float()``.

    Can be used for quantization to float16 which doesn't require determining
    ranges.

    Args:
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        quant_min: minimum value in quantized domain (TODO: align behavior with other observers)
        quant_max: maximum value in quantized domain
        custom_op_name: (temporary) specify this observer for an operator that doesn't require any observation
                        (Can be used in Graph Mode Passes for special case ops).
        compute_dtype (deprecated): if set, marks the future quantize function to use
                       dynamic quantization instead of static quantization.
                       This field is deprecated, use `is_dynamic=True` instead.
        is_dynamic: if True, the `quantize` function in the reference model
                    representation taking stats from this observer instance will
                    use dynamic quantization.
    """
    dtype: Incomplete
    qscheme: Incomplete
    quant_min: Incomplete
    quant_max: Incomplete
    eps: Incomplete
    custom_op: Incomplete
    def __init__(self, dtype=..., custom_op_name: str = '', compute_dtype=None, quant_min=None, quant_max=None, qscheme=None, eps=None, is_dynamic: bool = False) -> None: ...
    def forward(self, x): ...
    @torch.jit.export
    def extra_repr(self): ...
    @torch.jit.export
    def calculate_qparams(self) -> None: ...

class RecordingObserver(ObserverBase):
    """
    The module is mainly for debug and records the tensor values during runtime.

    Args:
        dtype: Quantized data type
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
    """
    __annotations__: Incomplete
    tensor_val: Incomplete
    def __init__(self, dtype=...) -> None: ...
    def forward(self, x): ...
    @torch.jit.export
    def calculate_qparams(self) -> None: ...
    @torch.jit.export
    def get_tensor_value(self): ...

class NoopObserver(ObserverBase):
    """
    Observer that doesn't do anything and just passes its configuration to the
    quantized module's ``.from_float()``.

    Primarily used for quantization to float16 which doesn't require determining
    ranges.

    Args:
        dtype: Quantized data type
        custom_op_name: (temporary) specify this observer for an operator that doesn't require any observation
                        (Can be used in Graph Mode Passes for special case ops).
    """
    dtype: Incomplete
    custom_op: Incomplete
    def __init__(self, dtype=..., custom_op_name: str = '') -> None: ...
    def forward(self, x): ...
    @torch.jit.export
    def calculate_qparams(self) -> None: ...

class ReuseInputObserver(ObserverBase):
    """This observer is used when we want to reuse the observer from the operator
    that produces the input Tensor, typically used for operators like reshape, e.g.
    ```
    x0 = ...
    x1 = x0.reshape()
    ```
    if we configure x0 to be observed by some observer, let's say MinMaxObserver,
    and reshape is configured with ReuseInputObserver, we'll reuse the observer instance
    for x0 for x1 (output of reshape). If x0 is not observed, we also won't observe x1.

    Note: this is only enabled in FX Graph Mode Quantization
    """
    def __init__(self) -> None: ...
    def forward(self, x): ...
    @torch.jit.export
    def calculate_qparams(self) -> None: ...

class MappingType(Enum):
    """How floating point number is mapped to integer number

    symmetric mapping means floating point range is symmetrically mapped to integer range
    let's say we have floating point range (-3.5, 10.2) and integer range (-8, 7) (int4)
    we'll use (-10.2, 10.2) as the range for floating point and map that to (-8, 7)
    e.g. scale = (10.2 - (-10.2)) / (7 - (-8))

    SYMMETRIC_NO_CLIPPING_ERR is a variant of symmetric mapping, where the scale is the max of smin
    and smax, where smin = min_val_neg / quant_min, and smax = max_val_pos / quant_max. By calculating
    smin and smax individually, there can be less round error on negative values, and no out-of-range
    of all floating point values.

    asymmetric mapping means we just directly map the floating point range to integer range,
    for the above example, we will map (-3.5, 10.2) to (-8, 7) and calculate quantization parameter
    based on this mapping
    e.g. scale = (10.2 - (-3.5)) / (7 - (-8))
    """
    SYMMETRIC = ...
    SYMMETRIC_NO_CLIPPING_ERR = ...
    ASYMMETRIC = ...

class ZeroPointDomain(Enum):
    """Enum that indicate whether zero_point is in integer domain or floating point domain

    integer domain: quantized_val = (float_val / scale) (integer) + zero_point (integer)
    float domain: quantized_val = (float_val - (zero_point (float) - scale * mid_point)) / scale
    none domain: quantized_val = (float_val / scale)
    """
    INT = ...
    FLOAT = ...
    NONE = ...

class TorchAODType(Enum):
    """
    Placeholder for dtypes that do not exist in PyTorch core yet.
    """
    INT1 = ...
    INT2 = ...
    INT3 = ...
    INT4 = ...
    INT5 = ...
    INT6 = ...
    INT7 = ...

@dataclass(frozen=True)
class Granularity:
    """
    Base class for representing the granularity of quantization.

    This class serves as a parent for specific granularity types used in
    quantization operations, such as per-tensor or per-axis quantization.
    """

@dataclass(frozen=True)
class PerBlock(Granularity):
    """
    Represents per-block granularity in quantization. See
    :func:`~torchao.quantization.quant_primitives.quantize_affine` for docs for
    `block_size`

    Attributes:
        block_size (Tuple[int, ...]): The size of each quantization group
    """
    block_size: tuple[int, ...]

@dataclass(frozen=True)
class PerTensor(Granularity):
    """
    Represents per-tensor granularity in quantization.

    This granularity type calculates the quantization parameters
    based off the entire tensor.

    """

@dataclass(frozen=True)
class PerAxis(Granularity):
    """
    Represents per-axis granularity in quantization.

    This granularity type calculates different quantization parameters
    along a specified axis of the tensor.

    For example if the input tensor is shape [8, 16] and axis=0, then
    the quantization parameters are calculated for each row of the tensor.
    Giving a total of 8 quantization parameters.

    Attributes:
        axis (int): The axis along which reduction is performed.
    """
    axis: int

@dataclass(frozen=True)
class PerGroup(Granularity):
    """
    Represents per-channel group granularity in quantization.

    This granularity type calculates different quantization parameters
    for each group of <group_size> elements.

    For example if the input tensor is shape [8, 16], and the group size is 4, then
    the input tensor is reshaped to [64, 4]
    quantization parameters are calculated for each group of 4 elements,
    giving a total of 64 quantization parameters.

    Attributes:
        group_size (int): The size of each quantization group

    """
    group_size: int

class PerRow(Granularity):
    """
    Represents row-wise granularity in quantization.

    This is a special case of per-axis quantization and is unique to Float8 matmuls
    where the input is quantized with a block_size of (1, ..., input.shape[-1]). And the weight
    is quantized with a block_size of (1, weight.shape[1]).
    """
class PerToken(Granularity):
    """
    Represents per-token granularity in quantization.

    This granularity type calculates a different set of quantization parameters
    for each token, which is represented as the last dimension of the tensor.

    For example, if the input tensor has shape [2, 3, 4], then there are 6 tokens
    with 4 elements each, and we will calculate 6 sets of quantization parameters,
    one for each token.

    If the input tensor has only two dimensions, e.g. [8, 16], then this is
    equivalent to `PerAxis(axis=0)`, which yields 8 sets of quantization parameters.
    """

def get_block_size(input_shape: tuple[int, ...], granularity: Granularity) -> tuple[int, ...]:
    """Get the block size based on the input shape and granularity type.

    Args:
        input_shape: The input tensor shape possibly more than 2 dimensions
        granularity: The granularity type of the quantization
    """

class AffineQuantizedObserverBase(ABC, torch.nn.Module, metaclass=abc.ABCMeta):
    """Observer module for affine quantization (https://github.com/pytorch/ao/tree/main/torchao/quantization#affine-quantization)

    Args:
      `granularity` and `block_size`: The granularity of the quantization,
        must specify at least one, if both are specified `block_size` takes precedence
        Current supported granularity type are `PerTensor` and `PerAxis`
      other args: please see `:class:torchao.dtypes.AffineQuantizedTensor`
    """
    with_args: Incomplete
    mapping_type: Incomplete
    target_dtype: Incomplete
    granularity: Incomplete
    quant_min: Incomplete
    quant_max: Incomplete
    eps: Incomplete
    scale_dtype: Incomplete
    zero_point_dtype: Incomplete
    preserve_zero: Incomplete
    zero_point_domain: Incomplete
    block_size: Incomplete
    original_dtype: Incomplete
    def __init__(self, mapping_type: MappingType, target_dtype: torch.dtype, granularity: Granularity, quant_min: int | None = None, quant_max: int | None = None, eps: float | None = None, scale_dtype: torch.dtype | None = None, zero_point_dtype: torch.dtype | None = None, preserve_zero: bool = True, zero_point_domain: ZeroPointDomain | None = ..., **kwargs) -> None: ...
    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """forward function should take the input tensor
        and updates internal stats and return the original input Tensor
        """
    @abstractmethod
    def calculate_qparams(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate quantization parameter based on the stats attached to the observer module
        and returns a tuple of scale and zero_point Tensor
        """
    def convert(self, model: torch.fx.GraphModule, observer_node: Node):
        """
        Converts the observer node in the graph into its quantized representation

        Args:
            model: graph module to conver the observer node in
            observer_node: the observer node to convert
        """

def get_observer_state_dict(mod):
    """
    Returns the state dict corresponding to the observer stats.
    Traverse the model state_dict and extract out the stats.
    """
def load_observer_state_dict(mod, obs_dict) -> None:
    """
    Given input model and a state_dict containing model observer stats,
    load the stats back into the model. The observer state_dict can be saved
    using torch.ao.quantization.get_observer_state_dict
    """

default_observer: Incomplete
default_placeholder_observer = PlaceholderObserver
default_debug_observer = RecordingObserver
default_weight_observer: Incomplete
weight_observer_range_neg_127_to_127: Incomplete
default_histogram_observer: Incomplete
default_per_channel_weight_observer: Incomplete
per_channel_weight_observer_range_neg_127_to_127: Incomplete
default_dynamic_quant_observer: Incomplete
default_float_qparams_observer: Incomplete
default_float_qparams_observer_4bit: Incomplete
default_fixed_qparams_range_neg1to1_observer: Incomplete
default_fixed_qparams_range_0to1_observer: Incomplete
default_symmetric_fixed_qparams_observer = default_fixed_qparams_range_neg1to1_observer
default_affine_fixed_qparams_observer = default_fixed_qparams_range_0to1_observer
default_reuse_input_observer = ReuseInputObserver
