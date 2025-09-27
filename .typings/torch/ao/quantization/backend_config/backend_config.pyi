import torch
from _typeshed import Incomplete
from dataclasses import dataclass
from enum import Enum
from torch.ao.quantization.utils import Pattern
from typing import Any, Callable

__all__ = ['BackendConfig', 'BackendPatternConfig', 'DTypeConfig', 'DTypeWithConstraints', 'ObservationType']

class ObservationType(Enum):
    """An enum that represents different ways of how an operator/operator pattern
    should be observed
    """
    OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT = 0
    OUTPUT_SHARE_OBSERVER_WITH_INPUT = 1
    INPUT_OUTPUT_NOT_OBSERVED = 2

@dataclass
class DTypeWithConstraints:
    """
    Config for specifying additional constraints for a given dtype, such as quantization
    value ranges, scale value ranges, and fixed quantization params, to be used in
    :class:`~torch.ao.quantization.backend_config.DTypeConfig`.

    The constraints currently supported are:

    * `quant_min_lower_bound` and `quant_max_upper_bound`: Lower and upper
      bounds for the minimum and maximum quantized values respectively. If
      the QConfig's `quant_min` and `quant_max` fall outside this range,
      then the QConfig will be ignored.

    * `scale_min_lower_bound` and `scale_max_upper_bound`: Lower and upper
      bounds for the minimum and maximum scale values respectively. If the
      QConfig's minimum scale value (currently exposed as `eps`) falls below
      the lower bound, then the QConfig will be ignored. Note that the upper
      bound is currently not enforced.

    * `scale_exact_match` and `zero_point_exact_match`: Exact match requirements
      for scale and zero point, to be used for operators with fixed quantization
      parameters such as sigmoid and tanh. If the observer specified in the QConfig
      is neither `FixedQParamsObserver` nor `FixedQParamsFakeQuantize`, or if
      the quantization parameters don't match, then the QConfig will be ignored.
    """
    dtype: torch.dtype | None = ...
    quant_min_lower_bound: int | float | None = ...
    quant_max_upper_bound: int | float | None = ...
    scale_min_lower_bound: int | float | None = ...
    scale_max_upper_bound: int | float | None = ...
    scale_exact_match: float | None = ...
    zero_point_exact_match: int | None = ...

@dataclass
class DTypeConfig:
    '''
    Config object that specifies the supported data types passed as arguments to
    quantize ops in the reference model spec, for input and output activations,
    weights, and biases.

    For example, consider the following reference model:

      quant1 - [dequant1 - fp32_linear - quant2] - dequant2

    The pattern in the square brackets refers to the reference pattern of
    statically quantized linear. Setting the input dtype as `torch.quint8`
    in the DTypeConfig means we pass in `torch.quint8` as the dtype argument
    to the first quantize op (quant1). Similarly, setting the output dtype as
    `torch.quint8` means we pass in `torch.quint8` as the dtype argument to
    the second quantize op (quant2).

    Note that the dtype here does not refer to the interface dtypes of the
    op. For example, the "input dtype" here is not the dtype of the input
    tensor passed to the quantized linear op. Though it can still be the
    same as the interface dtype, this is not always the case, e.g. the
    interface dtype is fp32 in dynamic quantization but the "input dtype"
    specified in the DTypeConfig would still be quint8. The semantics of
    dtypes here are the same as the semantics of the dtypes specified in
    the observers.

    These dtypes are matched against the ones specified in the user\'s
    QConfig. If there is a match, and the QConfig satisfies the constraints
    specified in the DTypeConfig (if any), then we will quantize the given
    pattern using this DTypeConfig. Otherwise, the QConfig is ignored and
    the pattern will not be quantized.

    Example usage::

        >>> # xdoctest: +SKIP(failing)
        >>> dtype_config1 = DTypeConfig(
        ...     input_dtype=torch.quint8,
        ...     output_dtype=torch.quint8,
        ...     weight_dtype=torch.qint8,
        ...     bias_dtype=torch.float)

        >>> dtype_config2 = DTypeConfig(
        ...     input_dtype=DTypeWithConstraints(
        ...         dtype=torch.quint8,
        ...         quant_min_lower_bound=0,
        ...         quant_max_upper_bound=255,
        ...     ),
        ...     output_dtype=DTypeWithConstraints(
        ...         dtype=torch.quint8,
        ...         quant_min_lower_bound=0,
        ...         quant_max_upper_bound=255,
        ...     ),
        ...     weight_dtype=DTypeWithConstraints(
        ...         dtype=torch.qint8,
        ...         quant_min_lower_bound=-128,
        ...         quant_max_upper_bound=127,
        ...     ),
        ...     bias_dtype=torch.float)

        >>> dtype_config1.input_dtype
        torch.quint8

        >>> dtype_config2.input_dtype
        torch.quint8

        >>> dtype_config2.input_dtype_with_constraints
        DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=0, quant_max_upper_bound=255, scale_min_lower_bound=None, scale_max_upper_bound=None)
    '''
    input_dtype_with_constraints: DTypeWithConstraints
    output_dtype_with_constraints: DTypeWithConstraints
    weight_dtype_with_constraints: DTypeWithConstraints
    bias_dtype: torch.dtype | None
    is_dynamic: bool | None
    def __init__(self, input_dtype: torch.dtype | DTypeWithConstraints | None = None, output_dtype: torch.dtype | DTypeWithConstraints | None = None, weight_dtype: torch.dtype | DTypeWithConstraints | None = None, bias_dtype: torch.dtype | None = None, is_dynamic: bool | None = None) -> None: ...
    @property
    def input_dtype(self) -> torch.dtype | None: ...
    @property
    def output_dtype(self) -> torch.dtype | None: ...
    @property
    def weight_dtype(self) -> torch.dtype | None: ...
    @classmethod
    def from_dict(cls, dtype_config_dict: dict[str, Any]) -> DTypeConfig:
        '''
        Create a ``DTypeConfig`` from a dictionary with the following items (all optional):
            "input_dtype": torch.dtype or ``DTypeWithConstraints``
            "output_dtype": torch.dtype or ``DTypeWithConstraints``
            "weight_dtype": torch.dtype or ``DTypeWithConstraints``
            "bias_type": torch.dtype
            "is_dynamic": bool
        '''
    def to_dict(self) -> dict[str, Any]:
        """
        Convert this ``DTypeConfig`` to a dictionary with the items described in
        :func:`~torch.ao.quantization.backend_config.DTypeConfig.from_dict`.
        """

class BackendConfig:
    '''Config that defines the set of patterns that can be quantized on a given backend, and how reference
    quantized models can be produced from these patterns.

    A pattern in this context refers to a module, a functional, an operator, or a directed acyclic graph
    of the above. Each pattern supported on the target backend can be individually configured through
    :class:`~torch.ao.quantization.backend_config.BackendPatternConfig` in terms of:

    (1) The supported input/output activation, weight, and bias data types

    (2) How observers and quant/dequant ops are inserted in order to construct the reference pattern, and

    (3) (Optionally) Fusion, QAT, and reference module mappings.

    The format of the patterns is described in:
    https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/backend_config/README.md

    Example usage::

        import torch
        from torch.ao.quantization.backend_config import (
            BackendConfig,
            BackendPatternConfig,
            DTypeConfig,
            ObservationType,
        )

        weighted_int8_dtype_config = DTypeConfig(
            input_dtype=torch.quint8,
            output_dtype=torch.quint8,
            weight_dtype=torch.qint8,
            bias_dtype=torch.float)

        def fuse_conv2d_relu(is_qat, conv, relu):
            return torch.ao.nn.intrinsic.ConvReLU2d(conv, relu)

        # For quantizing Linear
        linear_config = BackendPatternConfig(torch.nn.Linear)             .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)             .add_dtype_config(weighted_int8_dtype_config)             .set_root_module(torch.nn.Linear)             .set_qat_module(torch.ao.nn.qat.Linear)             .set_reference_quantized_module(torch.ao.nn.quantized.reference.Linear)

        # For fusing Conv2d + ReLU into ConvReLU2d
        conv_relu_config = BackendPatternConfig((torch.nn.Conv2d, torch.nn.ReLU))             .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)             .add_dtype_config(weighted_int8_dtype_config)             .set_fused_module(torch.ao.nn.intrinsic.ConvReLU2d)             .set_fuser_method(fuse_conv2d_relu)

        # For quantizing ConvReLU2d
        fused_conv_relu_config = BackendPatternConfig(torch.ao.nn.intrinsic.ConvReLU2d)             .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)             .add_dtype_config(weighted_int8_dtype_config)             .set_root_module(torch.nn.Conv2d)             .set_qat_module(torch.ao.nn.intrinsic.qat.ConvReLU2d)             .set_reference_quantized_module(torch.ao.nn.quantized.reference.Conv2d)

        backend_config = BackendConfig("my_backend")             .set_backend_pattern_config(linear_config)             .set_backend_pattern_config(conv_relu_config)             .set_backend_pattern_config(fused_conv_relu_config)

    '''
    name: Incomplete
    _pattern_complex_format_to_config: dict[Pattern, BackendPatternConfig]
    def __init__(self, name: str = '') -> None: ...
    def __repr__(self) -> str: ...
    def set_name(self, name: str) -> BackendConfig:
        """
        Set the name of the target backend.
        """
    def set_backend_pattern_config(self, config: BackendPatternConfig) -> BackendConfig:
        """
        Set the config for an pattern that can be run on the target backend.
        This overrides any existing config for the given pattern.
        """
    def set_backend_pattern_configs(self, configs: list[BackendPatternConfig]) -> BackendConfig:
        """
        Set the configs for patterns that can be run on the target backend.
        This overrides any existing config for a given pattern if it was previously registered already.
        """
    @property
    def configs(self) -> list[BackendPatternConfig]:
        """
        Return a copy of the list of configs set in this `BackendConfig`.
        """
    @classmethod
    def from_dict(cls, backend_config_dict: dict[str, Any]) -> BackendConfig:
        '''
        Create a ``BackendConfig`` from a dictionary with the following items:

            "name": the name of the target backend

            "configs": a list of dictionaries that each represents a `BackendPatternConfig`

        '''
    def to_dict(self) -> dict[str, Any]:
        """
        Convert this ``BackendConfig`` to a dictionary with the items described in
        :func:`~torch.ao.quantization.backend_config.BackendConfig.from_dict`.
        """

class BackendPatternConfig:
    """
    Config object that specifies quantization behavior for a given operator pattern.
    For a detailed example usage, see :class:`~torch.ao.quantization.backend_config.BackendConfig`.
    """
    pattern: Pattern | None
    observation_type: Incomplete
    dtype_configs: list[DTypeConfig]
    root_module: type[torch.nn.Module] | None
    qat_module: type[torch.nn.Module] | None
    reference_quantized_module: type[torch.nn.Module] | None
    fused_module: type[torch.nn.Module] | None
    fuser_method: Callable | None
    _root_node_getter: Callable | None
    _extra_inputs_getter: Callable | None
    _num_tensor_args_to_observation_type: dict[int, ObservationType]
    _input_type_to_index: dict[str, int]
    _pattern_complex_format: Pattern | None
    def __init__(self, pattern: Pattern | None = None) -> None: ...
    def __repr__(self) -> str: ...
    def set_pattern(self, pattern: Pattern) -> BackendPatternConfig:
        """
        Set the pattern to configure.

        The pattern can be a float module, functional operator, pytorch operator, or a tuple
        combination of the above. Tuple patterns are treated as sequential patterns, and
        currently only tuples of 2 or 3 elements are supported.
        """
    def set_observation_type(self, observation_type: ObservationType) -> BackendPatternConfig:
        """
        Set how observers should be inserted in the graph for this pattern.

        Observation type here refers to how observers (or quant-dequant ops) will be placed
        in the graph. This is used to produce the desired reference patterns understood by
        the backend. Weighted ops such as linear and conv require different observers
        (or quantization parameters passed to quantize ops in the reference model) for the
        input and the output.

        There are two observation types:

            `OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT` (default): the output observer instance
            will be different from the input. This is the most common observation type.

            `OUTPUT_SHARE_OBSERVER_WITH_INPUT`: the output observer instance will be the
            same as the input. This is useful for operators like `cat`.

        Note: This will be renamed in the near future, since we will soon insert QuantDeQuantStubs
        with observers (and fake quantizes) attached instead of observers themselves.
        """
    def add_dtype_config(self, dtype_config: DTypeConfig) -> BackendPatternConfig:
        """
        Add a set of supported data types passed as arguments to quantize ops in the
        reference model spec.
        """
    def set_dtype_configs(self, dtype_configs: list[DTypeConfig]) -> BackendPatternConfig:
        """
        Set the supported data types passed as arguments to quantize ops in the
        reference model spec, overriding all previously registered data types.
        """
    def set_root_module(self, root_module: type[torch.nn.Module]) -> BackendPatternConfig:
        """
        Set the module that represents the root for this pattern.

        When we construct the reference quantized model during the convert phase,
        the root modules (e.g. torch.nn.Linear for torch.ao.nn.intrinsic.LinearReLU)
        will be swapped to the corresponding reference quantized modules (e.g.
        torch.ao.nn.reference.quantized.Linear). This allows custom backends to
        specify custom reference quantized module implementations to match the
        numerics of their lowered operators. Since this is a one-to-one mapping,
        both the root module and the reference quantized module must be specified
        in the same BackendPatternConfig in order for the conversion to take place.
        """
    def set_qat_module(self, qat_module: type[torch.nn.Module]) -> BackendPatternConfig:
        """
        Set the module that represents the QAT implementation for this pattern.
        """
    def set_reference_quantized_module(self, reference_quantized_module: type[torch.nn.Module]) -> BackendPatternConfig:
        """
        Set the module that represents the reference quantized implementation for
        this pattern's root module.

        For more detail, see :func:`~torch.ao.quantization.backend_config.BackendPatternConfig.set_root_module`.
        """
    def set_fused_module(self, fused_module: type[torch.nn.Module]) -> BackendPatternConfig:
        """
        Set the module that represents the fused implementation for this pattern.
        """
    def set_fuser_method(self, fuser_method: Callable) -> BackendPatternConfig:
        """
        Set the function that specifies how to fuse this BackendPatternConfig's pattern.

        The first argument of this function should be `is_qat`, and the rest of the arguments
        should be the items in the tuple pattern. The return value of this function should be
        the resulting fused module.

        For example, the fuser method for the pattern `(torch.nn.Linear, torch.nn.ReLU)` can be:

            def fuse_linear_relu(is_qat, linear, relu):
                return torch.ao.nn.intrinsic.LinearReLU(linear, relu)

        For a more complicated example, see https://gist.github.com/jerryzh168/8bea7180a8ba3c279f2c9b050f2a69a6.
        """
    def _set_root_node_getter(self, root_node_getter: Callable) -> BackendPatternConfig: ...
    def _set_extra_inputs_getter(self, extra_inputs_getter: Callable) -> BackendPatternConfig: ...
    def _set_num_tensor_args_to_observation_type(self, num_tensor_args_to_observation_type: dict[int, ObservationType]) -> BackendPatternConfig: ...
    def _set_input_type_to_index(self, input_type_to_index: dict[str, int]) -> BackendPatternConfig: ...
    def _set_pattern_complex_format(self, pattern: Pattern) -> BackendPatternConfig:
        """
        Set the pattern to configure, using the reversed nested tuple format.

        See the BackendConfig README for more detail:
        https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/backend_config/README.md#advanced-pattern-specification
        """
    @classmethod
    def from_dict(cls, backend_pattern_config_dict: dict[str, Any]) -> BackendPatternConfig:
        '''
        Create a ``BackendPatternConfig`` from a dictionary with the following items:

            "pattern": the pattern being configured
            "observation_type": the :class:`~torch.ao.quantization.backend_config.ObservationType` that specifies how
            observers should be inserted for this pattern
            "dtype_configs": a list of dictionaries that represents :class:`~torch.ao.quantization.backend_config.DTypeConfig` s
            "root_module": a :class:`torch.nn.Module` that represents the root for this pattern
            "qat_module": a :class:`torch.nn.Module` that represents the QAT implementation for this pattern
            "reference_quantized_module": a :class:`torch.nn.Module` that represents the reference quantized
            implementation for this pattern\'s root module.
            "fused_module": a :class:`torch.nn.Module` that represents the fused implementation for this pattern
            "fuser_method": a function that specifies how to fuse the pattern for this pattern
            "pattern_complex_format": the pattern specified in the reversed nested tuple format (deprecated)

        '''
    def to_dict(self) -> dict[str, Any]:
        """
        Convert this ``BackendPatternConfig`` to a dictionary with the items described in
        :func:`~torch.ao.quantization.backend_config.BackendPatternConfig.from_dict`.
        """
