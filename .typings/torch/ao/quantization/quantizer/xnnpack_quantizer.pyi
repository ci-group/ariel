import functools
import torch
from _typeshed import Incomplete
from torch.ao.quantization.quantizer import Quantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import OperatorConfig, OperatorPatternType, QuantizationConfig
from torch.fx import Node
from typing import Callable

__all__ = ['XNNPACKQuantizer', 'get_symmetric_quantization_config']

@functools.lru_cache
def get_symmetric_quantization_config(is_per_channel: bool = False, is_qat: bool = False, is_dynamic: bool = False, act_qmin: int = -128, act_qmax: int = 127, weight_qmin: int = -127, weight_qmax: int = 127): ...

class XNNPACKQuantizer(Quantizer):
    """
    !!! DEPRECATED !!!
    XNNPACKQuantizer is a marked as deprected. It will be removed in the future.
    It has been moved to executorch.backends.xnnpack.quantizer.xnnpack_quantizer.XNNPACKQuantizer.
    Please use the new quantizer instead.
    """
    supported_config_and_operators: Incomplete
    STATIC_QAT_ONLY_OPS: Incomplete
    STATIC_OPS: Incomplete
    DYNAMIC_OPS: Incomplete
    global_config: QuantizationConfig | None
    operator_type_config: dict[torch._ops.OpOverloadPacket, QuantizationConfig | None]
    module_type_config: dict[Callable, QuantizationConfig | None]
    module_name_config: dict[str, QuantizationConfig | None]
    def __init__(self) -> None: ...
    @classmethod
    def get_supported_quantization_configs(cls) -> list[QuantizationConfig]: ...
    @classmethod
    def get_supported_operator_for_quantization_config(cls, quantization_config: QuantizationConfig | None) -> list[OperatorPatternType]: ...
    def set_global(self, quantization_config: QuantizationConfig) -> XNNPACKQuantizer: ...
    def set_operator_type(self, operator_type: torch._ops.OpOverloadPacket, quantization_config: QuantizationConfig) -> XNNPACKQuantizer: ...
    def set_module_type(self, module_type: Callable, quantization_config: QuantizationConfig):
        """Set quantization_config for a submodule with type: `module_type`, for example:
        quantizer.set_module_name(Sub) or quantizer.set_module_name(nn.Linear), it will quantize all supported operator/operator
        patterns in the submodule with this module type with the given `quantization_config`
        """
    def set_module_name(self, module_name: str, quantization_config: QuantizationConfig | None):
        '''Set quantization_config for a submodule with name: `module_name`, for example:
        quantizer.set_module_name("blocks.sub"), it will quantize all supported operator/operator
        patterns in the submodule with this module name with the given `quantization_config`
        '''
    def transform_for_annotation(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """Transforms scalar values to tensor attributes"""
    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """just handling global spec for now"""
    def _annotate_all_static_patterns(self, model: torch.fx.GraphModule, quantization_config: QuantizationConfig | None, filter_fn: Callable[[Node], bool] | None = None) -> torch.fx.GraphModule: ...
    def _annotate_all_dynamic_patterns(self, model: torch.fx.GraphModule, quantization_config: QuantizationConfig | None, filter_fn: Callable[[Node], bool] | None = None) -> torch.fx.GraphModule: ...
    def _annotate_for_static_quantization_config(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule: ...
    def _annotate_for_dynamic_quantization_config(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule: ...
    def validate(self, model: torch.fx.GraphModule) -> None: ...
    @classmethod
    def get_supported_operators(cls) -> list[OperatorConfig]: ...
