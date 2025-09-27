import torch
import typing
from dataclasses import dataclass
from torch.ao.quantization.quantizer import QuantizationSpec
from torch.fx import Node
from typing import Callable, NamedTuple

__all__ = ['OperatorConfig', 'OperatorPatternType', 'QuantizationConfig', 'get_input_act_qspec', 'get_output_act_qspec', 'get_weight_qspec', 'get_bias_qspec', 'OP_TO_ANNOTATOR', 'propagate_annotation']

@dataclass(eq=True, frozen=True)
class QuantizationConfig:
    input_activation: QuantizationSpec | None
    output_activation: QuantizationSpec | None
    weight: QuantizationSpec | None
    bias: QuantizationSpec | None
    is_qat: bool = ...
OperatorPatternType = typing.Annotated[list[Callable], None]
AnnotatorType = Callable[[torch.fx.GraphModule, QuantizationConfig | None, Callable[[Node], bool] | None], list[list[Node]] | None]
OP_TO_ANNOTATOR: dict[str, AnnotatorType]

class OperatorConfig(NamedTuple):
    config: QuantizationConfig
    operators: list[OperatorPatternType]

def get_input_act_qspec(quantization_config: QuantizationConfig | None): ...
def get_output_act_qspec(quantization_config: QuantizationConfig | None): ...
def get_weight_qspec(quantization_config: QuantizationConfig | None): ...
def get_bias_qspec(quantization_config: QuantizationConfig | None): ...
def propagate_annotation(model: torch.fx.GraphModule) -> None: ...
