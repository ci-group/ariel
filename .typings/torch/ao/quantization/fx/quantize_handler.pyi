import torch
from _typeshed import Incomplete
from abc import ABC
from torch.ao.quantization.utils import NodePattern
from typing import Callable

__all__ = ['QuantizeHandler', 'BinaryOpQuantizeHandler', 'CatQuantizeHandler', 'ConvReluQuantizeHandler', 'LinearReLUQuantizeHandler', 'BatchNormQuantizeHandler', 'EmbeddingQuantizeHandler', 'RNNDynamicQuantizeHandler', 'DefaultNodeQuantizeHandler', 'FixedQParamsOpQuantizeHandler', 'CopyNodeQuantizeHandler', 'GeneralTensorShapeOpQuantizeHandler', 'CustomModuleQuantizeHandler', 'StandaloneModuleQuantizeHandler']

class QuantizeHandler(ABC):
    """Base handler class for the quantizer patterns"""
    node_pattern: Incomplete
    modules: Incomplete
    root_node: Incomplete
    is_custom_module_: Incomplete
    is_standalone_module_: Incomplete
    num_tensor_args: int
    def __init__(self, node_pattern: NodePattern, modules: dict[str, torch.nn.Module], root_node_getter: Callable | None = None, is_custom_module: bool = False, is_standalone_module: bool = False) -> None:
        """Records pattern information in __init__, which will be used
        in convert
        """
    def is_general_tensor_value_op(self) -> bool:
        """
        Returns True if the operator works for both floating point and
        quantized input, and does some computation based on the input Tensor,
        or the ops that only re-arranges the Tensor values or query some metadata
        about the Tensor
        so we need to insert observer/fake_quant for the output of the
        operator (same observer instance as input)
        since the distribution of values is different for input and output
        Tensors (for HistogramObserver) while they share the same quantization
        parameters
        Example operator: avgpool2d, reshape, transpose, maxpool2d
        Example observed operator:
        observer_0 - avgpool2d - observer_0 (same observer instance as input)
        """
    def is_custom_module(self): ...
    def is_standalone_module(self): ...

class BinaryOpQuantizeHandler(QuantizeHandler): ...
class CatQuantizeHandler(QuantizeHandler): ...
class ConvReluQuantizeHandler(QuantizeHandler): ...
class LinearReLUQuantizeHandler(QuantizeHandler): ...
class BatchNormQuantizeHandler(QuantizeHandler): ...
class EmbeddingQuantizeHandler(QuantizeHandler): ...
class RNNDynamicQuantizeHandler(QuantizeHandler): ...
class DefaultNodeQuantizeHandler(QuantizeHandler):
    """Common quantized op, first input and first output will be quantized"""
class FixedQParamsOpQuantizeHandler(QuantizeHandler): ...
class CopyNodeQuantizeHandler(QuantizeHandler): ...
class GeneralTensorShapeOpQuantizeHandler(QuantizeHandler): ...
class CustomModuleQuantizeHandler(QuantizeHandler): ...
class StandaloneModuleQuantizeHandler(QuantizeHandler): ...
