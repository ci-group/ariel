import abc
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from torch import Tensor
from torch.ao.quantization import ObserverOrFakeQuantize
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.fx import Node
from typing import Callable

__all__ = ['Quantizer', 'QuantizationSpecBase', 'QuantizationSpec', 'FixedQParamsQuantizationSpec', 'EdgeOrNode', 'SharedQuantizationSpec', 'DerivedQuantizationSpec', 'QuantizationAnnotation']

class QuantizationSpecBase(ABC):
    """Base class for different types of quantization specs that allows users to
    specify how to quantize a Tensor (input/output of a Node) in the model
    """

@dataclass(eq=True, frozen=True)
class QuantizationSpec(QuantizationSpecBase):
    """Quantization spec for common operators that allows user to specify how to
    quantize a Tensor, this includes dtype, quant_min, quant_max etc.
    """
    dtype: torch.dtype
    observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor
    quant_min: int | None = ...
    quant_max: int | None = ...
    qscheme: torch.qscheme | None = ...
    ch_axis: int | None = ...
    is_dynamic: bool = ...
    def __post_init__(self) -> None: ...

@dataclass(eq=True, frozen=True)
class FixedQParamsQuantizationSpec(QuantizationSpecBase):
    dtype: torch.dtype
    scale: float
    zero_point: int
    quant_min: int | None = ...
    quant_max: int | None = ...
    qscheme: torch.qscheme | None = ...
    is_dynamic: bool = ...
EdgeOrNode = tuple[Node, Node] | Node

@dataclass(eq=True, frozen=True)
class SharedQuantizationSpec(QuantizationSpecBase):
    """
    Quantization spec for the Tensors whose quantization parameters are shared with other Tensors
    """
    edge_or_node: EdgeOrNode

@dataclass(eq=True, frozen=True)
class DerivedQuantizationSpec(QuantizationSpecBase):
    """Quantization spec for the Tensors whose quantization parameters are derived from other Tensors"""
    derived_from: list[EdgeOrNode]
    derive_qparams_fn: Callable[[list[ObserverOrFakeQuantize]], tuple[Tensor, Tensor]]
    dtype: torch.dtype
    quant_min: int | None = ...
    quant_max: int | None = ...
    qscheme: torch.qscheme | None = ...
    ch_axis: int | None = ...
    is_dynamic: bool = ...

@dataclass
class QuantizationAnnotation:
    """How are input arguemnt or output should be quantized,
    expressed as QuantizationSpec, this corresponds to how a Tensor in the
    operator Graph is observed (PTQ) or fake quantized (QAT)
    """
    input_qspec_map: dict[Node, QuantizationSpecBase | None] = field(default_factory=dict)
    output_qspec: QuantizationSpecBase | None = ...
    allow_implicit_sharing: bool = ...
    _annotated: bool = ...

class Quantizer(ABC, metaclass=abc.ABCMeta):
    def transform_for_annotation(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """Allows for user defined transforms to run before annotating the graph.
        This allows quantizer to allow quantizing part of the model that are otherwise not quantizable.
        For example quantizer can
        a) decompose a compound operator like scaled dot product attention,
        into bmm and softmax if quantizer knows how to quantize bmm/softmax but not sdpa
        or b) transform scalars to tensor to allow quantizing scalares.

        Note: this is an optional method
        """
    @abstractmethod
    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule: ...
    @abstractmethod
    def validate(self, model: torch.fx.GraphModule) -> None: ...
    def prepare_obs_or_fq_callback(self, model: torch.fx.GraphModule, edge_or_node_to_obs_or_fq: dict[EdgeOrNode, ObserverOrFakeQuantize]) -> None:
        """A callback that will be called after the observers or fake quants are created
        for each sharing group, but before they are inserted into the graph. The
        callback can be used to make final quantization adjustments, such as enforcing
        specific scale and zero point on model input or output.

        Args:
          * `model`: the graph module being prepared.
          * `edge_or_node_to_obs_or_fq`: a dictionary mapping each annotated edge and
            node to the corresponding observer or fake quant object. Note that multiple
            edges and/or nodes can map to the same observer / fake quant instance if
            they were annotated with SharedQuantizationSpec. This dictionary can be
            modified by the callback.
        """
