import torch
from .quantizer import QuantizationAnnotation, Quantizer
from _typeshed import Incomplete
from torch.fx import Node

__all__ = ['ComposableQuantizer']

class ComposableQuantizer(Quantizer):
    """
    ComposableQuantizer allows users to combine more than one quantizer into a single quantizer.
    This allows users to quantize a model with multiple quantizers. E.g., embedding quantization
    maybe supported by one quantizer while linear layers and other ops might be supported by another
    quantizer.

    ComposableQuantizer is initialized with a list of `Quantizer` instances.
    The order of the composition matters since that is the order in which the quantizers will be
    applies.
    Example:
    ```
    embedding_quantizer = EmbeddingQuantizer()
    linear_quantizer = MyLinearQuantizer()
    xnnpack_quantizer = (
        XNNPackQuantizer()
    )  # to handle ops not quantized by previous two quantizers
    composed_quantizer = ComposableQuantizer(
        [embedding_quantizer, linear_quantizer, xnnpack_quantizer]
    )
    prepared_m = prepare_pt2e(model, composed_quantizer)
    ```
    """
    quantizers: Incomplete
    _graph_annotations: dict[Node, QuantizationAnnotation]
    def __init__(self, quantizers: list[Quantizer]) -> None: ...
    def _record_and_validate_annotations(self, gm: torch.fx.GraphModule, quantizer: Quantizer) -> None: ...
    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """just handling global spec for now"""
    def transform_for_annotation(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule: ...
    def validate(self, model: torch.fx.GraphModule) -> None: ...
