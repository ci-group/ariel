import torch
import torch.fx
from _typeshed import Incomplete
from torch._jit_internal import boolean_dispatched as boolean_dispatched
from torch.fx import Transformer as Transformer
from torch.fx.node import Argument as Argument, Target as Target
from torch.fx.operator_schemas import _torchscript_type_to_python_type as _torchscript_type_to_python_type
from typing import Any

class AnnotateTypesWithSchema(Transformer):
    """
    Use Python function signatures to annotate types for `Nodes` within an FX graph.
    This pulls out Python function signatures for:

        1. Standard `torch.nn` Module calls
        2. `torch.nn.functional` calls
        3. Attribute fetches via `get_attr`

    Example usage:

        m = torchvision.models.resnet18()

        traced = torch.fx.symbolic_trace(m)

        traced = AnnotateTypesWithSchema(traced).transform()

    """
    annotate_functionals: Incomplete
    annotate_modules: Incomplete
    annotate_get_attrs: Incomplete
    def __init__(self, module: torch.nn.Module, annotate_functionals: bool = True, annotate_modules: bool = True, annotate_get_attrs: bool = True) -> None: ...
    def call_function(self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]): ...
    def call_module(self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]): ...
    def get_attr(self, target: torch.fx.node.Target, args: tuple[Argument, ...], kwargs: dict[str, Any]): ...
    def _extract_python_return_type(self, target: Target) -> Any | None:
        """
        Given a Python call target, try to extract the Python return annotation
        if it is available, otherwise return None

        Args:

            target (Callable): Python callable to get return annotation for

        Returns:

            Optional[Any]: Return annotation from the `target`, or None if it was
                not available.
        """
