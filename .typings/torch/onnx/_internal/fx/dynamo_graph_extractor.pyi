import torch
import torch.export as torch_export
import torch.fx
import types
from _typeshed import Incomplete
from collections.abc import Mapping, Sequence
from torch.onnx._internal import _exporter_legacy as _exporter_legacy, io_adapter as io_adapter
from torch.utils import _pytree as pytree
from typing import Any, Callable

class _PyTreeExtensionContext:
    """Context manager to register PyTree extension."""
    _extensions: dict[type, tuple[pytree.FlattenFunc, pytree.UnflattenFunc]]
    def __init__(self) -> None: ...
    def __enter__(self): ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: types.TracebackType | None) -> None: ...
    def register_pytree_node(self, class_type: type, flatten_func: pytree.FlattenFunc, unflatten_func: pytree.UnflattenFunc):
        """Register PyTree extension for a custom python type.

        Args:
            class_type: The custom python type.
            flatten_func: The flatten function.
            unflatten_func: The unflatten function.

        Raises:
            AssertionError: If the custom python type is already registered.
        """
    def _register_huggingface_model_output_extension(self): ...

class DynamoFlattenOutputStep(io_adapter.FlattenOutputStep):
    """Flatten nested collection and custom python types and return a flat list of elements.

    Extended from :class:`io_adapter.FlattenOutputStep` to support flattening arbitrary
    types via pytree extension. By default this supports many common user defined python
    types such as :class:`ModelOutput` from HuggingFace transformers.

    The pytree extension can be customized by passing in a ``_PyTreeExtensionContext``
    object. See :meth:`_PyTreeExtensionContext.register_pytree_node`.
    """
    _pytree_extension_context: Incomplete
    def __init__(self, pytree_extension_context: _PyTreeExtensionContext | None = None) -> None: ...
    def apply(self, model_outputs: Any, model: torch.nn.Module | Callable | torch_export.ExportedProgram | None = None) -> Sequence[Any]:
        """Flatten the model outputs, under the context of pytree extension."""

def _wrap_model_with_output_adapter(model: torch.nn.Module | Callable, output_adapter: DynamoFlattenOutputStep) -> Callable:
    """Wrap model with output adapter.

    This is a helper function to enable :func:`dynamo.export` on models that produce
    custom user defined types outputs. It wraps the model with an output adapter to
    convert the outputs to :func:`dynamo.export` compatible types, i.e. :class:`torch.Tensor`.

    The adapting logic is controlled by ``output_adapter``.

    Args:
        model: PyTorch model or function.
        output_adapter: Output adapter to apply to model output.
    Returns:
        Wrapped model.
    """

class DynamoExport(_exporter_legacy.FXGraphExtractor):
    """Generates a FX GraphModule using torch.dynamo.export API
    Args:
        aten_graph: If True, exports a graph with ATen operators.
                    If False, exports a graph with Python operators.
    """
    aten_graph: Incomplete
    def __init__(self, aten_graph: bool | None = None) -> None: ...
    def generate_fx(self, options: _exporter_legacy.ResolvedExportOptions, model: torch.nn.Module | Callable, model_args: Sequence[Any], model_kwargs: Mapping[str, Any]) -> torch.fx.GraphModule: ...
    def pre_export_passes(self, options: _exporter_legacy.ResolvedExportOptions, original_model: torch.nn.Module | Callable, fx_module: torch.fx.GraphModule, fx_module_args: Sequence[Any]): ...
