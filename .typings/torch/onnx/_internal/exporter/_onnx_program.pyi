import contextlib
import onnxruntime as ort
import os
import tempfile
import torch
from _typeshed import Incomplete
from collections.abc import Sequence
from torch.onnx._internal._lazy_import import onnx, onnxscript_ir as ir
from typing import Any, Callable

__all__ = ['ONNXProgram']

class ONNXProgram:
    """A class to represent an ONNX program that is callable with torch tensors.

    Attributes:
        model: The ONNX model as an ONNX IR model object.
        exported_program: The exported program that produced the ONNX model.
    """
    model: ir.Model
    exported_program: Incomplete
    _inference_session: ort.InferenceSession | None
    _tempdir: tempfile.TemporaryDirectory | None
    _capture_strategy: str | None
    def __init__(self, model: ir.Model, exported_program: torch.export.ExportedProgram | None) -> None:
        """Initialize the ONNX program with the specified model and exported program.
        Args:
            model: The ONNX model.
            exported_program: The exported program that produced the ONNX model. Optional.
        """
    def __repr__(self) -> str: ...
    def __call__(self, *args, **kwargs) -> Sequence[torch.Tensor]:
        """Run the ONNX model with the same arguments you would provide to the GraphModule."""
    def compute_values(self, value_names: Sequence[str], args=(), kwargs=None) -> Sequence[torch.Tensor]:
        """Compute the values of the specified names in the ONNX model.

        This method is used to compute the values of the specified names in the ONNX model.
        The values are returned as a dictionary mapping names to tensors.

        Args:
            value_names: The names of the values to compute.

        Returns:
            A dictionary mapping names to tensors.
        """
    @property
    def model_proto(self) -> onnx.ModelProto:
        """Return the ONNX ``ModelProto`` object."""
    def optimize(self) -> None:
        """Optimize the ONNX model.

        This method optimizes the ONNX model by performing constant folding and
        eliminating redundancies in the graph. The optimization is done in-place.
        """
    def save(self, destination: str | os.PathLike, *, include_initializers: bool = True, keep_initializers_as_inputs: bool = False, external_data: bool | None = None):
        """Save the ONNX model to the specified destination.

        When ``external_data`` is ``True`` or the model is larger than 2GB,
        the weights are saved as external data in a separate file.

        Initializer (model weights) serialization behaviors:

        * ``include_initializers=True``, ``keep_initializers_as_inputs=False`` (default):
          The initializers are included in the saved model.
        * ``include_initializers=True``, ``keep_initializers_as_inputs=True``:
          The initializers are included in the saved model and kept as model inputs.
          Choose this option if you want the ability to override the model weights
          during inference.
        * ``include_initializers=False``, ``keep_initializers_as_inputs=False``:
          The initializers are not included in the saved model and are not listed
          as model inputs. Choose this option if you want to attach the initializers
          to the ONNX model in a separate, post-processing, step.
        * ``include_initializers=False``, ``keep_initializers_as_inputs=True``:
          The initializers are not included in the saved model but are listed as model
          inputs. Choose this option if you want to supply the initializers during
          inference and want to minimize the size of the saved model.

        Args:
            destination: The path to save the ONNX model to.
            include_initializers: Whether to include the initializers in the saved model.
            keep_initializers_as_inputs: Whether to keep the initializers as inputs in the saved model.
                If `True`, the initializers are added as inputs to the model which means they can be overwritten.
                by providing the initializers as model inputs.
            external_data: Whether to save the weights as external data in a separate file.

        Raises:
            TypeError: If ``external_data`` is ``True`` and ``destination`` is not a file path.
        """
    def apply_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Apply the weights from the specified state dict to the ONNX model.

        Use this method to replace FakeTensors or other weights.

        Args:
            state_dict: The state dict containing the weights to apply to the ONNX model.
        """
    def initialize_inference_session(self, initializer: Callable[[str | bytes], ort.InferenceSession] = ...) -> None:
        """Initialize the ONNX Runtime inference session.

        Args:
            initializer: The function to initialize the ONNX Runtime inference
                session with the specified model. By default, it uses the
                :func:`_ort_session_initializer` function.
        """
    def release(self) -> None:
        """Release the inference session.

        You may call this method to release the resources used by the inference session.
        """
    def _rename_dynamic_axes(self, dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any]) -> None:
        """Rename dynamic axes in a model according to the specified dynamic_axes names."""
