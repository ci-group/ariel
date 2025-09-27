import torch._C._onnx as _C_onnx
from _typeshed import Incomplete
from torch.onnx import _constants as _constants

class _InternalGlobals:
    """Globals used internally by ONNX exporter.

    NOTE: Be very judicious when adding any new variables. Do not create new
    global variables unless they are absolutely necessary.
    """
    _export_onnx_opset_version: Incomplete
    _training_mode: _C_onnx.TrainingMode
    _in_onnx_export: bool
    export_training: bool
    operator_export_type: _C_onnx.OperatorExportTypes
    onnx_shape_inference: bool
    _autograd_inlining: bool
    def __init__(self) -> None: ...
    @property
    def training_mode(self):
        """The training mode for the exporter."""
    @training_mode.setter
    def training_mode(self, training_mode: _C_onnx.TrainingMode): ...
    @property
    def export_onnx_opset_version(self) -> int:
        """Opset version used during export."""
    @export_onnx_opset_version.setter
    def export_onnx_opset_version(self, value: int): ...
    @property
    def in_onnx_export(self) -> bool:
        """Whether it is in the middle of ONNX export."""
    @in_onnx_export.setter
    def in_onnx_export(self, value: bool): ...
    @property
    def autograd_inlining(self) -> bool:
        """Whether Autograd must be inlined."""
    @autograd_inlining.setter
    def autograd_inlining(self, value: bool): ...

GLOBALS: Incomplete
