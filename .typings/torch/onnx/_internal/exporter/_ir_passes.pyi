from _typeshed import Incomplete
from collections.abc import Sequence
from torch.onnx._internal._lazy_import import onnxscript_ir as ir
from torch.onnx._internal.exporter import _constants as _constants

_ONNX_DOMAIN: str
logger: Incomplete

def rename_inputs(model: ir.Model, new_names: Sequence[str]) -> None: ...
def rename_outputs(model: ir.Model, new_names: Sequence[str]) -> None: ...
def _all_values(model: ir.Model):
    """Yield all values in a model."""
def _replace_names(shape_expr: str, rename_mapping: dict[str, str]) -> str:
    """Replace all known names in a shape expression with new names."""
def rename_axis(model: ir.Model, rename_mapping: dict[str, str]) -> None:
    """Rename dynamic axes in a model according to the specified dynamic_axes names."""
def add_torchlib_common_imports(model: ir.Model, opset_version: int = ...) -> None:
    """Hack to add torchlib common imports to the model."""
def _maybe_set_opset_version(opset_imports: dict[str, int], domain: str, version: int | None) -> None:
    """Set the opset version for the domain."""
def add_opset_imports(model: ir.Model) -> None:
    """Collect all opsets used and add opset imports to the model and functions."""
