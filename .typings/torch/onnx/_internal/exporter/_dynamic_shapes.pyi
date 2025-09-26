import inspect
from collections.abc import Sequence
from torch.export.dynamic_shapes import Dim as Dim, _DimHint as _DimHint
from torch.utils import _pytree as _pytree
from typing import Any

def from_dynamic_axes_to_dynamic_shapes(model, args: tuple[Any, ...], kwargs: dict[str, Any] | None, *, dynamic_axes=None, output_names: set[str], input_names: Sequence[str] | None = None) -> tuple[dict[str, Any | None] | None, tuple[Any, ...], dict[str, Any] | None]:
    '''
    Converts dynamic_axes into dynamic_shapes by wrapping the axis names with ``torch.export.Dim.DYNAMIC``.

    dynamic_axes examples:
    (1) dynamic_axes = {"x": {0: "my_custom_axis_name_1"}, "y": {1: "my_custom_axis_name_2"}}
    (2) dynamic_axes = {"x": [0], "y": [1]}

    these will be converted to dynamic_shapes respectively:
    (1) dynamic_shapes = {"x": {0: Dim.DYNAMIC}, "y": {1: Dim.DYNAMIC}}
    (2) dynamic_shapes = {"x": {0: Dim.DYNAMIC}, "y": {1: Dim.DYNAMIC}}

    Detail on Dim.DYNAMIC: `#133620 <https://github.com/pytorch/pytorch/pull/133620>`_
    '''
def from_dynamic_shapes_to_dynamic_axes(dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any], input_names: Sequence[str], exception: Exception) -> dict[str, Any] | None:
    '''
    Converts dynamic_shapes into dynamic_axes by removing torch.export.Dim wrapping
    and converting to list or dict form based on whether dimension names are present.

    dynamic_shapes examples:
    (1) dynamic_shapes = {"x": {0: Dim("my_custom_axis_name_1")}, "y": {1: Dim("my_custom_axis_name_2")}}
    (2) dynamic_shapes = ({0: Dim("my_custom_axis_name_1"}, {1: Dim("my_custom_axis_name_2")})

    these will be converted to dynamic_axes respectively:
    (1) dynamic_axes = {"x": [0], "y": [1]}
    (2) dynamic_axes = {"x": [0], "y": [1]}

    NOTE: If the model input is nested, so is the dynamic_shapes, we need to flatten the dynamic_shapes,
    and then assign the axes to the input names in the order they are provided.

    NOTE: input_names are used to assign the axes to the correct input names. If the input names are not
    provided, or less than the dynamic inputs/axes, it raises an error.
    '''
def _any_str_or_dim_in_dynamic_shapes(dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any]) -> bool:
    """Check if there is any string or Dim in the dynamic_shapes."""
def convert_str_to_export_dim(dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None) -> tuple[dict[str, Any] | tuple[Any, ...] | list[Any] | None, bool]: ...
def create_rename_mapping(inputs, dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any]) -> dict[str, str]:
    """Create a mapping from old names to new names for dynamic axes."""
def _get_custom_axis_name(axis: Dim | str) -> str:
    """Get the custom axis name from a torch.export.Dim."""
def _unflatten_dynamic_shapes_with_inputs_tree(inputs: list[Any], dynamic_shapes: dict[str, Any]) -> dict[str, Any | None]: ...
def _flatten_dynamic_shapes_to_axes(dynamic_shapes: dict[str, Any | None] | tuple[Any, ...] | list[Any]) -> tuple[list[Any], _pytree.TreeSpec]: ...
def _signature(model) -> inspect.Signature: ...
