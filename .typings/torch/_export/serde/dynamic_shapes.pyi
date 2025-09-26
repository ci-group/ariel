import dataclasses
from .serialize import _dataclass_to_dict as _dataclass_to_dict
from torch._dynamo.exc import UserError as UserError, UserErrorType as UserErrorType
from torch.export.dynamic_shapes import Dim as Dim, _DerivedDim as _DerivedDim, _DimHint as _DimHint, _check_dynamic_shapes as _check_dynamic_shapes, _tree_map_with_path as _tree_map_with_path
from torch.utils._pytree import tree_map as tree_map
from typing import Any

@dataclasses.dataclass
class RootDim:
    """
    This represents a Dim object.
    """
    min: int
    max: int | None
    derived: list[str]

@dataclasses.dataclass
class DynamicShapesSpec:
    """
    This stores a dynamic_shapes spec for de/serialization.
    """
    dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None
    dims: dict[str, RootDim]

def _postprocess_serialized_shapes(dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None, dims: dict[str, dict[str, int | list[str] | None]], to_dict: bool | None = False) -> DynamicShapesSpec | dict[str, Any]:
    """
    Sorts dims and dumps to dictionary format.
    """
def _dump_dynamic_shapes(dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None, args: tuple[Any], kwargs: dict[str, Any] | None = None, to_dict: bool | None = False) -> DynamicShapesSpec | dict[str, Any]:
    '''
    Utility function for dynamic shapes serialization, serializing a dynamic_shapes spec.
    Returns a DynamicShapesSpec dataclass containing 2 fields, "dynamic_shapes" and "dims".
    Uses args & kwargs to distinguish between tensor-level and dim-level specs (only for Nones).

    dynamic_shapes: A pytree structure mirroring the dynamic_shapes input to export():
        - Each tensor input is represented with a list of values, non-tensor inputs with None.
        - dynamic dimensions (i.e. symbols) in tensors and Dim enums are represented with strings.
        - static dimensions are represented with ints.

    dims: A dictionary mapping each symbol name to the min/max range and derived dim names.

    For example:
    ```
    dx = Dim("dx", min=4, max=16)
    dy = dx + 1

    inputs = (
        [
            torch.randn(4, 4),
            torch.randn(5, 4),
        ],
        torch.randn(4),
        torch.randn(4, 4),
        "hello",
    )
    dynamic_shapes = {
        "a": [
            (dx, 4),
            (dy, 4),
        ],
        "b": (Dim.STATIC,),
        "c": None,
        "d": None,
    }
    out = _dump_dynamic_shapes(dynamic_shapes, inputs, to_dict=True)
    ```
    would generate the following output:
    ```
    {
        \'dynamic_shapes\': (
            [
                [\'dx\', 4],
                [\'dx + 1\', 4],
            ],
            [\'_DimHint.STATIC\'],
            [\'_DimHint.STATIC\', \'_DimHint.STATIC\'],
            None,
        ),
        \'dims\': {
            \'dx\': {
                \'min\': 4,
                \'max\': 16,
                \'derived\': [\'dx + 1\'],
            },
        },
    }
    ```
    '''
def _load_dynamic_shapes(spec: DynamicShapesSpec | dict[str, Any], from_dict: bool | None = False) -> dict[str, Any] | tuple[Any] | list[Any] | None:
    """
    Utility function for dynamic shapes serialization.
    Deserializes a DynamicShapesSpec or corresponding dictionary into a dynamic_shapes input to export().
    """
