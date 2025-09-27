import dataclasses
import torch
from collections.abc import Sequence
from torch.onnx.ops import _dtype_mappings as _dtype_mappings

_INT_TYPE: str
_FLOAT_TYPE: str
_STRING_TYPE: str
_INT_SEQ_TYPE: str
_FLOAT_SEQ_TYPE: str
_STRING_SEQ_TYPE: str

@dataclasses.dataclass
class EncodedAttrs:
    '''Class to encode attributes from dictionary into lists of FX compatible attributes.

    Since FX does not support dictionaries, we need to encode the attributes into
    lists. This class provides a way to encode and decode the attributes.

    Attributes:
        attr_keys: List of attribute keys.
        attr_types: List of attribute types. Values can be "i" (int), "f" (float),
            "s" (string), "is" (int sequence), "fs" (float sequence), or "ss" (string sequence).
        attr_pos: List of tuples representing the start and end positions of each
            attribute in the corresponding list.
        attr_ints: List of integer attributes.
        attr_floats: List of float attributes.
        attr_strs: List of string attributes.
    '''
    attr_keys: list[str]
    attr_types: list[str]
    attr_pos: list[tuple[int, int]]
    attr_ints: list[int]
    attr_floats: list[float]
    attr_strs: list[str]
    @classmethod
    def from_dict(cls, attrs: dict[str, int | float | str | bool | Sequence[int] | Sequence[float] | Sequence[str] | Sequence[bool]]) -> EncodedAttrs: ...
    def to_dict(self) -> dict[str, int | float | str | list[int] | list[float] | list[str]]:
        """Convert the encoded attributes back to a dictionary for creating an ONNX node."""

def _symbolic(inputs: Sequence[torch.Tensor | None], op_type: str, onnx_dtype: int, *, shape: Sequence[int | torch.SymInt], attr_keys: Sequence[str], attr_types: Sequence[str], attr_pos: Sequence[tuple[int, int]], attr_ints: Sequence[int], attr_floats: Sequence[float], attr_strs: Sequence[str], metadata_props_keys: Sequence[str] = (), metadata_props_values: Sequence[str] = (), domain: str = '', version: int | None = None) -> torch.Tensor: ...
@_symbolic.register_fake
def _(inputs: Sequence[torch.Tensor], op_type: str, onnx_dtype: int, *, shape: Sequence[int | torch.SymInt], attr_keys: Sequence[str], attr_types: Sequence[str], attr_pos: Sequence[tuple[int, int]], attr_ints: Sequence[int], attr_floats: Sequence[float], attr_strs: Sequence[str], metadata_props_keys: Sequence[str] = (), metadata_props_values: Sequence[str] = (), domain: str = '', version: int | None = None) -> torch.Tensor: ...
def _symbolic_multi_out(inputs: Sequence[torch.Tensor | None], op_type: str, onnx_dtypes: Sequence[int], *, shapes: Sequence[Sequence[int | torch.SymInt]], attr_keys: Sequence[str], attr_types: Sequence[str], attr_pos: Sequence[tuple[int, int]], attr_ints: Sequence[int], attr_floats: Sequence[float], attr_strs: Sequence[str], metadata_props_keys: Sequence[str] = (), metadata_props_values: Sequence[str] = (), domain: str = '', version: int | None = None) -> list[torch.Tensor]: ...
