import torch
import torch.fx
from _typeshed import Incomplete
from collections.abc import Sequence
from onnxscript import ir
from torch.onnx._internal.exporter import _registration as _registration, _schemas as _schemas
from typing import Any, Callable

logger: Incomplete
_TORCH_DTYPE_TO_ONNX_COMPATIBLE: dict[torch.dtype, ir.DataType]

def _torch_dtype_to_onnx_compatible_dtype(dtype: torch.dtype) -> ir.DataType: ...
def _attribute_type_compatible_with_arg(attr: _schemas.AttributeParameter, value: ir.Value | int | float | bool | Sequence[int] | Sequence[float] | None) -> bool:
    """Check if the attribute type is compatible with the argument."""
def _param_type_compatible_with_arg(param: _schemas.Parameter, value: ir.TypeProtocol | str | int | float | complex | Sequence[int] | Sequence[float] | None, assigned_types: dict[str, ir.TypeProtocol]) -> bool: ...
def _get_type_from_tensor(tensor: torch.Tensor | torch.SymBool | torch.SymInt | torch.SymFloat | Sequence[torch.Tensor]) -> ir.TypeProtocol: ...
def _get_first_tensor_in_node_list(nodes: Sequence[torch.fx.Node | Any]) -> torch.Tensor | None: ...
def _get_named_fx_node_args(node: torch.fx.Node) -> dict[str, torch.fx.node.Argument]: ...
def get_matching_overload(node: torch.fx.Node, overloads: Sequence[_registration.OnnxDecompMeta]) -> tuple[Callable | None, str]:
    """Get the overload that matches the node's arguments.

    Args:
        node: The node to match.
        overloads: The OnnxDecompMeta with overloads and their signatures to match against.

    Returns:
        A tuple containing the matched overload and a string describing the reason for failure or success.
    """
def _arg_has_complex_dtype(arg) -> bool:
    """Check if the node has complex dtype recursively."""
def dispatch(node: torch.fx.Node, registry: _registration.ONNXRegistry) -> tuple[Callable | None, str]:
    """Dispatch a node to an ONNX function based on the node's target and the ONNX registry.

    Args:
        node: The node to dispatch.
        registry: The ONNX registry to use for dispatching.

    Returns:
        A tuple containing the matched ONNX function and a string describing the reason for failure or success.
    """
