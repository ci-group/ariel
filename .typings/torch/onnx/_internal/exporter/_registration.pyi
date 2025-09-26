import dataclasses
import torch._ops
import types
from _typeshed import Incomplete
from torch.onnx._internal._lazy_import import onnxscript as onnxscript, onnxscript_apis as onnxscript_apis
from torch.onnx._internal.exporter import _constants as _constants, _schemas as _schemas
from torch.onnx._internal.exporter._torchlib import _torchlib_registry as _torchlib_registry
from typing import Callable, Literal
from typing_extensions import TypeAlias

TorchOp: TypeAlias = torch._ops.OpOverload | types.BuiltinFunctionType | Callable
logger: Incomplete

@dataclasses.dataclass
class OnnxDecompMeta:
    """A wrapper of onnx-script function with additional metadata.

    onnx_function: The onnx-script function from torchlib.
    fx_target: The PyTorch node callable target.
    signature: The ONNX signature of the function. When None, the signature is inferred.
    is_custom: Whether the function is a custom function.
    is_complex: Whether the function is a function that handles complex valued inputs.
    opset_introduced:
        The ONNX opset version in which the function was introduced.
        Its specifies the minimum ONNX opset version required to use the function.
    device: The device the function is registered to. If None, it is registered to all devices.
    skip_signature_inference: Whether to skip signature inference for the function.
    """
    onnx_function: Callable
    fx_target: TorchOp
    signature: _schemas.OpSignature | None
    is_custom: bool = ...
    is_complex: bool = ...
    opset_introduced: int = ...
    device: Literal['cuda', 'cpu'] | str | None = ...
    skip_signature_inference: bool = ...
    def __post_init__(self) -> None: ...

def _get_overload(qualified_name: str) -> torch._ops.OpOverload | None:
    """Obtain the torch op from <namespace>::<op_name>[.<overload>]"""

class ONNXRegistry:
    """Registry for ONNX functions.

    The registry maintains a mapping from qualified names to symbolic functions under a
    fixed opset version. It supports registering custom onnx-script functions and for
    dispatcher to dispatch calls to the appropriate function.

    """
    _opset_version: Incomplete
    functions: dict[TorchOp | str, list[OnnxDecompMeta]]
    def __init__(self) -> None:
        """Initializes the registry"""
    @property
    def opset_version(self) -> int:
        """The ONNX opset version the exporter should target."""
    @classmethod
    def from_torchlib(cls, opset_version=...) -> ONNXRegistry:
        """Populates the registry with ATen functions from torchlib.

        Args:
            torchlib_registry: The torchlib registry to use for populating the registry.
        """
    def _register(self, target: TorchOp, onnx_decomposition: OnnxDecompMeta) -> None:
        """Registers a OnnxDecompMeta to an operator.

        Args:
            target: The PyTorch node callable target.
            onnx_decomposition: The OnnxDecompMeta to register.
        """
    def register_op(self, target: TorchOp, function: Callable, is_complex: bool = False) -> None:
        """Registers a custom operator: torch.ops.<namespace>.<op_name>.<overload>.

        Args:
            target: The PyTorch node callable target.
            function: The onnx-script function to register.
            is_complex: Whether the function is a function that handles complex valued inputs.
        """
    def get_decomps(self, target: TorchOp) -> list[OnnxDecompMeta]:
        """Returns a list of OnnxDecompMeta for the given op: torch.ops.<namespace>.<op_name>.<overload>.

        The list is ordered by the time of registration. The custom operators should come
        first in the list.

        Args:
            target: The PyTorch node callable target.
        Returns:
            A list of OnnxDecompMeta corresponding to the given name, or None if
            the name is not in the registry.
        """
    def is_registered(self, target: TorchOp) -> bool:
        """Returns whether the given op is registered: torch.ops.<namespace>.<op_name>.<overload>.

        Args:
            target: The PyTorch node callable target.

        Returns:
            True if the given op is registered, otherwise False.
        """
    def _cleanup_registry_based_on_opset_version(self) -> None:
        """Pick the implementation with the highest opset version valid until the current opset version."""
    def __repr__(self) -> str: ...
