from torch.onnx._internal.exporter import _registration as _registration
from typing import Callable

def get_onnx_implemented_overloads(registry: _registration.ONNXRegistry) -> list[_registration.TorchOp]:
    """
    Creates a set of OperatorBase and Callable objects that represent ONNX-supported PyTorch operations.

    Args:
        registry: The ONNX registry for PyTorch.

    Returns:
        A collection of OperatorBase and Callable objects representing ONNX-supported PyTorch operations.
    """
def create_onnx_friendly_decomposition_table(onnx_registered_ops: set[_registration.TorchOp]) -> dict[_registration.TorchOp, Callable]:
    """
    This function creates a dictionary of op overloads and their decomposition functions
    for ops that do not have ONNX symbolic functions. If an op already has an ONNX symbolic function,
    its decomposition function is excluded from the table. The decomposition table is a subset of PyTorch's
    built-in aten-to-aten decomposition.

    Args:
        onnx_registered_ops: All ops that have an ONNX decomposition implemented.

    Returns:
        Dict[torch._ops.OperatorBase, Callable]: A dictionary that maps op overloads to their corresponding
        decomposition functions.
    """
