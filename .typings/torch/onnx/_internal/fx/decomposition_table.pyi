import torch._ops
from torch.onnx._internal.fx import registration as registration
from typing import Callable

def _create_onnx_supports_op_overload_table(registry) -> set[torch._ops.OperatorBase | Callable]:
    """
    Creates a set of OperatorBase and Callable objects that represent ONNX-supported PyTorch operations.

    Args:
        registry (OnnxRegistry): The ONNX registry for PyTorch.

    Returns:
        A collection of OperatorBase and Callable objects representing ONNX-supported PyTorch operations.
    """
def create_onnx_friendly_decomposition_table(registry) -> dict[torch._ops.OperatorBase, Callable]:
    """
    This function creates a dictionary of op overloads and their decomposition functions
    for ops that do not have ONNX symbolic functions. If an op already has an ONNX symbolic function,
    its decomposition function is excluded from the table. The decomposition table is a subset of PyTorch's
    built-in aten-to-aten decomposition.

    Args:
        registry: The ONNX registry for PyTorch.

    Returns:
        Dict[torch._ops.OperatorBase, Callable]: A dictionary that maps op overloads to their corresponding
        decomposition functions.
    """
