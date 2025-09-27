import torch
from typing import Any, Callable

__all__ = ['normalize_source_name', 'module_to_nested_dict', 'track_dynamism_across_examples', 'clone_and_convert_to_meta']

KeyPath = tuple[Any, ...]
NonTensorShapeFn = Callable[[int | float], tuple[Any, ...]]

def normalize_source_name(name: str) -> str: ...
def module_to_nested_dict(module: torch.nn.Module) -> dict[str, Any]:
    """Recursively converts an nn.Module into a nested dictionary with explicit 'parameters' and 'modules' keys."""
def track_dynamism_across_examples(example_inputs: list[Any]) -> dict[Any, Any]:
    """
    This function analyzes a list of example inputs to determine the dynamism of their shapes.
    It tracks whether the dimensions of tensors or non-tensor values change across
    different examples. The function returns a dictionary where each key represents
    a path to a value in the input examples, and the corresponding value is a tuple
    indicating which dimensions are dynamic (i.e., change across examples). This
    helps in understanding how the structure of data varies across different instances.
    """
def clone_and_convert_to_meta(example_input: Any) -> Any:
    """
    This function takes a list of example inputs and for each tensor, clones it and converts it to device=meta.
    For non-tensor values, it keeps the reference. It uses pytree to handle nested structures recursively.
    """
