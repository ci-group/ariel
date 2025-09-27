import torch
from collections.abc import Sequence
from typing import Any, TypeVar, overload

__all__ = ['scatter', 'scatter_kwargs', 'gather']

T = TypeVar('T', dict, list, tuple)

@overload
def scatter(inputs: torch.Tensor, target_gpus: Sequence[int | torch.device], dim: int = ...) -> tuple[torch.Tensor, ...]: ...
@overload
def scatter(inputs: T, target_gpus: Sequence[int | torch.device], dim: int = ...) -> list[T]: ...
def scatter_kwargs(inputs: tuple[Any, ...], kwargs: dict[str, Any] | None, target_gpus: Sequence[int | torch.device], dim: int = 0) -> tuple[tuple[Any, ...], tuple[dict[str, Any], ...]]:
    """Scatter with support for kwargs dictionary."""
def gather(outputs: Any, target_device: int | torch.device, dim: int = 0) -> Any:
    """Gather tensors from different GPUs on a specified device.

    This function is useful for gathering the results of a distributed computation.
    It takes a sequence of objects, one for each GPU, and returns a single object
    on the specified device.

    Args:
        outputs (Any): A sequence of objects (potentially tensors) to gather.
        target_device (Union[int, torch.device]): The device to gather the tensors to.
            Use 'cpu' for CPU to avoid a deprecation warning.
        dim (int, optional): The dimension along which to gather. Default: 0.

    Returns:
        Any: A gathered object (potentially tensor) on the specified device.
    """
