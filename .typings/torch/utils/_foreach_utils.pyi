import torch
from _typeshed import Incomplete
from torch import Tensor as Tensor
from torch.autograd.grad_mode import no_grad as no_grad
from typing_extensions import TypeAlias

def _get_foreach_kernels_supported_devices() -> list[str]:
    """Return the device type list that supports foreach kernels."""
def _get_fused_kernels_supported_devices() -> list[str]:
    """Return the device type list that supports fused kernels in optimizer."""
TensorListList: TypeAlias = list[list[Tensor | None]]
Indices: TypeAlias = list[int]
_foreach_supported_types: Incomplete

def _group_tensors_by_device_and_dtype(tensorlistlist: TensorListList, with_indices: bool = False) -> dict[tuple[torch.device, torch.dtype], tuple[TensorListList, Indices]]: ...
def _device_has_foreach_support(device: torch.device) -> bool: ...
def _has_foreach_support(tensors: list[Tensor], device: torch.device) -> bool: ...
