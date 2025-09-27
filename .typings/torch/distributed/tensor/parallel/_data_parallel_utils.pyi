import torch
from torch.distributed._functional_collectives import AsyncCollectiveTensor as AsyncCollectiveTensor
from torch.distributed.tensor import DTensor as DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec as DTensorSpec
from typing import no_type_check

@no_type_check
def sync_grad_hook(grad, *, device_handle=None, compute_stream=None): ...
def _flatten_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, DTensorSpec | None]: ...
@no_type_check
def _unflatten_tensor(tensor, spec, *, device_handle=None, compute_stream=None): ...
