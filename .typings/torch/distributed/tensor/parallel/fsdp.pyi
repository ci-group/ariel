import torch
import torch.distributed as dist
from _typeshed import Incomplete
from torch.distributed._shard.sharded_tensor import Shard
from torch.distributed.fsdp._fsdp_extensions import FSDPExtensions
from torch.distributed.tensor import DTensor, DeviceMesh
from typing import Any

__all__ = ['DTensorExtensions']

class DTensorExtensions(FSDPExtensions):
    """
    DTensorExtension is the TensorFlattener extension needed for 2D FSDP + TP.

    This is the implementation for FSDPExtensions defined in
    https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_fsdp_extensions.py
    """
    compute_stream: Incomplete
    device_handle: Incomplete
    def __init__(self, device_handle) -> None: ...
    def pre_flatten_transform(self, tensor: torch.Tensor) -> tuple[torch.Tensor, Any | None]: ...
    def post_unflatten_transform(self, tensor: torch.Tensor, param_extension: Any) -> torch.Tensor: ...
    def chunk_tensor(self, tensor: torch.Tensor, rank: int, world_size: int, num_devices_per_node: int, pg: dist.ProcessGroup, device: torch.device | None = None) -> torch.Tensor: ...
    def chunk_dtensor(self, tensor: torch.Tensor, rank: int, device_mesh: DeviceMesh) -> torch.Tensor: ...
    def pre_load_state_dict_transform(self, tensor: torch.Tensor) -> tuple[torch.Tensor, list[Shard]]: ...
    def all_gather_dtensor(self, tensor: DTensor, parent_mesh: DeviceMesh | None) -> torch.Tensor: ...
