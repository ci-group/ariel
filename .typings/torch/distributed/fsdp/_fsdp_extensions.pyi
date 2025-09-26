import abc
import torch
import torch.distributed as dist
from abc import ABC, abstractmethod
from torch.distributed._shard.sharded_tensor.api import ShardedTensor as ShardedTensor
from torch.distributed._shard.sharded_tensor.shard import Shard as Shard
from torch.distributed.fsdp._shard_utils import _all_gather_dtensor as _all_gather_dtensor, _create_chunk_dtensor as _create_chunk_dtensor, _create_chunk_sharded_tensor as _create_chunk_sharded_tensor
from torch.distributed.tensor import DTensor as DTensor, DeviceMesh as DeviceMesh
from typing import Any

class FSDPExtensions(ABC, metaclass=abc.ABCMeta):
    """
    This enables some customizable hooks to enable composability with tensor
    parallelism. To activate these hooks, use :func:`_set_fsdp_extensions` to
    set a custom :class:`FSDPExtensions` that implements the hooks.
    """
    @abstractmethod
    def pre_flatten_transform(self, tensor: torch.Tensor) -> tuple[torch.Tensor, Any | None]:
        """E.g. converting ``DistributedTensor`` to local tensor."""
    @abstractmethod
    def post_unflatten_transform(self, tensor: torch.Tensor, param_extension: Any) -> torch.Tensor:
        """E.g. converting local tensor to ``DistributedTensor``."""
    @abstractmethod
    def chunk_tensor(self, tensor: torch.Tensor, rank: int, world_size: int, num_devices_per_node: int, pg: dist.ProcessGroup, device: torch.device | None = None) -> torch.Tensor:
        """Shards a tensor to chunks and returns the local chunk."""
    @abstractmethod
    def chunk_dtensor(self, tensor: torch.Tensor, rank: int, device_mesh: DeviceMesh) -> torch.Tensor:
        """Shards a tensor/DTensor to DTensor and returns the local DTensor."""
    @abstractmethod
    def pre_load_state_dict_transform(self, tensor: torch.Tensor) -> tuple[torch.Tensor, list[Shard]]:
        """
        This is to be called before loading a *sharded* model state dict and
        should return the tensor and list of shards from which to load data.
        """
    @abstractmethod
    def all_gather_dtensor(self, tensor: DTensor, parent_mesh: DeviceMesh | None) -> torch.Tensor:
        """
        This is to be called before loading a *sharded* DTensor state dict.
        This gathers tensor in FSDP dimension and returns local tensor of
        TP DTensor.
        """

_extensions: FSDPExtensions | None

def _set_fsdp_extensions(flattener: FSDPExtensions) -> None: ...
def _ext_pre_flatten_transform(tensor: torch.Tensor, fsdp_extension: FSDPExtensions | None = None) -> tuple[torch.Tensor, Any | None]: ...
def _ext_post_unflatten_transform(tensor: torch.Tensor, param_extension: Any, fsdp_extension: FSDPExtensions | None = None) -> torch.Tensor: ...
def _ext_chunk_tensor(tensor: torch.Tensor, rank: int, world_size: int, num_devices_per_node: int, pg: dist.ProcessGroup, fsdp_extension: FSDPExtensions | None = None) -> torch.Tensor: ...
def _ext_chunk_dtensor(tensor: torch.Tensor, rank: int, device_mesh: DeviceMesh, fsdp_extension: FSDPExtensions | None = None) -> torch.Tensor: ...
def _ext_pre_load_state_dict_transform(tensor: torch.Tensor, fsdp_extension: FSDPExtensions | None = None) -> tuple[torch.Tensor, list[Shard]]: ...
def _ext_all_gather_dtensor(tensor: DTensor, parent_mesh: DeviceMesh | None, fsdp_extension: FSDPExtensions | None = None) -> torch.Tensor: ...
