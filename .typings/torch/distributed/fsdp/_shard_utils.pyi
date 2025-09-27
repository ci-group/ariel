import torch
import torch.distributed as dist
from torch._utils import _get_device_module as _get_device_module
from torch.distributed import distributed_c10d as distributed_c10d
from torch.distributed._shard.sharded_tensor import Shard as Shard, ShardedTensor as ShardedTensor, ShardedTensorMetadata as ShardedTensorMetadata, TensorProperties as TensorProperties
from torch.distributed._shard.sharding_spec import ShardMetadata as ShardMetadata
from torch.distributed.tensor import DTensor as DTensor, DeviceMesh as DeviceMesh, Replicate as Replicate

def _get_remote_device_str(rank, device_type, num_devices_per_node): ...
def _create_chunk_sharded_tensor(tensor: torch.Tensor, rank: int, world_size: int, num_devices_per_node: int, pg: dist.ProcessGroup, device: torch.device | None = None) -> ShardedTensor:
    """
    Shard a tensor to chunks along the first dimension. The local rank will gets its
    corresponding chunk as the local shard to create a ShardedTensor.
    """
def _create_chunk_dtensor(tensor: torch.Tensor, rank: int, device_mesh: DeviceMesh) -> DTensor:
    """
    Shard a tensor to chunks along the first dimension. The local rank will gets its
    corresponding chunk as the local tensor to create a DTensor.
    """
def _all_gather_dtensor(tensor: DTensor, root_mesh: DeviceMesh | None) -> torch.Tensor:
    """
    All gather a DTensor in its sharded dimension and return the local tensor.
    """
