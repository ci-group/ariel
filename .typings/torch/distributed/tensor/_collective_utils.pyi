import torch
import torch.distributed.tensor._dtensor_spec as dtensor_spec
from _typeshed import Incomplete
from dataclasses import dataclass
from torch._C._distributed_c10d import _resolve_process_group as _resolve_process_group
from torch._logging import warning_once as warning_once
from torch.distributed.device_mesh import DeviceMesh as DeviceMesh, _mesh_resources as _mesh_resources
from torch.distributed.distributed_c10d import ProcessGroup as ProcessGroup, Work as Work, _get_group_size_by_name as _get_group_size_by_name, broadcast as broadcast, get_group_rank as get_group_rank, get_rank as get_rank, scatter as scatter

logger: Incomplete

def _shard_dim_alltoall_meta(input, gather_dim, shard_dim, group_name): ...
def shard_dim_alltoall(input, gather_dim, shard_dim, mesh, mesh_dim): ...
def mesh_scatter(output: torch.Tensor, scatter_list: list[torch.Tensor], mesh: DeviceMesh, mesh_dim: int = 0, async_op: bool = False, *, group_src: int = 0) -> Work | None:
    """
    scatter a list of tensors to a device mesh dimension. We by default
    use the first rank of the mesh dimension as the source of truth, i.e
    for a 2d mesh [[0, 1], [2, 3]], if we scatter on mesh_dim = 1, we will
    scatter the tensor list on rank 0 to rank 0/1, and tensor list on rank
    2 to rank 2/3.

    Args:
        output (torch.Tensor): the tensor to receive the scattered list.
        scatter_list (List[torch.Tensor]): the tensor list to be scattered.
        mesh_dim (int, optional): indicate which mesh dimension we want
            to scatter on, we by default choose the first rank on the
            mesh dimension as source of truth.

    Keyword args:
        group_src (int, optional): the group rank of the source data for the
        logical/global tensor, on the specific mesh dimension. By default, we
        use ``group_rank=0`` on each DeviceMesh dimension as the source data
        to preserve the single-device semantic. If passing ``None`` explicitly,
        this method simply uses its local data with no communication.

    Returns:
        A :class:`Work` object
    """
def mesh_broadcast(tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int = 0, async_op: bool = False, *, group_src: int = 0) -> Work | None:
    """
    broadcast the tensor to a device mesh dimension. We by default
    use the first rank of the mesh dimension as the source of truth, i.e
    for a 2d mesh [[0, 1], [2, 3]], if we broadcast on mesh_dim = 1, we will
    broadcast the tensor on rank 0 to rank 0/1, and tensor on rank 2
    to rank 2/3.

    Args:
        tensor (torch.Tensor): tensor to broadcast.
        mesh_dim (int, optional): indicate which mesh dimension we want
            to scatter on, we by default choose the first rank on the
            mesh dimension as source of truth.

    Keyword args:
        group_src (int, optional): the group rank of the source data for the
        logical/global tensor, on the specific mesh dimension. By default, we
        use ``group_rank=0`` on each DeviceMesh dimension as the source data
        to preserve the single-device semantic. If passing ``None`` explicitly,
        this method simply uses its local data with no communication.

    Returns:
        A :class:`Work` object
    """
def pad_tensor(tensor: torch.Tensor, pad_dim: int, pad_size: int) -> torch.Tensor: ...
def unpad_tensor(tensor: torch.Tensor, pad_dim: int, pad_size: int) -> torch.Tensor: ...
def fill_empty_tensor_to_shards(shards: list[torch.Tensor], shard_dim: int, num_empty_tensors: int) -> list[torch.Tensor]: ...
def check_tensor_meta(local_tensor, check_shape_stride: bool = False) -> dtensor_spec.TensorMeta | None: ...
def spec_to_bytes(spec: dtensor_spec.DTensorSpec) -> int: ...

@dataclass
class MeshTopoInfo:
    """
    Mesh information for collective cost estimation
    """
    mesh: DeviceMesh
    mesh_dim_devices: list[int]
    mesh_dim_bandwidth: list[float]
    mesh_dim_latency: list[float]
    @staticmethod
    def build_from_mesh(mesh: DeviceMesh) -> MeshTopoInfo: ...

def allgather_cost(bytes_gb: float, mesh_topo: MeshTopoInfo, mesh_dim: int) -> float: ...
def allreduce_cost(bytes_gb: float, mesh_topo: MeshTopoInfo, mesh_dim: int) -> float: ...
def reduce_scatter_cost(bytes_gb: float, mesh_topo: MeshTopoInfo, mesh_dim: int) -> float: ...
def redistribute_cost(current_spec: dtensor_spec.DTensorSpec, target_spec: dtensor_spec.DTensorSpec) -> float:
    """
    This function returns the cost of redistribute from current to target DTensorSpec.

    NOTE:
    1. Only consider communication cost here, since computation costs for redistribute
       are quite trival (i.e. we only need to narrow or simple division)
    2. Only consider redistribute cost on same mesh, cross mesh communication cost is
       not quite needed for operator strategy estimation/selection.
    """
