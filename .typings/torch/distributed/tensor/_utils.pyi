import torch
from collections.abc import Sequence
from torch._prims_common import ShapeType as ShapeType
from torch.distributed.device_mesh import DeviceMesh as DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec as DTensorSpec
from torch.distributed.tensor.placement_types import Partial as Partial, Placement as Placement, Replicate as Replicate, Shard as Shard, _StridedShard as _StridedShard

def _explicit_order_placements(mesh_shape: ShapeType, placements: Sequence[Placement]) -> Sequence[tuple[int, Placement]]:
    """
    Replace Strided Shards with regular shards in an adjusted order.

    Returns a list of (mesh_dim, placement) tuples where the list order is the sharding order.

    ex.
    [Shard(0), _StridedShard(0, split_factor=2), Shard(0)] ->
    [(0, Shard(0)), (2, Shard(0)), (1, Shard(0))]

    """
def compute_local_shape_and_global_offset(global_shape: ShapeType, mesh: DeviceMesh, placements: Sequence[Placement]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Compute the local tensor shape and the global offsets into the original tensor
    of a DTensor on its current global rank. This is useful for checkpointing purpose.

    Example:
    global_tensor = [[0,  1,  2,  3,  4], sharded on mesh (DP=2, TP=2) with (Shard(1), Shard(1))
                     [10, 11, 12, 13, 14]]

    This table shows the return value of local_shape and global_offset for each rank.
    (`local_tensor` is for illustration only).

    Note how the first coordinate of global_offset is always 0, corresponding to tensor dim 0 being replicated.

    Rank        local_tensor        local_shape     global_offset
    -------------------------------------------------------------
    0           [[0, 1],            (2, 2)          (0, 0)
                 [10, 11]]

    1           [[2],               (2, 1)          (0, 2)
                 [12]]

    2           [[3],               (2, 1)          (0, 3)
                 [13]]

    3           [[4],               (2, 1)          (0, 4)
                 [14]]

    Args:
        global_shape (ShapeType): The global shape of the DTensor.
        mesh (:class:`DeviceMesh`): The device mesh this DTensor is distributed on.
        placements (Sequence[:class:`Placement`]]): The placements of the DTensor.

    Return:
        local_shape: the shape of the DTensor's _local_tensor on the current rank.
        global_offset: a tuple of offsets for each dimension of the global tensor shape,
        identifying how this shard fits into the global tensor in each dimension.

    """
def _compute_local_shape_and_global_offset(global_shape: ShapeType, mesh_shape: ShapeType, my_coordinate: list[int] | None, placements: Sequence[Placement]) -> tuple[tuple[int, ...], tuple[int, ...]]: ...
def compute_global_tensor_info(tensor: torch.Tensor, mesh: DeviceMesh, placements: Sequence[Placement]) -> tuple[list[int], list[int]]:
    """
    Compute the global size and stride of a DTensor from the given local tensor.
    The local size is multiplited by `world_size` per Sharding dim.
    The local stride is multiplited by `world_size` per Sharding dim, as long as the
    dimension is outside sharding dim.

    For example, if we have a local tensor with size (4, 8, 2) and stride (16, 1, 8).
    If the DTensor placements are [Shard(2)] and world_size is 2;
    then the global size is (4, 8, 4) and stride is (16 * 2, 1, 8).

    Args:
        tensor (:class:`torch.Tensor`):
            Local tensor which DTensor will be constructed from.
        mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology
            of devices for the DTensor.
        placements (Sequence[:class:`Placement`]]):
            The attribute of the DTensor that describes its layout
            on the mesh topology.

    Return:
        tensor_shape: A List of int which specifies the size of DTensor which build
            on top of the local tensor.
        tensor_stride: A List of int which specifies the stride of DTensor.
    """
def compute_global_tensor_shape(shape: torch.Size, mesh: DeviceMesh, placements: Sequence[Placement]) -> torch.Size:
    """
    Compute the global size of a DTensor from the given local tensor shape,
    the mesh and placements. Different from `compute_global_tensor_info`,
    which assumes sharding is even, this util allgathers local shards' shapes
    from all ranks and thus can support uneven sharding.
    NOTE: Currently this function only supports 1D mesh.

    Args:
        shape (:class:`torch.Size`):
            Shape of the local tensor
        mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology
            of devices for the DTensor.
        placements (Sequence[:class:`Placement`]]):
            The attribute of the DTensor that describes its layout
            on the mesh topology.

    Return:
        tensor_shape: Shape of the global DTensor.
    """
def try_find_mesh_from_args(op_call: torch._ops.OpOverload, args: Sequence[object]) -> DeviceMesh:
    """
    Find the device mesh object from args.
    It returns None if no mesh is found.
    NOTE: we can optimize this search if needed
    """
def compute_local_stride(global_stride: ShapeType, mesh: DeviceMesh, placements: Sequence[Placement]) -> tuple[int, ...]:
    """
    Compute the stride of a local tensor shard, given the global stride of the DTensor.
    NOTE: Currently this function is assuming the DTensor is evenly shardable.
    """
def normalize_to_torch_size(size) -> torch.Size:
    """
    Unify variable types of size argument to torch.Size
    Acceptable types include:
        int, Sequence[int], Tuple[int], Tuple[Sequence[int]],
        or torch.Size
    """
