import torch
from dataclasses import dataclass
from torch.distributed.device_mesh import DeviceMesh as DeviceMesh
from torch.distributed.tensor.placement_types import Partial as Partial, Placement as Placement, Replicate as Replicate, Shard as Shard
from typing import Any, NamedTuple

class TensorMeta(NamedTuple):
    shape: torch.Size
    stride: tuple[int, ...]
    dtype: torch.dtype

@dataclass
class DTensorSpec:
    mesh: DeviceMesh
    placements: tuple[Placement, ...]
    tensor_meta: TensorMeta | None = ...
    _hash: int | None = ...
    def __post_init__(self) -> None: ...
    def __setattr__(self, attr: str, value: Any) -> None: ...
    def _hash_impl(self) -> int: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __str__(self) -> str:
        """
        human readable representation of the DTensorSpec
        """
    @property
    def shape(self) -> torch.Size: ...
    @property
    def stride(self) -> tuple[int, ...]: ...
    @property
    def ndim(self) -> int: ...
    @property
    def num_shards(self) -> int: ...
    @property
    def device_mesh(self) -> DeviceMesh: ...
    @property
    def dim_map(self) -> list[int]:
        """
        dim_map is a property we derive from `placements` of
        the distributed tensor. It simply return a list of ints
        where dim_map[i] denotes the sharding mapping to the mesh
        dimension, and len(dim_map) == dist_tensor.ndim
        dim_map[i] = -1: means tensor dim i replicate on mesh
        dim_map[i] = j: means tensor dim i shard on mesh dim j

        For example, we have a dist tensor that have the shape of
        [18, 20, 30], and device_mesh([0, 1, 2, 3]), placements:
        [Shard(1)], the dim_map of this placement would be:
        [-1, 0, -1]. This representation is pretty helpful during
        sharding propagation where we could know exactly each
        tensor dimension is sharded or not.

        Note that if placements contains `_Partial`, we have to
        explicitly deal with it, so that when we create a DTensorSpec
        with dim_map, we could properly record the pending sums.
        """
    @property
    def num_shards_map(self) -> list[int]:
        """
        dim_map is a property we derive from `placements` of
        the distributed tensor. Unlike `dim_map`, `num_shards_map`
        denotes how many shards each tensor dim has. Like `dim_map`:
            len(num_shards_map) == dist_tensor.ndim
            num_shards_map[i] = 1: means tensor dim i is not sharded
            num_shards_map[i] = j: means tensor dim i has j shards in total

        For example, we have a dist tensor of shape [18, 20, 30],
        a device_mesh ([[0, 1, 2, 3], [4, 5, 6, 7]]), and placements
        ([Shard(1), Shard(0)]), the num_shards_map of this distributed tensor
        would be: [4, 2, 1].
        """
    @property
    def sums(self) -> list[int]:
        """
        sums is a property we derive from `placements` of the
        distributed tensor. It simply return a list of ints where
        sums[i] denotes the pending sum (partial) on mesh dim i
        """
    @classmethod
    def from_dim_map(cls, mesh: DeviceMesh, dim_map: list[int], sums: list[int], tensor_meta: TensorMeta | None = None) -> DTensorSpec:
        """
        Construct a DTensorSpec from dim_map list and pending sum.

        Args:
            mesh (class:`DeviceMesh`): device mesh to be used in the DTensorSpec
            dim_map (List[int]): a list of integer that represents sharding on each
                tensor dimension, see `dim_map` property doc for details
            sums (List[int]): a list of integer that represents the dist tensor have
                pending sum on which device mesh dimension.
            tensor meta (TensorMeta): DTensor metadata

        Return:
            a class:`DTensorSpec` object
        """
    def is_replicated(self) -> bool:
        """
        return True if the current DTensorSpec replicates on all mesh dims (devices)
        """
    def is_sharded(self) -> bool:
        """
        return True if the current DTensorSpec is sharded on any mesh dims (devices)
        """
    def shallow_copy_with_tensor_meta(self, tensor_meta: TensorMeta | None) -> DTensorSpec:
        """
        Shallow copy the DTensorSpec with a new tensor_meta.
        """
