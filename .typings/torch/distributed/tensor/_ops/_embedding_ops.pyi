import torch
from _typeshed import Incomplete
from dataclasses import dataclass, field
from torch.distributed.device_mesh import DeviceMesh as DeviceMesh
from torch.distributed.tensor._op_schema import OpSchema as OpSchema, OpStrategy as OpStrategy, PlacementList as PlacementList, StrategyType as StrategyType
from torch.distributed.tensor._ops.utils import expand_to_full_mesh_op_strategy as expand_to_full_mesh_op_strategy, register_op_strategy as register_op_strategy
from torch.distributed.tensor.placement_types import Partial as Partial, Placement as Placement, Replicate as Replicate, Shard as Shard

aten: Incomplete

@dataclass
class MaskBuffer:
    data: torch.Tensor | None = ...
    refcount: int = ...
    def materialize_mask(self, mask) -> None: ...
    def release_mask(self) -> None: ...
    def apply_mask(self, tensor) -> None: ...

@dataclass(frozen=True)
class _MaskPartial(Partial):
    """
    A partial mask placement devised for rowwise sharded embedding op, where we need
    to mask and adjust the indices to the local embedding shard, embedding masking
    is a special type of the Partial placement

    NOTE: the lifecycle of this MaskPartial placement follows the corresponding DTensor
    lifecycle, i.e. the indices_mask would only be alive during the lifetime of the DTensor.
    """
    mask_buffer: MaskBuffer = field(default_factory=MaskBuffer)
    offset_shape: torch.Size | None = ...
    offset_dim: int = ...
    def _partition_value(self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int) -> torch.Tensor: ...
    def _reduce_value(self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int) -> torch.Tensor: ...
    def _reduce_shard_value(self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int, shard_spec: Placement) -> torch.Tensor: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str:
        """
        machine readable representation of the MaskPartial placement
        """
    def __str__(self) -> str:
        """
        human readable representation of the MaskPartial placement
        """

def embedding_strategy(op_schema: OpSchema) -> StrategyType:
    """
    This strategy handles embedding op. We have two possible embedding shardings:
    rowwise and colwise
    """
def embedding_dense_backward_strategy(op_schema: OpSchema) -> StrategyType:
    """
    This strategy handles embedding op. We have two possible embedding shardings:
    rowwise and colwise
    """
