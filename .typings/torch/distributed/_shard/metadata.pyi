from dataclasses import dataclass
from torch.distributed.remote_device import _remote_device as _remote_device

@dataclass
class ShardMetadata:
    """
    Represents a shard of the overall Tensor including its
    offsets, lengths and device placement.

    Args:
        shard_offsets(List[int]): Offsets in the original tensor indicating
            the start offsets for this shard. Should have the same rank as
            the original tensor.
        shard_sizes(List[int]): Integers indicating the size of each
            dimension for this shard. Should have the same rank as the
            original tensor.
        placement(:class:`torch.distributed._remote_device`):
            Specifies the placement of this shard.
    """
    __slots__ = ...
    shard_offsets: list[int]
    shard_sizes: list[int]
    placement: _remote_device | None
    def __init__(self, shard_offsets: list[int], shard_sizes: list[int], placement: str | _remote_device | None = None) -> None: ...
    def __hash__(self): ...
