import torch
from dataclasses import dataclass, field
from enum import Enum
from torch.distributed._shard.metadata import ShardMetadata as ShardMetadata

class MEM_FORMAT_ENCODING(Enum):
    TORCH_CONTIGUOUS_FORMAT = 0
    TORCH_CHANNELS_LAST = 1
    TORCH_PRESERVE_FORMAT = 2

@dataclass
class TensorProperties:
    """Properties used to create :class:`Tensor`"""
    dtype: torch.dtype = field(default=torch.get_default_dtype())
    layout: torch.layout = field(default=torch.strided)
    requires_grad: bool = ...
    memory_format: torch.memory_format = field(default=torch.contiguous_format)
    pin_memory: bool = ...
    def __getstate__(self): ...
    def __setstate__(self, state) -> None: ...
    @staticmethod
    def create_from_tensor(tensor: torch.Tensor) -> TensorProperties: ...

@dataclass
class ShardedTensorMetadata:
    """
    Represents metadata for :class:`ShardedTensor`
    """
    shards_metadata: list[ShardMetadata] = field(default_factory=list)
    size: torch.Size = field(default=torch.Size([]))
    tensor_properties: TensorProperties = field(default_factory=TensorProperties)
