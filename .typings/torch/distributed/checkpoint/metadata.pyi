import os
import torch
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from torch.distributed.checkpoint.stateful import StatefulT
from typing import Any

__all__ = ['ChunkStorageMetadata', 'TensorStorageMetadata', 'BytesStorageMetadata', 'Metadata', 'MetadataIndex', 'TensorProperties', 'StorageMeta']

@dataclass
class ChunkStorageMetadata:
    """
    Each chunk is expected to have the same properties of the TensorStorageMetadata
    that includes it.
    """
    offsets: torch.Size
    sizes: torch.Size

class _MEM_FORMAT_ENCODING(Enum):
    """Describe the memory format of a tensor."""
    TORCH_CONTIGUOUS_FORMAT = 0
    TORCH_CHANNELS_LAST = 1
    TORCH_PRESERVE_FORMAT = 2

@dataclass
class TensorProperties:
    """Properties used to create :class:`Tensor`"""
    dtype: torch.dtype = field(default_factory=torch.get_default_dtype)
    layout: torch.layout = field(default=torch.strided)
    requires_grad: bool = ...
    memory_format: torch.memory_format = field(default=torch.contiguous_format)
    pin_memory: bool = ...
    def __getstate__(self): ...
    def __setstate__(self, state) -> None: ...
    @staticmethod
    def create_from_tensor(tensor: torch.Tensor) -> TensorProperties: ...

@dataclass
class TensorStorageMetadata:
    properties: TensorProperties
    size: torch.Size
    chunks: list[ChunkStorageMetadata]

@dataclass
class BytesStorageMetadata: ...
STORAGE_TYPES = TensorStorageMetadata | BytesStorageMetadata
STATE_DICT_TYPE = dict[str, StatefulT | Any]

@dataclass
class StorageMeta:
    checkpoint_id: str | os.PathLike | None = ...
    save_id: str | None = ...
    load_id: str | None = ...
    modules: list[str] = field(default_factory=list)

@dataclass
class Metadata:
    """This class represents the metadata of the checkpoint."""
    state_dict_metadata: dict[str, STORAGE_TYPES]
    planner_data: Any = ...
    storage_data: Any = ...
    storage_meta: StorageMeta | None = ...

@dataclass(frozen=True)
class MetadataIndex:
    """This class represents a lookup key for items in a state dict or Metadata."""
    fqn: str
    offset: torch.Size | None = ...
    index: int | None = field(hash=False, compare=False, default=None)
    def __init__(self, fqn: str, offset: Sequence[int] | None = None, index: int | None = None) -> None: ...
