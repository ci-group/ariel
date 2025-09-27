import torch
from _typeshed import Incomplete
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata as ChunkStorageMetadata, MetadataIndex as MetadataIndex, TensorProperties as TensorProperties, TensorStorageMetadata as TensorStorageMetadata
from torch.distributed.checkpoint.planner import TensorWriteData as TensorWriteData, WriteItem as WriteItem, WriteItemType as WriteItemType
from typing import Any

aten: Incomplete

class LocalShardsWrapper(torch.Tensor):
    """
    A wrapper class to hold local shards of a DTensor.
    This class is used largely for checkpointing purposes and implicity subtypes
    the _Checkpointable protocol.
    """
    __slots__: Incomplete
    _local_shards: list[torch.Tensor]
    _storage_meta: TensorStorageMetadata
    @staticmethod
    def __new__(cls, local_shards: list[torch.Tensor], local_offsets: list[tuple[int, ...]]) -> LocalShardsWrapper: ...
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None): ...
    @staticmethod
    def handle_all_gather_into_tensor(args, kwargs) -> torch.Tensor: ...
    @staticmethod
    def handle_wait_tensor(args, kwargs) -> torch.Tensor: ...
    @staticmethod
    def handle_to_copy(args, kwargs) -> torch.Tensor: ...
    @staticmethod
    def handle_view(args, kwargs) -> LocalShardsWrapper: ...
    @staticmethod
    def handle_equal(args, kwargs) -> bool:
        """
        LocalShardsWrapper equal impl also checks for equality of storage metadata
        and the order of shards
        """
    @staticmethod
    def handle_detach(args, kwargs) -> LocalShardsWrapper: ...
    @staticmethod
    def handle_clone(args, kwargs) -> LocalShardsWrapper: ...
    @staticmethod
    def handle_new_empty(args, kwargs) -> LocalShardsWrapper: ...
    @property
    def device(self) -> torch._C.device: ...
    @property
    def is_meta(self) -> bool: ...
    def is_pinned(self) -> bool: ...
    def requires_grad_(self, requires_grad: bool = True) -> LocalShardsWrapper: ...
    def local_shards(self) -> list[torch.Tensor]:
        """
        Returns a list of :class:`torch.Tensor' corresponding to the
        local shards for this rank. Returns an empty list if the current rank
        does not host any shards for this Tensor.
        """
    def local_sizes(self) -> list[torch.Size]:
        """
        Returns a list of :class:`torch.Size' corresponding to the
        local sizes for the shards on this rank. Returns an empty list if the current rank
        does not host any shards for this Tensor.
        """
    def local_offsets(self) -> list[torch.Size]:
        """
        Returns a list of :class:`torch.Size' corresponding to the
        local offsets for the shards on this rank. Returns an empty list if the current rank
        does not host any shards for this Tensor.
        """
    @property
    def local_chunks(self) -> list[ChunkStorageMetadata]:
        """
        Returns a :class:`list[ChunkStorageMetadata]` object corresponding to the
        metadata for each tensor shard
        """
    def storage_metadata(self) -> TensorStorageMetadata:
        """
        Returns a :class:`TensorStorageMetadata` object corresponding to the
        metadata for the local tensor on current rank
        """
    def is_empty_shard(self) -> bool:
        """
        Returns a :class:`bool` object indicating if the local tensor on current rank
        is an empty tensor
        """
    def __create_write_items__(self, fqn: str, object: Any) -> list[WriteItem]:
        """
        For compatibility with DCP, we support creation of WriteItems
        such that they can be saved properly.
        """
    def __create_chunk_list__(self) -> list[ChunkStorageMetadata]:
        """
        For compatibility with DCP, we support creation of chunk lists
        such that they can be saved properly.
        """
    def __get_tensor_shard__(self, index: MetadataIndex) -> torch.Tensor:
        """
        For compatibility with DCP, we support finding shard based on index
        Return a 'torch.Tensor' shard based on 'MetadataIndex'.
        """
    def _get_tensor_size_bytes(self) -> int: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
