from torch.distributed._shard.metadata import ShardMetadata as ShardMetadata

def _check_shard_metadata_pair_overlap(shard1: ShardMetadata, shard2: ShardMetadata):
    """
    Checks if two shards overlap.
    """
def _find_nd_overlapping_shards(shards: list[ShardMetadata], sharded_dims: list[int]) -> tuple[int, int] | None: ...
def _find_1d_overlapping_shards(shards: list[ShardMetadata], dim: int) -> tuple[int, int] | None: ...
def validate_non_overlapping_shards_metadata(shards: list[ShardMetadata]):
    """
    Ensures none of the shards overlap with each other.

    Args:
        shards(List[ShardMetadata]): List of :class:`ShardMetadata` objects representing
            each shard.
    Raises:
        ``ValueError`` if there's overlap in any two shards.
    """
def check_tensor(shards_metadata, tensor_dims) -> None:
    """
    Checks if the shards_metadata is compatible with the provided tensor dims.

    Args:
        shards_metadata(List[ShardMetadata]): List of :class:`ShardMetadata`
            objects representing each shard of the tensor.
        tensor_dims(Sequence of int): Dimensions of tensor to verify
    Raises:
        ``ValueError`` if not compatible.
    """
def get_split_size(dim_size, chunks):
    """
    Computes the split size inline with ``torch.chunk``

    Args:
        dim_size(int): Size of the dimension being chunked.
        chunks(int): Number of chunks to create for ``dim_size``.

    Returns:
        An int indicating the split size to use.
    """
def get_chunked_dim_size(dim_size, split_size, idx):
    """
    Computes the dim size of the chunk for provided ``idx`` given ``dim_size``
    and ``split_size``.

    Args:
        dim_size(int): Size of the dimension being chunked.
        split_size(int): The chunk size for each chunk of ``dim_size``.
        idx(int): The index of chunk whose dim size is being requested.

    Returns:
        An int indicating the dim size of the chunk.
    """
def get_chunk_sharding_params(sharding_dim_size, world_size, spec, rank):
    """
    Generate the start pos and offset length for the current rank for
    chunk sharding.

    Args:
        sharding_dim_size(int): The dimension length which we shard on.
        world_size(int): number of ranks.
        spec (:class:`torch.distributed._shard.sharding_spec.ChunkShardingSpec`):
            sharding spec.
        rank(int): # of cuda process.

    Returns:
        start_pos(int): start position of sharded tensor on the given rank.
        chunk_size(int): chunk size of sharded tensor on the given rank.
    """
