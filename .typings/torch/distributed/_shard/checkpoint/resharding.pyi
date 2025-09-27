from torch.distributed.checkpoint.metadata import ChunkStorageMetadata as ChunkStorageMetadata

__all__: list[str]

def _check_shard_metadata_pair_overlap(shard1: ChunkStorageMetadata, shard2: ChunkStorageMetadata) -> bool:
    """Check if two shards overlap."""
def _shards_get_overlap_region_wrt_saved_tensor(saved_shard: ChunkStorageMetadata, current_shard: ChunkStorageMetadata) -> list[tuple[int, int, int, int]]:
    """
    Return the overlapping region between saved_shard and current_shard.

    There returned list has the same number of elements as the tensor's dimension.
    For each element, we produce a tuple with the following contents:
        (dimension, `saved_shard` offset, `current_shard` offset, length)

    Offsets are relative to each shard.
    """
