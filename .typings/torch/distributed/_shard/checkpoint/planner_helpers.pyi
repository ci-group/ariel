from .metadata import ChunkStorageMetadata, TensorStorageMetadata
from .planner import ReadItem

__all__ = ['create_read_items_for_chunk_list']

def create_read_items_for_chunk_list(fqn: str, checkpoint_md: TensorStorageMetadata, local_chunks: list[ChunkStorageMetadata]) -> list[ReadItem]:
    """
    Create a list of ``ReadItem`` based on the checkpoint and local chunks.

    This applies the resharding algorithm and computes the reads needed
    to satisfy ``local_chunks`` with a checkpoint described by ``checkpoint_md``.

    Args:
        fqn (str) : The state_dict FQN to pass to ``ReadItem``.
        checkpoint_md (TensorStorageMetadata): metadata for a given tensor
            from a checkpoint.
        local_chunks (List[ChunkStorageMetadata]): Local chunks that needs to be
            loaded.

    Returns:
        A list of ``ReadItem`` that will satisfy all input chunks.
    """
