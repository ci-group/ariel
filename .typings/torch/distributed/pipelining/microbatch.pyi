from _typeshed import Incomplete
from typing import Any

__all__ = ['TensorChunkSpec', 'split_args_kwargs_into_chunks', 'merge_chunks']

class _CustomReducer:
    """
    Custom reducer class that can be used to specify a custom operation that
    reduces losses of multiple microbatches into one value.

    Example:
    >>> # xdoctest: +SKIP
    >>> sum_reducer = _CustomReducer(
    >>>     torch.tensor(0.0),
    >>>     lambda a, b: a + b
    >>> )
    """
    init_value: Incomplete
    reduce_fn: Incomplete
    def __init__(self, init_value, reduce_fn) -> None: ...

class _LossReducer(_CustomReducer): ...

class TensorChunkSpec:
    """
    Class used to specify chunking of inputs
    """
    split_dim: Incomplete
    def __init__(self, split_dim) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    @staticmethod
    def from_tuple(chunk_dims: tuple[int, ...]):
        """
        A helper for creating a tuple of `TensorChunkSpec` from a tuple of chunk
        dimensions (int's).
        Example:
            >>> # xdoctest: +SKIP
            >>> # There are three positional arguments to the model, and
            >>> # we are chunking them along dimension 0, 0 and 1, respectively
            >>> args_chunk_spec = TensorChunkSpec.from_tuple((0, 0, 1))
        """
    @staticmethod
    def from_dict(chunk_dims: dict[str, int]):
        '''
        A helper for creating a dictionary of `TensorChunkSpec` from a
        dictionary of chunk dimensions (int\'s).
        Example:
            >>> # xdoctest: +SKIP
            >>> # Chunk dimension 0 for the "id" argument, 1 for the "mask" argument
            >>> kwargs_chunk_spec = TensorChunkSpec.from_dict({"id": 0, "mask": 1})
        '''

class _Replicate: ...

def split_args_kwargs_into_chunks(args: tuple[Any, ...], kwargs: dict[str, Any] | None, chunks: int, args_chunk_spec: tuple[TensorChunkSpec, ...] | None = None, kwargs_chunk_spec: dict[str, TensorChunkSpec] | None = None) -> tuple[list[tuple], list[dict]]:
    """
    Given a sequence of args and kwargs, split them into a number of chunks
    according to  their respective chunking specs.

    Args:
        args: Tuple of args
        kwargs: Dict of kwargs
        chunks: Number of chunks to split the args and kwargs into
        args_chunk_spec: chunking specs for args, in same shape as args
        kwargs_chunk_spec: chunking specs for kwargs, in same shape as kwargs

    Returns:
        args_split: List of sharded args
        kwargs_split: List of sharded kwargs
    """
def merge_chunks(chunks: list[Any], chunk_spec):
    """
    Given a list of chunks, merge them into a single value according to
    the chunk spec.

    Args:
        chunks: list of chunks
        chunk_spec: Chunking spec for the chunks

    Returns:
        value: Merged value
    """
