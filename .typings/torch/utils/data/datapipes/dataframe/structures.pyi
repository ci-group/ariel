from collections.abc import Iterator
from torch.utils.data.datapipes.datapipe import DataChunk
from typing import Any

__all__ = ['DataChunkDF']

class DataChunkDF(DataChunk):
    """DataChunkDF iterating over individual items inside of DataFrame containers, to access DataFrames user `raw_iterator`."""
    def __iter__(self) -> Iterator[Any]: ...
    def __len__(self) -> int: ...
