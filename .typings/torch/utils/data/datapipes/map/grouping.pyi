from _typeshed import Incomplete
from torch.utils.data.datapipes.datapipe import DataChunk, MapDataPipe
from typing import TypeVar

__all__ = ['BatcherMapDataPipe']

_T = TypeVar('_T')

class BatcherMapDataPipe(MapDataPipe[DataChunk]):
    """
    Create mini-batches of data (functional name: ``batch``).

    An outer dimension will be added as ``batch_size`` if ``drop_last`` is set to ``True``,
    or ``length % batch_size`` for the last batch if ``drop_last`` is set to ``False``.

    Args:
        datapipe: Iterable DataPipe being batched
        batch_size: The size of each batch
        drop_last: Option to drop the last batch if it's not full

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> dp = SequenceWrapper(range(10))
        >>> batch_dp = dp.batch(batch_size=2)
        >>> list(batch_dp)
        [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    """
    datapipe: MapDataPipe
    batch_size: int
    drop_last: bool
    wrapper_class: Incomplete
    def __init__(self, datapipe: MapDataPipe[_T], batch_size: int, drop_last: bool = False, wrapper_class: type[DataChunk] = ...) -> None: ...
    def __getitem__(self, index) -> DataChunk: ...
    def __len__(self) -> int: ...
