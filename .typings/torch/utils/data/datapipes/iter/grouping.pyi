from _typeshed import Incomplete
from collections import defaultdict
from collections.abc import Generator, Iterator
from torch.utils.data.datapipes.datapipe import DataChunk, IterDataPipe
from typing import Any, Callable, TypeVar

__all__ = ['BatcherIterDataPipe', 'GrouperIterDataPipe', 'UnBatcherIterDataPipe']

_T_co = TypeVar('_T_co', covariant=True)

class BatcherIterDataPipe(IterDataPipe[DataChunk]):
    """
    Creates mini-batches of data (functional name: ``batch``).

    An outer dimension will be added as ``batch_size`` if ``drop_last`` is set to ``True``, or ``length % batch_size`` for the
    last batch if ``drop_last`` is set to ``False``.

    Args:
        datapipe: Iterable DataPipe being batched
        batch_size: The size of each batch
        drop_last: Option to drop the last batch if it's not full
        wrapper_class: wrapper to apply onto each batch (type ``List``) before yielding,
            defaults to ``DataChunk``

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp = IterableWrapper(range(10))
        >>> dp = dp.batch(batch_size=3, drop_last=True)
        >>> list(dp)
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """
    datapipe: IterDataPipe
    batch_size: int
    drop_last: bool
    wrapper_class: Incomplete
    def __init__(self, datapipe: IterDataPipe, batch_size: int, drop_last: bool = False, wrapper_class: type[DataChunk] = ...) -> None: ...
    def __iter__(self) -> Iterator[DataChunk]: ...
    def __len__(self) -> int: ...

class UnBatcherIterDataPipe(IterDataPipe):
    """
    Undos batching of data (functional name: ``unbatch``).

    In other words, it flattens the data up to the specified level within a batched DataPipe.

    Args:
        datapipe: Iterable DataPipe being un-batched
        unbatch_level: Defaults to ``1`` (only flattening the top level). If set to ``2``,
            it will flatten the top two levels, and ``-1`` will flatten the entire DataPipe.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> source_dp = IterableWrapper([[[0, 1], [2]], [[3, 4], [5]], [[6]]])
        >>> dp1 = source_dp.unbatch()
        >>> list(dp1)
        [[0, 1], [2], [3, 4], [5], [6]]
        >>> dp2 = source_dp.unbatch(unbatch_level=2)
        >>> list(dp2)
        [0, 1, 2, 3, 4, 5, 6]
    """
    datapipe: Incomplete
    unbatch_level: Incomplete
    def __init__(self, datapipe: IterDataPipe, unbatch_level: int = 1) -> None: ...
    def __iter__(self): ...
    def _dive(self, element, unbatch_level) -> Generator[Incomplete, Incomplete]: ...

class GrouperIterDataPipe(IterDataPipe[DataChunk]):
    '''
    Groups data from IterDataPipe by keys from ``group_key_fn``, yielding a ``DataChunk`` with batch size up to ``group_size``.

    (functional name: ``groupby``).

    The samples are read sequentially from the source ``datapipe``, and a batch of samples belonging to the same group
    will be yielded as soon as the size of the batch reaches ``group_size``. When the buffer is full,
    the DataPipe will yield the largest batch with the same key, provided that its size is larger
    than ``guaranteed_group_size``. If its size is smaller, it will be dropped if ``drop_remaining=True``.

    After iterating through the entirety of source ``datapipe``, everything not dropped due to the buffer capacity
    will be yielded from the buffer, even if the group sizes are smaller than ``guaranteed_group_size``.

    Args:
        datapipe: Iterable datapipe to be grouped
        group_key_fn: Function used to generate group key from the data of the source datapipe
        keep_key: Option to yield the matching key along with the items in a tuple,
            resulting in `(key, [items])` otherwise returning [items]
        buffer_size: The size of buffer for ungrouped data
        group_size: The max size of each group, a batch is yielded as soon as it reaches this size
        guaranteed_group_size: The guaranteed minimum group size to be yielded in case the buffer is full
        drop_remaining: Specifies if the group smaller than ``guaranteed_group_size`` will be dropped from buffer
            when the buffer is full

    Example:
        >>> import os
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> def group_fn(file):
        ...     return os.path.basename(file).split(".")[0]
        >>> source_dp = IterableWrapper(["a.png", "b.png", "a.json", "b.json", "a.jpg", "c.json"])
        >>> dp0 = source_dp.groupby(group_key_fn=group_fn)
        >>> list(dp0)
        [[\'a.png\', \'a.json\', \'a.jpg\'], [\'b.png\', \'b.json\'], [\'c.json\']]
        >>> # A group is yielded as soon as its size equals to `group_size`
        >>> dp1 = source_dp.groupby(group_key_fn=group_fn, group_size=2)
        >>> list(dp1)
        [[\'a.png\', \'a.json\'], [\'b.png\', \'b.json\'], [\'a.jpg\'], [\'c.json\']]
        >>> # Scenario where `buffer` is full, and group \'a\' needs to be yielded since its size > `guaranteed_group_size`
        >>> dp2 = source_dp.groupby(group_key_fn=group_fn, buffer_size=3, group_size=3, guaranteed_group_size=2)
        >>> list(dp2)
        [[\'a.png\', \'a.json\'], [\'b.png\', \'b.json\'], [\'a.jpg\'], [\'c.json\']]
    '''
    datapipe: Incomplete
    group_key_fn: Incomplete
    keep_key: Incomplete
    max_buffer_size: Incomplete
    buffer_elements: defaultdict[Any, list]
    curr_buffer_size: int
    group_size: Incomplete
    guaranteed_group_size: Incomplete
    drop_remaining: Incomplete
    wrapper_class: Incomplete
    def __init__(self, datapipe: IterDataPipe[_T_co], group_key_fn: Callable[[_T_co], Any], *, keep_key: bool = False, buffer_size: int = 10000, group_size: int | None = None, guaranteed_group_size: int | None = None, drop_remaining: bool = False) -> None: ...
    def _remove_biggest_key(self): ...
    def __iter__(self): ...
    def reset(self) -> None: ...
    def __getstate__(self): ...
    def __setstate__(self, state) -> None: ...
    def __del__(self) -> None: ...
