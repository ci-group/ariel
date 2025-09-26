import random
from _typeshed import Incomplete
from collections.abc import Iterator
from torch.utils.data.datapipes.datapipe import IterDataPipe, MapDataPipe
from typing import TypeVar

__all__ = ['ShufflerIterDataPipe']

_T_co = TypeVar('_T_co', covariant=True)

class ShufflerIterDataPipe(IterDataPipe[_T_co]):
    """
    Shuffle the input MapDataPipe via its indices (functional name: ``shuffle``).

    When it is used with :class:`~torch.utils.data.DataLoader`, the methods to
    set up random seed are different based on :attr:`num_workers`.

    For single-process mode (:attr:`num_workers == 0`), the random seed is set before
    the :class:`~torch.utils.data.DataLoader` in the main process. For multi-process
    mode (:attr:`num_worker > 0`), ``worker_init_fn`` is used to set up a random seed
    for each worker process.

    Args:
        datapipe: MapDataPipe being shuffled
        indices: a list of indices of the MapDataPipe. If not provided, we assume it uses 0-based indexing

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> dp = SequenceWrapper(range(10))
        >>> shuffle_dp = dp.shuffle().set_seed(0)
        >>> list(shuffle_dp)
        [7, 8, 1, 5, 3, 4, 2, 0, 9, 6]
        >>> list(shuffle_dp)
        [6, 1, 9, 5, 2, 4, 7, 3, 8, 0]
        >>> # Reset seed for Shuffler
        >>> shuffle_dp = shuffle_dp.set_seed(0)
        >>> list(shuffle_dp)
        [7, 8, 1, 5, 3, 4, 2, 0, 9, 6]

    Note:
        Even thought this ``shuffle`` operation takes a ``MapDataPipe`` as the input, it would return an
        ``IterDataPipe`` rather than a ``MapDataPipe``, because ``MapDataPipe`` should be non-sensitive to
        the order of data order for the sake of random reads, but ``IterDataPipe`` depends on the order
        of data during data-processing.
    """
    datapipe: MapDataPipe[_T_co]
    _enabled: bool
    _seed: int | None
    _rng: random.Random
    indices: Incomplete
    _shuffled_indices: list
    def __init__(self, datapipe: MapDataPipe[_T_co], *, indices: list | None = None) -> None: ...
    def set_shuffle(self, shuffle: bool = True): ...
    def set_seed(self, seed: int): ...
    def __iter__(self) -> Iterator[_T_co]: ...
    def reset(self) -> None: ...
    def __len__(self) -> int: ...
    def __getstate__(self): ...
    def __setstate__(self, state) -> None: ...
