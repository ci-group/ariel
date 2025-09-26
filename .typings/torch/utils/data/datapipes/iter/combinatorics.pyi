import random
from _typeshed import Incomplete
from collections.abc import Iterator
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.sampler import Sampler
from typing import TypeVar

__all__ = ['SamplerIterDataPipe', 'ShufflerIterDataPipe']

_T_co = TypeVar('_T_co', covariant=True)

class SamplerIterDataPipe(IterDataPipe[_T_co]):
    """
    Generate sample elements using the provided ``Sampler`` (defaults to :class:`SequentialSampler`).

    Args:
        datapipe: IterDataPipe to sample from
        sampler: Sampler class to generate sample elements from input DataPipe.
            Default is :class:`SequentialSampler` for IterDataPipe
    """
    datapipe: IterDataPipe
    sampler: Sampler
    sampler_args: Incomplete
    sampler_kwargs: Incomplete
    def __init__(self, datapipe: IterDataPipe, sampler: type[Sampler] = ..., sampler_args: tuple | None = None, sampler_kwargs: dict | None = None) -> None: ...
    def __iter__(self) -> Iterator[_T_co]: ...
    def __len__(self) -> int: ...

class ShufflerIterDataPipe(IterDataPipe[_T_co]):
    """
    Shuffle the input DataPipe with a buffer (functional name: ``shuffle``).

    The buffer with ``buffer_size`` is filled with elements from the datapipe first. Then,
    each item will be yielded from the buffer by reservoir sampling via iterator.

    ``buffer_size`` is required to be larger than ``0``. For ``buffer_size == 1``, the
    datapipe is not shuffled. In order to fully shuffle all elements from datapipe,
    ``buffer_size`` is required to be greater than or equal to the size of datapipe.

    When it is used with :class:`torch.utils.data.DataLoader`, the methods to
    set up random seed are different based on :attr:`num_workers`.

    For single-process mode (:attr:`num_workers == 0`), the random seed is set before
    the :class:`~torch.utils.data.DataLoader` in the main process. For multi-process
    mode (:attr:`num_worker > 0`), `worker_init_fn` is used to set up a random seed
    for each worker process.

    Args:
        datapipe: The IterDataPipe being shuffled
        buffer_size: The buffer size for shuffling (default to ``10000``)
        unbatch_level: Specifies if it is necessary to unbatch source data before
            applying the shuffle

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp = IterableWrapper(range(10))
        >>> shuffle_dp = dp.shuffle()
        >>> list(shuffle_dp)
        [0, 4, 1, 6, 3, 2, 9, 5, 7, 8]
    """
    datapipe: IterDataPipe[_T_co]
    buffer_size: int
    _buffer: list[_T_co]
    _enabled: bool
    _seed: int | None
    _rng: random.Random
    def __init__(self, datapipe: IterDataPipe[_T_co], *, buffer_size: int = 10000, unbatch_level: int = 0) -> None: ...
    def set_shuffle(self, shuffle: bool = True): ...
    def set_seed(self, seed: int): ...
    def __iter__(self) -> Iterator[_T_co]: ...
    def __len__(self) -> int: ...
    def reset(self) -> None: ...
    def __getstate__(self): ...
    def __setstate__(self, state) -> None: ...
    def __del__(self) -> None: ...
