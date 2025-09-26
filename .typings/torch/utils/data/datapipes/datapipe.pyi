from _typeshed import Incomplete
from collections.abc import Iterable, Iterator
from torch.utils.data.datapipes._hook_iterator import _SnapshotState
from torch.utils.data.datapipes._typing import _DataPipeMeta, _IterDataPipeMeta
from torch.utils.data.dataset import Dataset, IterableDataset
from typing import Callable, TypeVar

__all__ = ['DataChunk', 'DFIterDataPipe', 'IterDataPipe', 'MapDataPipe']

_T = TypeVar('_T')
_T_co = TypeVar('_T_co', covariant=True)

class DataChunk(list[_T]):
    items: Incomplete
    def __init__(self, items: Iterable[_T]) -> None: ...
    def as_str(self, indent: str = '') -> str: ...
    def __iter__(self) -> Iterator[_T]: ...
    def raw_iterator(self) -> Iterator[_T]: ...

class IterDataPipe(IterableDataset[_T_co], metaclass=_IterDataPipeMeta):
    """
    Iterable-style DataPipe.

    All DataPipes that represent an iterable of data samples should subclass this.
    This style of DataPipes is particularly useful when data come from a stream, or
    when the number of samples is too large to fit them all in memory. ``IterDataPipe`` is lazily initialized and its
    elements are computed only when ``next()`` is called on the iterator of an ``IterDataPipe``.

    All subclasses should overwrite :meth:`__iter__`, which would return an
    iterator of samples in this DataPipe. Calling ``__iter__`` of an ``IterDataPipe`` automatically invokes its
    method ``reset()``, which by default performs no operation. When writing a custom ``IterDataPipe``, users should
    override ``reset()`` if necessary. The common usages include resetting buffers, pointers,
    and various state variables within the custom ``IterDataPipe``.

    Note:
        Only `one` iterator can be valid for each ``IterDataPipe`` at a time,
        and the creation a second iterator will invalidate the first one. This constraint is necessary because
        some ``IterDataPipe`` have internal buffers, whose states can become invalid if there are multiple iterators.
        The code example below presents details on how this constraint looks in practice.
        If you have any feedback related to this constraint, please see `GitHub IterDataPipe Single Iterator Issue`_.

    These DataPipes can be invoked in two ways, using the class constructor or applying their
    functional form onto an existing ``IterDataPipe`` (recommended, available to most but not all DataPipes).
    You can chain multiple `IterDataPipe` together to form a pipeline that will perform multiple
    operations in succession.

    .. _GitHub IterDataPipe Single Iterator Issue:
        https://github.com/pytorch/data/issues/45

    Note:
        When a subclass is used with :class:`~torch.utils.data.DataLoader`, each
        item in the DataPipe will be yielded from the :class:`~torch.utils.data.DataLoader`
        iterator. When :attr:`num_workers > 0`, each worker process will have a
        different copy of the DataPipe object, so it is often desired to configure
        each copy independently to avoid having duplicate data returned from the
        workers. :func:`~torch.utils.data.get_worker_info`, when called in a worker
        process, returns information about the worker. It can be used in either the
        dataset's :meth:`__iter__` method or the :class:`~torch.utils.data.DataLoader` 's
        :attr:`worker_init_fn` option to modify each copy's behavior.

    Examples:
        General Usage:
            >>> # xdoctest: +SKIP
            >>> from torchdata.datapipes.iter import IterableWrapper, Mapper
            >>> dp = IterableWrapper(range(10))
            >>> map_dp_1 = Mapper(dp, lambda x: x + 1)  # Using class constructor
            >>> map_dp_2 = dp.map(lambda x: x + 1)  # Using functional form (recommended)
            >>> list(map_dp_1)
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            >>> list(map_dp_2)
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            >>> filter_dp = map_dp_1.filter(lambda x: x % 2 == 0)
            >>> list(filter_dp)
            [2, 4, 6, 8, 10]
        Single Iterator Constraint Example:
            >>> from torchdata.datapipes.iter import IterableWrapper, Mapper
            >>> source_dp = IterableWrapper(range(10))
            >>> it1 = iter(source_dp)
            >>> list(it1)
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            >>> it1 = iter(source_dp)
            >>> it2 = iter(source_dp)  # The creation of a new iterator invalidates `it1`
            >>> next(it2)
            0
            >>> next(it1)  # Further usage of `it1` will raise a `RunTimeError`
    """
    functions: dict[str, Callable]
    reduce_ex_hook: Callable | None
    getstate_hook: Callable | None
    str_hook: Callable | None
    repr_hook: Callable | None
    _valid_iterator_id: int | None
    _number_of_samples_yielded: int
    _snapshot_state: _SnapshotState
    _fast_forward_iterator: Iterator | None
    def __iter__(self) -> Iterator[_T_co]: ...
    def __getattr__(self, attribute_name): ...
    @classmethod
    def register_function(cls, function_name, function) -> None: ...
    @classmethod
    def register_datapipe_as_function(cls, function_name, cls_to_register, enable_df_api_tracing: bool = False): ...
    def __getstate__(self):
        """
        Serialize `lambda` functions when `dill` is available.

        If this doesn't cover your custom DataPipe's use case, consider writing custom methods for
        `__getstate__` and `__setstate__`, or use `pickle.dumps` for serialization.
        """
    def __reduce_ex__(self, *args, **kwargs): ...
    @classmethod
    def set_getstate_hook(cls, hook_fn) -> None: ...
    @classmethod
    def set_reduce_ex_hook(cls, hook_fn) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __dir__(self): ...
    def reset(self) -> None:
        """
        Reset the `IterDataPipe` to the initial state.

        By default, no-op. For subclasses of `IterDataPipe`, depending on their functionalities,
        they may want to override this method with implementations that
        may clear the buffers and reset pointers of the DataPipe.
        The `reset` method is always called when `__iter__` is called as part of `hook_iterator`.
        """

class DFIterDataPipe(IterDataPipe):
    def _is_dfpipe(self): ...

class MapDataPipe(Dataset[_T_co], metaclass=_DataPipeMeta):
    """
    Map-style DataPipe.

    All datasets that represent a map from keys to data samples should subclass this.
    Subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given, unique key. Subclasses can also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`~torch.utils.data.Sampler` implementations and the default options
    of :class:`~torch.utils.data.DataLoader`.

    These DataPipes can be invoked in two ways, using the class constructor or applying their
    functional form onto an existing `MapDataPipe` (recommend, available to most but not all DataPipes).

    Note:
        :class:`~torch.utils.data.DataLoader` by default constructs an index
        sampler that yields integral indices. To make it work with a map-style
        DataPipe with non-integral indices/keys, a custom sampler must be provided.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper, Mapper
        >>> dp = SequenceWrapper(range(10))
        >>> map_dp_1 = dp.map(lambda x: x + 1)  # Using functional form (recommended)
        >>> list(map_dp_1)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> map_dp_2 = Mapper(dp, lambda x: x + 1)  # Using class constructor
        >>> list(map_dp_2)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> batch_dp = map_dp_1.batch(batch_size=2)
        >>> list(batch_dp)
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    """
    functions: dict[str, Callable]
    reduce_ex_hook: Callable | None
    getstate_hook: Callable | None
    str_hook: Callable | None
    repr_hook: Callable | None
    def __getattr__(self, attribute_name): ...
    @classmethod
    def register_function(cls, function_name, function) -> None: ...
    @classmethod
    def register_datapipe_as_function(cls, function_name, cls_to_register): ...
    def __getstate__(self):
        """
        Serialize `lambda` functions when `dill` is available.

        If this doesn't cover your custom DataPipe's use case, consider writing custom methods for
        `__getstate__` and `__setstate__`, or use `pickle.dumps` for serialization.
        """
    def __reduce_ex__(self, *args, **kwargs): ...
    @classmethod
    def set_getstate_hook(cls, hook_fn) -> None: ...
    @classmethod
    def set_reduce_ex_hook(cls, hook_fn) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __dir__(self): ...

class _DataPipeSerializationWrapper:
    _datapipe: Incomplete
    def __init__(self, datapipe) -> None: ...
    def __getstate__(self): ...
    def __setstate__(self, state) -> None: ...
    def __len__(self) -> int: ...

class _IterDataPipeSerializationWrapper(_DataPipeSerializationWrapper, IterDataPipe):
    _datapipe_iter: Iterator[_T_co] | None
    def __init__(self, datapipe: IterDataPipe[_T_co]) -> None: ...
    def __iter__(self) -> _IterDataPipeSerializationWrapper: ...
    def __next__(self) -> _T_co: ...

class _MapDataPipeSerializationWrapper(_DataPipeSerializationWrapper, MapDataPipe):
    def __getitem__(self, idx): ...
