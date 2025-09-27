import abc
from _typeshed import Incomplete
from collections.abc import Sequence
from torch import Generator, Tensor
from typing import Generic, Iterable, TypeVar

__all__ = ['Dataset', 'IterableDataset', 'TensorDataset', 'StackDataset', 'ConcatDataset', 'ChainDataset', 'Subset', 'random_split']

_T = TypeVar('_T')
_T_co = TypeVar('_T_co', covariant=True)
_T_dict = dict[str, _T_co]
_T_tuple = tuple[_T_co, ...]
_T_stack = TypeVar('_T_stack', _T_tuple, _T_dict)

class Dataset(Generic[_T_co]):
    """An abstract class representing a :class:`Dataset`.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`~torch.utils.data.Sampler` implementations and the default options
    of :class:`~torch.utils.data.DataLoader`. Subclasses could also
    optionally implement :meth:`__getitems__`, for speedup batched samples
    loading. This method accepts list of indices of samples of batch and returns
    list of samples.

    .. note::
      :class:`~torch.utils.data.DataLoader` by default constructs an index
      sampler that yields integral indices.  To make it work with a map-style
      dataset with non-integral indices/keys, a custom sampler must be provided.
    """
    def __getitem__(self, index) -> _T_co: ...
    def __add__(self, other: Dataset[_T_co]) -> ConcatDataset[_T_co]: ...

class IterableDataset(Dataset[_T_co], Iterable[_T_co], metaclass=abc.ABCMeta):
    '''An iterable Dataset.

    All datasets that represent an iterable of data samples should subclass it.
    Such form of datasets is particularly useful when data come from a stream.

    All subclasses should overwrite :meth:`__iter__`, which would return an
    iterator of samples in this dataset.

    When a subclass is used with :class:`~torch.utils.data.DataLoader`, each
    item in the dataset will be yielded from the :class:`~torch.utils.data.DataLoader`
    iterator. When :attr:`num_workers > 0`, each worker process will have a
    different copy of the dataset object, so it is often desired to configure
    each copy independently to avoid having duplicate data returned from the
    workers. :func:`~torch.utils.data.get_worker_info`, when called in a worker
    process, returns information about the worker. It can be used in either the
    dataset\'s :meth:`__iter__` method or the :class:`~torch.utils.data.DataLoader` \'s
    :attr:`worker_init_fn` option to modify each copy\'s behavior.

    Example 1: splitting workload across all workers in :meth:`__iter__`::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_DATALOADER)
        >>> # xdoctest: +SKIP("Fails on MacOS12")
        >>> class MyIterableDataset(torch.utils.data.IterableDataset):
        ...     def __init__(self, start, end):
        ...         super(MyIterableDataset).__init__()
        ...         assert end > start, "this example code only works with end >= start"
        ...         self.start = start
        ...         self.end = end
        ...
        ...     def __iter__(self):
        ...         worker_info = torch.utils.data.get_worker_info()
        ...         if worker_info is None:  # single-process data loading, return the full iterator
        ...             iter_start = self.start
        ...             iter_end = self.end
        ...         else:  # in a worker process
        ...             # split workload
        ...             per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
        ...             worker_id = worker_info.id
        ...             iter_start = self.start + worker_id * per_worker
        ...             iter_end = min(iter_start + per_worker, self.end)
        ...         return iter(range(iter_start, iter_end))
        ...
        >>> # should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
        >>> ds = MyIterableDataset(start=3, end=7)

        >>> # Single-process loading
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
        [tensor([3]), tensor([4]), tensor([5]), tensor([6])]

        >>> # xdoctest: +REQUIRES(POSIX)
        >>> # Multi-process loading with two worker processes
        >>> # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
        >>> # xdoctest: +IGNORE_WANT("non deterministic")
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=2)))
        [tensor([3]), tensor([5]), tensor([4]), tensor([6])]

        >>> # With even more workers
        >>> # xdoctest: +IGNORE_WANT("non deterministic")
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=12)))
        [tensor([3]), tensor([5]), tensor([4]), tensor([6])]

    Example 2: splitting workload across all workers using :attr:`worker_init_fn`::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_DATALOADER)
        >>> class MyIterableDataset(torch.utils.data.IterableDataset):
        ...     def __init__(self, start, end):
        ...         super(MyIterableDataset).__init__()
        ...         assert end > start, "this example code only works with end >= start"
        ...         self.start = start
        ...         self.end = end
        ...
        ...     def __iter__(self):
        ...         return iter(range(self.start, self.end))
        ...
        >>> # should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
        >>> ds = MyIterableDataset(start=3, end=7)

        >>> # Single-process loading
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
        [3, 4, 5, 6]
        >>>
        >>> # Directly doing multi-process loading yields duplicate data
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=2)))
        [3, 3, 4, 4, 5, 5, 6, 6]

        >>> # Define a `worker_init_fn` that configures each dataset copy differently
        >>> def worker_init_fn(worker_id):
        ...     worker_info = torch.utils.data.get_worker_info()
        ...     dataset = worker_info.dataset  # the dataset copy in this worker process
        ...     overall_start = dataset.start
        ...     overall_end = dataset.end
        ...     # configure the dataset to only process the split workload
        ...     per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
        ...     worker_id = worker_info.id
        ...     dataset.start = overall_start + worker_id * per_worker
        ...     dataset.end = min(dataset.start + per_worker, overall_end)
        ...

        >>> # Mult-process loading with the custom `worker_init_fn`
        >>> # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=2, worker_init_fn=worker_init_fn)))
        [3, 5, 4, 6]

        >>> # With even more workers
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=12, worker_init_fn=worker_init_fn)))
        [3, 4, 5, 6]
    '''
    def __add__(self, other: Dataset[_T_co]): ...

class TensorDataset(Dataset[tuple[Tensor, ...]]):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """
    tensors: tuple[Tensor, ...]
    def __init__(self, *tensors: Tensor) -> None: ...
    def __getitem__(self, index): ...
    def __len__(self) -> int: ...

class StackDataset(Dataset[_T_stack]):
    """Dataset as a stacking of multiple datasets.

    This class is useful to assemble different parts of complex input data, given as datasets.

    Example:
        >>> # xdoctest: +SKIP
        >>> images = ImageDataset()
        >>> texts = TextDataset()
        >>> tuple_stack = StackDataset(images, texts)
        >>> tuple_stack[0] == (images[0], texts[0])
        >>> dict_stack = StackDataset(image=images, text=texts)
        >>> dict_stack[0] == {'image': images[0], 'text': texts[0]}

    Args:
        *args (Dataset): Datasets for stacking returned as tuple.
        **kwargs (Dataset): Datasets for stacking returned as dict.
    """
    datasets: tuple | dict
    _length: Incomplete
    def __init__(self, *args: Dataset[_T_co], **kwargs: Dataset[_T_co]) -> None: ...
    def __getitem__(self, index): ...
    def __getitems__(self, indices: list): ...
    def __len__(self) -> int: ...

class ConcatDataset(Dataset[_T_co]):
    """Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """
    datasets: list[Dataset[_T_co]]
    cumulative_sizes: list[int]
    @staticmethod
    def cumsum(sequence): ...
    def __init__(self, datasets: Iterable[Dataset]) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx): ...
    @property
    def cummulative_sizes(self): ...

class ChainDataset(IterableDataset):
    """Dataset for chaining multiple :class:`IterableDataset` s.

    This class is useful to assemble different existing dataset streams. The
    chaining operation is done on-the-fly, so concatenating large-scale
    datasets with this class will be efficient.

    Args:
        datasets (iterable of IterableDataset): datasets to be chained together
    """
    datasets: Incomplete
    def __init__(self, datasets: Iterable[Dataset]) -> None: ...
    def __iter__(self): ...
    def __len__(self) -> int: ...

class Subset(Dataset[_T_co]):
    """
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset[_T_co]
    indices: Sequence[int]
    def __init__(self, dataset: Dataset[_T_co], indices: Sequence[int]) -> None: ...
    def __getitem__(self, idx): ...
    def __getitems__(self, indices: list[int]) -> list[_T_co]: ...
    def __len__(self) -> int: ...

def random_split(dataset: Dataset[_T], lengths: Sequence[int | float], generator: Generator | None = ...) -> list[Subset[_T]]:
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    Example:
        >>> # xdoctest: +SKIP
        >>> generator1 = torch.Generator().manual_seed(42)
        >>> generator2 = torch.Generator().manual_seed(42)
        >>> random_split(range(10), [3, 7], generator=generator1)
        >>> random_split(range(30), [0.3, 0.3, 0.4], generator=generator2)

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
