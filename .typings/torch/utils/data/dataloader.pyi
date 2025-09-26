from _typeshed import Incomplete
from collections.abc import Iterable
from torch.utils.data import _utils
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
from typing import Any, Callable, Generic, TypeVar
from typing_extensions import Self

__all__ = ['DataLoader', 'get_worker_info', 'default_collate', 'default_convert']

_T = TypeVar('_T')
_T_co = TypeVar('_T_co', covariant=True)
_worker_init_fn_t = Callable[[int], None]
_collate_fn_t = Callable[[list[_T]], Any]
default_collate: _collate_fn_t
default_convert = _utils.collate.default_convert
get_worker_info = _utils.worker.get_worker_info

class _DatasetKind:
    Map: int
    Iterable: int
    @staticmethod
    def create_fetcher(kind, dataset, auto_collation, collate_fn, drop_last): ...

class _InfiniteConstantSampler(Sampler):
    """Analogous to ``itertools.repeat(None, None)``.

    Used as sampler for :class:`~torch.utils.data.IterableDataset`.
    """
    def __iter__(self): ...

class DataLoader(Generic[_T_co]):
    """
    Data loader combines a dataset and a sampler, and provides an iterable over the given dataset.

    The :class:`~torch.utils.data.DataLoader` supports both map-style and
    iterable-style datasets with single- or multi-process loading, customizing
    loading order and optional automatic batching (collation) and memory pinning.

    See :py:mod:`torch.utils.data` documentation page for more details.

    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        sampler (Sampler or Iterable, optional): defines the strategy to draw
            samples from the dataset. Can be any ``Iterable`` with ``__len__``
            implemented. If specified, :attr:`shuffle` must not be specified.
        batch_sampler (Sampler or Iterable, optional): like :attr:`sampler`, but
            returns a batch of indices at a time. Mutually exclusive with
            :attr:`batch_size`, :attr:`shuffle`, :attr:`sampler`,
            and :attr:`drop_last`.
        num_workers (int, optional): how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the main process.
            (default: ``0``)
        collate_fn (Callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
        pin_memory (bool, optional): If ``True``, the data loader will copy Tensors
            into device/CUDA pinned memory before returning them.  If your data elements
            are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
            see the example below.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: ``False``)
        timeout (numeric, optional): if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative. (default: ``0``)
        worker_init_fn (Callable, optional): If not ``None``, this will be called on each
            worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
            input, after seeding and before data loading. (default: ``None``)
        multiprocessing_context (str or multiprocessing.context.BaseContext, optional): If
            ``None``, the default
            `multiprocessing context <https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods>`_ # noqa: D401
            of your operating system will
            be used. (default: ``None``)
        generator (torch.Generator, optional): If not ``None``, this RNG will be used
            by RandomSampler to generate random indexes and multiprocessing to generate
            ``base_seed`` for workers. (default: ``None``)
        prefetch_factor (int, optional, keyword-only arg): Number of batches loaded
            in advance by each worker. ``2`` means there will be a total of
            2 * num_workers batches prefetched across all workers. (default value depends
            on the set value for num_workers. If value of num_workers=0 default is ``None``.
            Otherwise, if value of ``num_workers > 0`` default is ``2``).
        persistent_workers (bool, optional): If ``True``, the data loader will not shut down
            the worker processes after a dataset has been consumed once. This allows to
            maintain the workers `Dataset` instances alive. (default: ``False``)
        pin_memory_device (str, optional): the device to :attr:`pin_memory` on if ``pin_memory`` is
            ``True``. If not given, the current :ref:`accelerator<accelerators>` will be the
            default. This argument is discouraged and subject to deprecated.
        in_order (bool, optional): If ``False``, the data loader will not enforce that batches
            are returned in a first-in, first-out order. Only applies when ``num_workers > 0``. (default: ``True``)


    .. warning:: If the ``spawn`` start method is used, :attr:`worker_init_fn`
                 cannot be an unpicklable object, e.g., a lambda function. See
                 :ref:`multiprocessing-best-practices` on more details related
                 to multiprocessing in PyTorch.

    .. warning:: ``len(dataloader)`` heuristic is based on the length of the sampler used.
                 When :attr:`dataset` is an :class:`~torch.utils.data.IterableDataset`,
                 it instead returns an estimate based on ``len(dataset) / batch_size``, with proper
                 rounding depending on :attr:`drop_last`, regardless of multi-process loading
                 configurations. This represents the best guess PyTorch can make because PyTorch
                 trusts user :attr:`dataset` code in correctly handling multi-process
                 loading to avoid duplicate data.

                 However, if sharding results in multiple workers having incomplete last batches,
                 this estimate can still be inaccurate, because (1) an otherwise complete batch can
                 be broken into multiple ones and (2) more than one batch worth of samples can be
                 dropped when :attr:`drop_last` is set. Unfortunately, PyTorch can not detect such
                 cases in general.

                 See `Dataset Types`_ for more details on these two types of datasets and how
                 :class:`~torch.utils.data.IterableDataset` interacts with
                 `Multi-process data loading`_.

    .. warning:: See :ref:`reproducibility`, and :ref:`dataloader-workers-random-seed`, and
                 :ref:`data-loading-randomness` notes for random seed related questions.

    .. warning:: Setting `in_order` to `False` can harm reproducibility and may lead to a skewed data
                 distribution being fed to the trainer in cases with imbalanced data.
    """
    dataset: Dataset[_T_co]
    batch_size: int | None
    num_workers: int
    pin_memory: bool
    drop_last: bool
    timeout: float
    sampler: Sampler | Iterable
    pin_memory_device: str
    prefetch_factor: int | None
    _iterator: _BaseDataLoaderIter | None
    __initialized: bool
    worker_init_fn: Incomplete
    in_order: Incomplete
    _dataset_kind: Incomplete
    batch_sampler: Incomplete
    generator: Incomplete
    collate_fn: Incomplete
    persistent_workers: Incomplete
    _IterableDataset_len_called: Incomplete
    def __init__(self, dataset: Dataset[_T_co], batch_size: int | None = 1, shuffle: bool | None = None, sampler: Sampler | Iterable | None = None, batch_sampler: Sampler[list] | Iterable[list] | None = None, num_workers: int = 0, collate_fn: _collate_fn_t | None = None, pin_memory: bool = False, drop_last: bool = False, timeout: float = 0, worker_init_fn: _worker_init_fn_t | None = None, multiprocessing_context=None, generator=None, *, prefetch_factor: int | None = None, persistent_workers: bool = False, pin_memory_device: str = '', in_order: bool = True) -> None: ...
    def _get_iterator(self) -> _BaseDataLoaderIter: ...
    @property
    def multiprocessing_context(self): ...
    __multiprocessing_context: Incomplete
    @multiprocessing_context.setter
    def multiprocessing_context(self, multiprocessing_context) -> None: ...
    def __setattr__(self, attr, val) -> None: ...
    def __iter__(self) -> _BaseDataLoaderIter: ...
    @property
    def _auto_collation(self): ...
    @property
    def _index_sampler(self): ...
    def __len__(self) -> int: ...
    def check_worker_number_rationality(self): ...

class _BaseDataLoaderIter:
    _dataset: Incomplete
    _shared_seed: Incomplete
    _pg: Incomplete
    _dataset_kind: Incomplete
    _IterableDataset_len_called: Incomplete
    _auto_collation: Incomplete
    _drop_last: Incomplete
    _index_sampler: Incomplete
    _num_workers: Incomplete
    _world_size: Incomplete
    _rank: Incomplete
    _pin_memory: Incomplete
    _pin_memory_device: Incomplete
    _timeout: Incomplete
    _collate_fn: Incomplete
    _sampler_iter: Incomplete
    _base_seed: Incomplete
    _persistent_workers: Incomplete
    _num_yielded: int
    _profile_name: Incomplete
    def __init__(self, loader: DataLoader) -> None: ...
    def __iter__(self) -> Self: ...
    def _reset(self, loader, first_iter: bool = False) -> None: ...
    def _next_index(self): ...
    def _next_data(self) -> None: ...
    def __next__(self) -> Any: ...
    def __len__(self) -> int: ...
    def __getstate__(self) -> None: ...

class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    _dataset_fetcher: Incomplete
    def __init__(self, loader) -> None: ...
    def _next_data(self): ...

class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
    """Iterates once over the DataLoader's dataset, as specified by the sampler."""
    _prefetch_factor: Incomplete
    _in_order: Incomplete
    _worker_init_fn: Incomplete
    _worker_result_queue: Incomplete
    _worker_pids_set: bool
    _shutdown: bool
    _workers_done_event: Incomplete
    _index_queues: Incomplete
    _workers: Incomplete
    _pin_memory_thread_done_event: Incomplete
    _data_queue: Incomplete
    _pin_memory_thread: Incomplete
    def __init__(self, loader) -> None: ...
    _send_idx: int
    _rcvd_idx: int
    _task_info: Incomplete
    _tasks_outstanding: int
    _workers_status: Incomplete
    _workers_num_tasks: Incomplete
    _worker_queue_idx_cycle: Incomplete
    def _reset(self, loader, first_iter: bool = False) -> None: ...
    def _try_get_data(self, timeout=...): ...
    def _get_data(self): ...
    def _next_data(self): ...
    def _try_put_index(self) -> None: ...
    def _process_data(self, data, worker_idx): ...
    def _mark_worker_as_unavailable(self, worker_id, shutdown: bool = False) -> None: ...
    def _shutdown_workers(self) -> None: ...
    @staticmethod
    def _clean_up_worker(w) -> None: ...
    def __del__(self) -> None: ...
