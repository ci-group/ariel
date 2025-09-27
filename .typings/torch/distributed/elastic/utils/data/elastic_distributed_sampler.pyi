from _typeshed import Incomplete
from collections.abc import Iterator
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from typing import TypeVar

__all__ = ['ElasticDistributedSampler']

T = TypeVar('T')

class ElasticDistributedSampler(DistributedSampler[T]):
    """
    Sampler that restricts data loading to a subset of
    the dataset for elastic training.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        start_index (optional):  Which index of the dataset to start sampling from
    """
    start_index: Incomplete
    num_samples: Incomplete
    total_size: Incomplete
    def __init__(self, dataset: Dataset[T], num_replicas: int | None = None, rank: int | None = None, start_index: int = 0) -> None: ...
    def __iter__(self) -> Iterator[T]: ...
    def __len__(self) -> int: ...
