import torch.distributed as dist
from _typeshed import Incomplete
from concurrent.futures import Future
from torch.distributed.checkpoint.metadata import Metadata as Metadata, STATE_DICT_TYPE as STATE_DICT_TYPE
from torch.distributed.checkpoint.storage import LoadPlanner as LoadPlanner, SavePlanner as SavePlanner, StorageReader as StorageReader, StorageWriter as StorageWriter
from typing import Any

__all__: list[str]

class _Checkpointer:
    """This base class specefies a high level API for saving and loading
    distributed `state_dict` 's. It provides an abstraction over the low-level APIs
    provided by :py:mod:`torch.distributed.checkpoint.storage`, essentially calling
    :py:meth: `torch.distributed.state_dict_saver.save` and
    :py:meth: `torch.distributed.state_dict_loader.load` with the provided storage
    readers and writers.

    .. warning::
        This feature is experimental and subject to removal/change.

    """
    storage_writer: Incomplete
    storage_reader: Incomplete
    process_group: Incomplete
    coordinator_rank: Incomplete
    no_dist: Incomplete
    load_planner: Incomplete
    save_planner: Incomplete
    def __init__(self, storage_writer: StorageWriter, storage_reader: StorageReader, *, process_group: dist.ProcessGroup | None = None, coordinator_rank: int = 0, no_dist: bool = False, load_planner: LoadPlanner | None = None, save_planner: SavePlanner | None = None) -> None:
        """Initializes the Checkpointer instance.

        Args:
            storage_writer: Instance of StorageWrite use to perform writes.
            storage_reader: StorageReader used to load data from.
            process_group: ProcessGroup to be used for cross-rank synchronization.
            coordinator_rank: Rank to use to coordinate the checkpoint. rank0 is used by default.
            no_dist: If ``True``, distributed checkpoint will not load in SPMD style. (Default: ``False``)
            loader_planner: Instance of LoadPlanner to use when loading.
            save_planner: Instance of SavePlanner to use when saving.
        """
    def save(self, state_dict: STATE_DICT_TYPE) -> Metadata:
        """Calls :py:meth: `torch.distributed.state_dict_saver.save`. Utilizing values passed during initialization."""
    def async_save(self, state_dict: STATE_DICT_TYPE) -> Future:
        """
        Calls :py:meth: `torch.distributed.state_dict_saver._async_save`. Utilizing values passed during initialization.

        Returns:
            Future: A future holding the resultant Metadata object from `save`.
        """
    def load(self, state_dict: dict[str, Any]) -> None:
        """Calls :py:meth: `torch.distributed.state_dict_loader.load`. Utilizing values passed during initialization."""
