import abc
import os
import torch.distributed as dist
from concurrent.futures import Future
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE as STATE_DICT_TYPE
from torch.distributed.checkpoint.planner import SavePlanner as SavePlanner
from torch.distributed.checkpoint.storage import StorageWriter as StorageWriter

class _AsyncCheckpointExecutor(abc.ABC, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def execute_save(self, staged_state_dict: STATE_DICT_TYPE, *, checkpoint_id: str | os.PathLike | None = None, storage_writer: StorageWriter | None = None, planner: SavePlanner | None = None, process_group: dist.ProcessGroup | None = None) -> Future:
        """
        Execute the checkpoint save request asynchronously.

        This method is intended to be used as an abstraction for
        implementing async checkpointing. The actual checkpoint save
        operation is executed in a separate thread or process depending
        on the implementation of this interface.
        """
