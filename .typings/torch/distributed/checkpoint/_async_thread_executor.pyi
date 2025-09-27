import os
import torch.distributed as dist
from _typeshed import Incomplete
from concurrent.futures import Future
from torch.distributed.checkpoint._async_executor import _AsyncCheckpointExecutor as _AsyncCheckpointExecutor
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE as STATE_DICT_TYPE
from torch.distributed.checkpoint.planner import SavePlanner as SavePlanner
from torch.distributed.checkpoint.storage import StorageWriter as StorageWriter

class _ThreadBasedAsyncCheckpointExecutor(_AsyncCheckpointExecutor):
    _executor: Incomplete
    def __init__(self) -> None: ...
    def execute_save(self, staged_state_dict: STATE_DICT_TYPE, *, checkpoint_id: str | os.PathLike | None = None, storage_writer: StorageWriter | None = None, planner: SavePlanner | None = None, process_group: dist.ProcessGroup | None = None) -> Future: ...
