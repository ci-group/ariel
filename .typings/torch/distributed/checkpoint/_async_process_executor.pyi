import os
import torch.distributed as dist
from _typeshed import Incomplete
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum
from torch.distributed.checkpoint._async_executor import _AsyncCheckpointExecutor as _AsyncCheckpointExecutor
from torch.distributed.checkpoint.logger import _dcp_method_logger as _dcp_method_logger, _init_logger as _init_logger
from torch.distributed.checkpoint.metadata import Metadata as Metadata, STATE_DICT_TYPE as STATE_DICT_TYPE
from torch.distributed.checkpoint.planner import SavePlanner as SavePlanner
from torch.distributed.checkpoint.storage import StorageWriter as StorageWriter
from torch.distributed.checkpoint.utils import _DistWrapper as _DistWrapper
from torch.distributed.elastic.agent.server.api import _get_fq_hostname as _get_fq_hostname
from torch.distributed.elastic.utils.distributed import get_free_port as get_free_port
from typing import Any

logger: Incomplete

class _CheckpointSaveProcessControlOpts(Enum):
    INIT_COMPLETE = 'init_complete'
    TERMINATE = 'terminate'

@dataclass(init=False, unsafe_hash=True)
class _CheckpointRequestIdentifier:
    checkpoint_id: str | os.PathLike | None
    uuid: str
    def __init__(self, checkpoint_id: str | os.PathLike | None) -> None: ...

@dataclass
class _AsyncCheckpointRequest:
    staged_state_dict: STATE_DICT_TYPE
    checkpoint_request_id: _CheckpointRequestIdentifier
    storage_writer: StorageWriter | None = ...
    planner: SavePlanner | None = ...

@dataclass(init=False)
class _ProcessGroupInitInfo:
    local_rank: int
    global_rank: int
    world_size: int
    tcp_store_master_addr: str
    tcp_store_master_port: int
    def __init__(self, process_group: dist.ProcessGroup | None = None) -> None: ...

class _AsyncCheckpointProcess:
    ctx: Incomplete
    _save_process: Incomplete
    def __init__(self, pg_init_info: _ProcessGroupInitInfo) -> None: ...
    def __del__(self) -> None: ...
    def _send(self, data: Any) -> None: ...
    def _wait_for_response(self, timeout: float | None = None) -> Any: ...
    def save(self, staged_state_dict: STATE_DICT_TYPE, *, checkpoint_id: str | os.PathLike | None = None, storage_writer: StorageWriter | None = None, planner: SavePlanner | None = None) -> Metadata: ...
    @staticmethod
    def _execute_save(state_dict: STATE_DICT_TYPE, *, checkpoint_request_id: _CheckpointRequestIdentifier, storage_writer: StorageWriter | None = None, planner: SavePlanner | None = None) -> Metadata: ...
    @staticmethod
    def _checkpointing_subprocess(pg_init_info: _ProcessGroupInitInfo, parent_conn) -> None: ...

_CHECKPOINT_PROCESS: _AsyncCheckpointProcess | None

class _ProcessBasedAsyncCheckpointExecutor(_AsyncCheckpointExecutor):
    _executor: Incomplete
    def __init__(self) -> None: ...
    @staticmethod
    def _execute_save_impl(*, pg_init_info: _ProcessGroupInitInfo | None, staged_state_dict: STATE_DICT_TYPE, checkpoint_id: str | os.PathLike | None = None, storage_writer: StorageWriter | None = None, planner: SavePlanner | None = None, process_group: dist.ProcessGroup | None = None) -> Metadata: ...
    def execute_save(self, staged_state_dict: STATE_DICT_TYPE, *, checkpoint_id: str | os.PathLike | None = None, storage_writer: StorageWriter | None = None, planner: SavePlanner | None = None, process_group: dist.ProcessGroup | None = None) -> Future:
        """
        NOTE:

        - Checkpoint process is implemented as a daemon process.
        The AsyncCheckpointProcess' lifetime is tied to the lifetime of the
        main process (e.g. trainer process).

        - The first call to execute_save_in_process() will initialize the checkpoint
        daemon process. Subsequent async checkpoint requests will not need process
        initialization. Therefore, the first async checkpoint request will take longer to complete.

        - Process initialization can have significant overhead, dominated by latency for all ranks to spawn
        a background process + process group initialization in the background process.
        """
