import os
import torch.distributed as dist
from .utils import _api_bc_check
from concurrent.futures import Future
from enum import Enum
from torch.distributed.checkpoint.metadata import Metadata, STATE_DICT_TYPE
from torch.distributed.checkpoint.planner import SavePlanner
from torch.distributed.checkpoint.storage import StorageWriter

__all__ = ['save_state_dict', 'save', 'async_save', 'AsyncCheckpointerType']

class AsyncCheckpointerType(Enum):
    """Enum for async checkpointer type."""
    THREAD = 'thread'
    PROCESS = 'process'

def save_state_dict(state_dict: STATE_DICT_TYPE, storage_writer: StorageWriter, process_group: dist.ProcessGroup | None = None, coordinator_rank: int = 0, no_dist: bool = False, planner: SavePlanner | None = None) -> Metadata:
    """This method is deprecated. Please switch to 'save'."""
@_api_bc_check
def save(state_dict: STATE_DICT_TYPE, *, checkpoint_id: str | os.PathLike | None = None, storage_writer: StorageWriter | None = None, planner: SavePlanner | None = None, process_group: dist.ProcessGroup | None = None, no_dist: bool = False) -> Metadata:
    '''
    Save a distributed model in SPMD style.

    This function is different from ``torch.save()`` as it handles
    ``ShardedTensor`` , and ``DTensor`` by having each rank only save their local shards.

    For each ``Stateful`` object (having both a ``state_dict`` and a ``load_state_dict``),
    save will call ``state_dict`` before serialization.

    .. warning::
        There is no guarantees of Backwards Compatibility across PyTorch versions
        for saved state_dicts.

    .. warning::
        If using the `process_group` argument, make sure that only its ranks
        call `save_state_dict` and that all data in state_dict belong to it.

    .. note::
        When saving checkpoint for FSDP\'s `ShardingStrategy.HYBRID_SHARD`, only one of
        the shard_group should be calling `save_state_dict` and the corresponding process
        group needs to be passed in.

    .. note::
        If no process group is available, this function assumes the intention is to save the
         state_dict in the local process.

    .. note:
        Rank 0 is assumed to be the coordinator rank.


    Args:
        state_dict (Dict[str, Any]): The state_dict to save.
        checkpoint_id (Union[str, os.PathLike, None]):
            The ID of this checkpoint instance. The meaning of the checkpoint_id
            depends on the storage. It can be a path to a folder or to a file.
            It can also be a key if the storage is a key-value store.
            (Default: ``None``)
        storage_writer (Optional[StorageWriter]):
            Instance of StorageWriter used to perform writes. If this is not
            specified, DCP will automatically infer the writer based on the
            checkpoint_id. If checkpoint_id is also None, an exception will
            be raised. (Default: ``None``)
        planner (Optional[SavePlanner]):
            Instance of SavePlanner. If this is not specified, the default
            planner will be used. (Default: ``None``)
        process_group (Optional[ProcessGroup]):
            ProcessGroup to be used for cross-rank synchronization.
            (Default: ``None``)
        no_dist (bool):
            If ``True``, this function will assume the intent is to load
            a checkpoint without using cross-rank synchronization.
            (Default: ``False``)

    Returns:
        Metadata: Metadata object for the saved checkpoint.

    Example:
        >>> # xdoctest: +SKIP
        >>> my_model = MyModule()

        >>> state_dict = {"model": my_model}

        >>> fs_storage_writer = torch.distributed.checkpoint.FileSystemWriter(
        ...     "/checkpoint/1"
        ... )
        >>> torch.distributed.checkpoint.save(
        >>>     state_dict=state_dict,
        >>>     storage_writer=fs_storage_writer,
        >>> )

    .. note::
        save_state_dict uses collectives to coordinate writes across ranks.
        For NCCL-based process groups, internal tensor representations of
        objects must be moved to the GPU device before communication takes place.
        In this case, the device used is given by ``torch.cuda.current_device()``
        and it is the user\'s responsibility to ensure that this is set so that
        each rank has an individual GPU, via ``torch.cuda.set_device()``.
    '''
def async_save(state_dict: STATE_DICT_TYPE, *, checkpoint_id: str | os.PathLike | None = None, storage_writer: StorageWriter | None = None, planner: SavePlanner | None = None, process_group: dist.ProcessGroup | None = None, async_checkpointer_type: AsyncCheckpointerType = ...) -> Future:
    '''Asynchronous version of ``save``. This code first de-stages the state_dict on to the
    staging storage (defaults to CPU memory), and then calls the `save` in a separate thread.

    .. warning::
        This feature is experimental and subject to change.

    Args:
        state_dict (Dict[str, Any]): The state_dict to save.
        checkpoint_id (Union[str, os.PathLike, None]):
            The ID of this checkpoint instance. The meaning of the checkpoint_id
            depends on the storage. It can be a path to a folder or to a file.
            It can also be a key if the storage is a key-value store.
            (Default: ``None``)
        storage_writer (Optional[StorageWriter]):
            Instance of StorageWriter used to perform \'stage\' and  \'save\'. If
            this is not specified, DCP will automatically infer the writer based on the
            checkpoint_id. If checkpoint_id is also None, an exception will
            be raised. (Default: ``None``)
        planner (Optional[SavePlanner]):
            Instance of SavePlanner. If this is not specified, the default
            planner will be used. (Default: ``None``)
        process_group (Optional[ProcessGroup]):
            ProcessGroup to be used for cross-rank synchronization.
            (Default: ``None``)

    Returns:
        Future: A future holding the resultant Metadata object from `save`.

    Example:
        >>> # xdoctest: +SKIP
        >>> my_model = MyModule()

        >>> state_dict = {"model": my_model}

        >>> fs_storage_writer = torch.distributed.checkpoint.FileSystemWriter(
        ...     "/checkpoint/1"
        ... )
        >>> checkpoint_future = torch.distributed.checkpoint.async_save(
        >>>     state_dict=state_dict,
        >>>     storage_writer=fs_storage_writer,
        >>> )
        >>>
        >>> # ... do some work ...
        >>>
        >>> checkpoint_future.result()

    '''
