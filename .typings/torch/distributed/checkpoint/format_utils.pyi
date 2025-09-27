import os
from _typeshed import Incomplete
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import Metadata, STATE_DICT_TYPE
from torch.distributed.checkpoint.planner import LoadPlan, LoadPlanner
from torch.distributed.checkpoint.storage import StorageReader
from torch.futures import Future

__all__ = ['dcp_to_torch_save', 'torch_save_to_dcp', 'BroadcastingTorchSaveReader', 'DynamicMetaLoadPlanner']

class BroadcastingTorchSaveReader(StorageReader):
    '''
    StorageReader for reading a Torch Save file. This reader will read the entire checkpoint
    on the coordinator rank, and then broadcast and shard each tensor to all ranks.

    . N.B. Intended to be used with DynamicMetaLoadPlanner

    .. warning::
        Current implementation only supports loading Tensors.

    >>> # xdoctest: +SKIP("undefined vars")
    >>> sd = {"mode": model}
    >>> dcp.load(
    >>>    sd,
    >>>    storage_reader=BroadcastingTorchSaveReader(),
    >>>    planner=DynamicMetaLoadPlanner(),
    >>>    checkpoint_id="path_to_model.pt"
    >>> )
    '''
    checkpoint_id: Incomplete
    coordinator_rank: Incomplete
    def __init__(self, checkpoint_id: str | os.PathLike | None = None, coordinator_rank: int = 0) -> None: ...
    def read_metadata(self) -> Metadata:
        """Extends the default StorageReader to support building the metadata file"""
    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        """
        Reads torch save data on the coordinator rank, and broadcast afterwards
        this incurrs a communication cost, but avoids having to load
        the entire checkpoint on each rank, hopefully preventing OOM issues
        """
    is_coordinator: Incomplete
    def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool) -> None:
        """Implementation of the StorageReader method"""
    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        """Implementation of the StorageReader method"""
    def prepare_global_plan(self, global_plan: list[LoadPlan]) -> list[LoadPlan]:
        """Implementation of the StorageReader method"""
    def reset(self, checkpoint_id: str | os.PathLike | None = None) -> None:
        """Implementation of the StorageReader method"""
    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool:
        """Implementation of the StorageReader method"""

class DynamicMetaLoadPlanner(DefaultLoadPlanner):
    '''
    Extension of DefaultLoadPlanner, which creates a new Metadata object based on the passed in state dict,
    avoiding the need to read metadata from disk. This is useful when reading formats which don\'t have a
    metadata file, like Torch Save files.

    . N.B. Intended to be used with BroadcastingTorchSaveReader

    .. warning::
        Current implementation only supports loading Tensors.

    >>> # xdoctest: +SKIP("undefined vars")
    >>> sd = {"mode": model}
    >>> dcp.load(
    >>>    sd,
    >>>    storage_reader=BroadcastingTorchSaveReader(),
    >>>    planner=DynamicMetaLoadPlanner(),
    >>>    checkpoint_id="path_to_model.pt"
    >>> )
    '''
    metadata: Incomplete
    def set_up_planner(self, state_dict: STATE_DICT_TYPE, metadata: Metadata | None = None, is_coordinator: bool = False) -> None:
        """Setups of the planner, extnding default behavior by creating the Metadata object from the state dict"""

def dcp_to_torch_save(dcp_checkpoint_dir: str | os.PathLike, torch_save_path: str | os.PathLike):
    """
    Given a directory containing a DCP checkpoint, this function will convert it into a
    Torch save file.

    Args:
        dcp_checkpoint_dir: Directory containing the DCP checkpoint.
        torch_save_path: Filename to store the converted Torch save file.

    .. warning::
        To avoid OOM, it's recommended to only run this function on a single rank.
    """
def torch_save_to_dcp(torch_save_path: str | os.PathLike, dcp_checkpoint_dir: str | os.PathLike):
    """
    Given the location of a torch save file, converts it into a DCP checkpoint.

    Args:
        torch_save_path: Filename of the Torch save file.
        dcp_checkpoint_dir: Directory to store the DCP checkpoint.

    .. warning::
        To avoid OOM, it's recommended to only run this function on a single rank.
    """
