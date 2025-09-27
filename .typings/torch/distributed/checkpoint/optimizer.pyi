import torch
from _typeshed import Incomplete
from collections.abc import Sequence
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import Metadata, MetadataIndex, STATE_DICT_TYPE
from torch.distributed.checkpoint.planner import LoadPlan, LoadPlanner
from torch.distributed.checkpoint.storage import StorageReader

__all__ = ['load_sharded_optimizer_state_dict']

STATE_DICT_2D_LAYOUT = dict[str, tuple[Sequence[int] | None, Sequence[int]]]

class _ReaderWithOffset(DefaultLoadPlanner):
    translation: dict[MetadataIndex, MetadataIndex]
    state_dict: STATE_DICT_TYPE
    metadata: Metadata
    fqn_to_offset: Incomplete
    def __init__(self, fqn_to_offset: dict[str, Sequence[int]]) -> None: ...
    def create_local_plan(self) -> LoadPlan: ...
    def lookup_tensor(self, index: MetadataIndex) -> torch.Tensor: ...

def load_sharded_optimizer_state_dict(model_state_dict: STATE_DICT_TYPE, optimizer_key: str, storage_reader: StorageReader, planner: LoadPlanner | None = None) -> STATE_DICT_TYPE:
    '''
    Load a state_dict in conjunction with FSDP sharded optimizer state.

    This is the current recommended way to checkpoint FSDP.
    >>> # xdoctest: +SKIP
    >>> import torch.distributed.checkpoint as dist_cp
    >>> # Save
    >>> model: torch.nn.Model
    >>> optim_params = model.parameters()
    >>> optim = torch.optim.SGD(optim_params, lr=0.01)
    >>> # Save
    >>> with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
    >>>     state_dict = {
    >>>         "optimizer": FSDP.optim_state_dict(model, optim),
    >>>         "model": model.state_dict()
    >>>     }
    >>>     dist_cp.save_state_dict(
    >>>         state_dict=optim_state,
    >>>         storage_writer=dist_cp.FileSystemWriter("checkpoint"),
    >>>         planner=dist_cp.DefaultSavePlanner(),
    >>>     )
    >>>
    >>> # Load
    >>> with FSDP.state_dict_type(model_tp, StateDictType.SHARDED_STATE_DICT):
    >>>     model_state_dict = model_tp.state_dict()
    >>>     checkpoint = {
    >>>         "model": model_state_dict
    >>>     }
    >>>     dist_cp.load_state_dict(
    >>>         state_dict=checkpoint,
    >>>         storage_reader=dist_cp.FileSystemReader(checkpoint_file),
    >>>         planner=dist_cp.DefaultLoadPlanner(),
    >>>     )
    >>>     model.load_state_dict(checkpoint["model_state"])
    >>>
    >>>     optim_state = dist_cp.load_sharded_optimizer_state_dict(
    >>>         model_state_dict,
    >>>         optimizer_key="optimizer",
    >>>         storage_reader=dist_cp.FileSystemReader("checkpoint"),
    >>>     )
    >>>
    >>>     flattened_osd = FSDP.optim_state_dict_to_load(
    >>>        model, optim, optim_state["optimizer"]
    >>>     )
    >>>
    >>>     optim.load_state_dict(flattened_osd)
    '''
