from ._traverse import OBJ_PATH as OBJ_PATH, STATE_DICT_ITEM as STATE_DICT_ITEM, set_element as set_element, traverse_state_dict as traverse_state_dict
from .utils import _element_wise_add as _element_wise_add, _normalize_device_info as _normalize_device_info
from torch.distributed._shard.sharded_tensor import Shard as Shard, ShardMetadata as ShardMetadata, ShardedTensor as ShardedTensor
from torch.distributed._shard.sharded_tensor.metadata import ShardedTensorMetadata as ShardedTensorMetadata
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE as STATE_DICT_TYPE
from torch.distributed.remote_device import _remote_device as _remote_device

def _flatten_sharded_tensors(state_dict: STATE_DICT_TYPE) -> STATE_DICT_TYPE:
    """
    Transform ``state_dict`` by flattening all nested ShardedTensor instances found.

    The resulting ShardedTensor instances are only correct regarding the local shard and
    MUST not be used for any other purpose but checkpointing, as no operator will work with them.

    This function should be used in conjunction with a state_dict produced by FSDP's
    StateDictType.SHARDED_STATE_DICT methods.
    """
