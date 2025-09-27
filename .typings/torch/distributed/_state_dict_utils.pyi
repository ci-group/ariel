import torch
import torch.distributed as dist
from collections.abc import MutableMapping
from torch.distributed import distributed_c10d as distributed_c10d
from torch.distributed._functional_collectives import AsyncCollectiveTensor as AsyncCollectiveTensor
from torch.distributed._shard.sharded_tensor import ShardedTensor as ShardedTensor
from torch.distributed.tensor import DTensor as DTensor, Replicate as Replicate, distribute_tensor as distribute_tensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset as compute_local_shape_and_global_offset
from typing import Any, Callable, NamedTuple

def _identity_func(obj: torch.Tensor, pg: dist.ProcessGroup | None, device: torch.device | None, companion_obj: Any) -> torch.Tensor: ...
def _all_gather_sharded_tensor(sharded_tensor: ShardedTensor, pg: dist.ProcessGroup | None = None, device: torch.device | None = None) -> torch.Tensor: ...

class CompanionMismatch(Exception): ...

def _iterate_state_dict(iter_object: Any, sharded_tensor_func: Callable, dtensor_func: Callable, tensor_func: Callable, *, pg: dist.ProcessGroup | None = None, device: torch.device | None = None, cpu_offload: bool = False, companion_obj: Any = None, ranks_only: tuple[int, ...] = (), type_check: bool = True, non_blocking: bool = True) -> dict[str, Any]:
    """Iterate through the state dict, applying the given functions to each tensor type.

    Args:
        iter_object (Any): the target state_dict.
        sharded_tensor_func (Callable): the function to apply to ShardedTensor
        dtensor_func (Callable): the function to apply to DTensor
        tensor_func (Callable): the function to apply to Tensor
        pg (Optional[dist.ProcessGroup]): process group passed to tensor functions
        device (Optional[torch.device]): device passed to tensor functions
        cpu_offload (bool): whether to offload the tensors to CPU memory. This option is ignored
            if a companion_obj is supplied.
        companion_obj (Any): A companion object to the state dict. If this object
            is supplied, we attempt to copy the tensor to the companion object.
        ranks_only (Tuple[int, ...]): if this tuple is empty, all ranks will
            have the same state_dicts. Otherwise only ranks that in ``ranks_only``
            have the same state_dicts. Other ranks will get empty state_dicts.
        type_check (bool): check if the instance data type is a supported type
            that can be saved by DCP.  The current supported data types are
            torch.Tensor, DTensor, int, float, str, list, dict, None.
        non_blocking (bool): whether to use non-blocking copy when copying to the companion object.
    """
def _gather_state_dict(state_dict: dict[str, Any], *, pg: dist.ProcessGroup | None = None, device: torch.device | None = None, cpu_offload: bool = False, ranks_only: tuple[int, ...] = (), type_check: bool = True) -> dict[str, Any]:
    """
    Given a state_dict, this API gathers all the ShardedTensors or DTensors in
    the state_dict.


    Args:
        state_dict (Dict[str, Any]): the target sharded state_dict.
        pg (Optional[dist.ProcessGroup]): the process group that is used to
            gather ShardedTensor. Note that gathering a DTensor will use
            the DeviceMesh. So this argument will be ignored when gathering a
            DTensor.
        device: (Optional[torch.device]): the device that is used to
            perform allgather for ShardedTensor. Note that gathering a DTensor
            will use the DeviceMesh. So this argument will be ignored when
            gathering a DTensor.
        cpu_offload (bool): whether to offload the tensors to CPU memory. The
            default value is False.
        ranks_only: (Tuple[int, ...]): if this tuple is empty, all ranks will
            have the same state_dicts. Otherwise only ranks that in ``ranks_only``
            have the same state_dicts. Other ranks will get empty state_dicts.
        type_check: (bool): check if the instance data type is a supported type
            that can be saved by DCP.  The current supported data types are
            torch.Tensor, DTensor, int, float, str, list, dict, None.

    Returns:
        The gathered state dictionary.
    """
def _offload_state_dict_to_cpu(state_dict: dict[str, Any], *, ranks_only: tuple[int, ...] = (), type_check: bool = True) -> dict[str, Any]:
    """
    Given a state_dict, this API offload all the tensors to CPU memory.

    Args:
        state_dict (Dict[str, Any]): the target state_dict.
        pg (Optional[dist.ProcessGroup]): the process group that is used to
            gather ShardedTensor. Note that gathering a DTensor will use
            the DeviceMesh. So this argument will be ignored when gathering a
            DTensor.
        ranks_only: (Tuple[int, ...]): if this tuple is empty, all ranks will
            have the same state_dicts. Otherwise only ranks that in ``ranks_only``
            have the same state_dicts. Other ranks will get empty state_dicts.
        type_check: (bool): check if the instance data type is a supported type
            that can be saved by DCP.  The current supported data types are
            torch.Tensor, DTensor, int, float, str, list, dict, None.

    Returns:
        The gathered state dictionary.
    """
def _copy_state_dict(state_dict: dict[str, Any], copy_state_dict: dict[str, Any], non_blocking: bool = False, type_check: bool = True) -> dict[str, Any]:
    """
    Copies all tensors in a given state dict into a different state_dict with the
    same structure. Additionally, a copied state dict with the same value references
    is returned. Editing the keys on this state dict will not affect the
    passed in copy_state_dict (but the value references are the same).

    .. warning::
        It is expected by this function that state_dict and copy_state_dict share
        the same structure and data types.

    .. warning::
        The current supported data types are
            torch.Tensor, DTensor, int, float, str, list, dict, None.

    Args:
        state_dict (Dict[str, Any]): the target state_dict.
        copy_state_dict (Dict[str, Any]):
            The state dict we are copying into. This state_dict must have exactly
             the same structure as the source `state_dict`.
        non_blocking: (bool): Whether copy ops should be performed asynchronously
        type_check (bool): check if the instance data type is a supported type
            that can be saved by DCP. The current supported data types are
            torch.Tensor, DTensor, int, float, str, list, dict, None.

    Returns:
        State Dict copy
    """
def _create_cpu_state_dict(state_dict: dict[str, Any], pin_memory: bool = False, share_memory: bool = False) -> dict[str, Any]:
    """
    Given a state_dict, create another state_dict with the same structure and elements.
    However, all tensors in the returned state_dict are new tensors on CPU. These
    tensors can be placed on pin_memory or share_memory based on the provided arguments.

    .. warning::
        Setting both `pin_memory` and `share_memory` to True significantly increases the
        latency of this method because of the nuances which require us to register memory
        as pinned directly as opposed to relying on the pin_memory cache allocator. This
        option should only be used for long lived tensors which are required to be shared.
        This is not the case as long as at least one of `pin_memory` or `share_memory` is
         set to False.

    """
def _check_state_dict_similarity(state_dict: dict[str, Any], compared_state_dict: dict[str, Any]) -> bool:
    """
    Given two state_dicts, check if the structures are the same. And
    if a [key, tensor] pair exist in one state_dict there must be
    the a corresponding pait, [key, other_tensor], in the other state_dict,
    where tensor and other_tensor have the same size and dtype.

    Return the check result.
    """

class _TensorInfo(NamedTuple):
    size: torch.Size
    dtype: torch.dtype

def _broadcast_tensors(full_state_dict: dict[str, Any], local_state_dict: dict[str, Any], keys: list[str], device: torch.device, pg: dist.ProcessGroup | None = None) -> None: ...
def _distribute_tensors(local_state_dict: dict[str, Any], keys: list[str], device: torch.device, pg: dist.ProcessGroup | None = None) -> None: ...
def _broadcast_state_dict(full_state_dict: dict[str, Any], local_state_dict: dict[str, Any], device: torch.device, pg: dist.ProcessGroup | None = None, strict: bool = False, cpu_offload: bool = False) -> None: ...
def _distribute_state_dict(full_state_dict: dict[str, Any], local_state_dict: dict[str, Any], device: torch.device, pg: dist.ProcessGroup | None = None) -> None: ...
PATH_ITEM = str | int
OBJ_PATH = tuple[PATH_ITEM, ...]
FLATTEN_MAPPING = dict[str, OBJ_PATH]
STATE_DICT_TYPE = dict[str, Any]
CONTAINER_TYPE = MutableMapping[PATH_ITEM, Any]

def _traverse_state_dict(state_dict: STATE_DICT_TYPE, visitor: Callable[[OBJ_PATH, Any], None]) -> None:
    """
    Invoke ``visitor`` for each value recursively in ``state_dict``.
    Mapping, list, and tuple will be flattened and other value types are treated
    as the terminal values and will invoke ``visitor``.
    """
def _flatten_state_dict(state_dict: STATE_DICT_TYPE) -> tuple[STATE_DICT_TYPE, FLATTEN_MAPPING]:
    """
    Flatten ``state_dict`` made of nested dicts and lists into a top level dictionary.

    Use ``unflatten_state_dict`` to revert this process.
    Returns:
        A tuple with the flatten state_dict and a mapping from original to new state_dict.
    N.B. The new keys are derived from the object paths, joined by dot.
        For example: ``{ 'a': {'b':...}}`` results in the key `a.b`.
    """
def _set_element(root_dict: STATE_DICT_TYPE, path: OBJ_PATH, value: Any) -> None:
    """Set ``value`` in ``root_dict`` along the ``path`` object path."""
def _unflatten_state_dict(state_dict: STATE_DICT_TYPE, mapping: FLATTEN_MAPPING) -> STATE_DICT_TYPE:
    """Restore the original nested state_dict according to ``mapping`` and the flattened ``state_dict``."""
