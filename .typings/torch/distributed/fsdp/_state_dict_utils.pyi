import contextlib
import torch.nn as nn
from ._fsdp_extensions import _ext_all_gather_dtensor as _ext_all_gather_dtensor, _ext_chunk_dtensor as _ext_chunk_dtensor, _ext_chunk_tensor as _ext_chunk_tensor, _ext_post_unflatten_transform as _ext_post_unflatten_transform, _ext_pre_load_state_dict_transform as _ext_pre_load_state_dict_transform
from ._unshard_param_utils import FLAT_PARAM as FLAT_PARAM, _unshard_fsdp_state_params as _unshard_fsdp_state_params
from _typeshed import Incomplete
from collections.abc import Generator, Iterator
from torch.distributed._shard.sharded_tensor import Shard as Shard, ShardedTensor as ShardedTensor, init_from_local_shards as init_from_local_shards
from torch.distributed.device_mesh import _mesh_resources as _mesh_resources
from torch.distributed.fsdp._common_utils import FSDP_PREFIX as FSDP_PREFIX, FSDP_WRAPPED_MODULE as FSDP_WRAPPED_MODULE, _FSDPState as _FSDPState, _get_module_fsdp_state_if_fully_sharded_module as _get_module_fsdp_state_if_fully_sharded_module, _has_fsdp_params as _has_fsdp_params, _is_composable as _is_composable, _module_handle as _module_handle, clean_tensor_name as clean_tensor_name
from torch.distributed.fsdp._debug_utils import SimpleProfiler as SimpleProfiler
from torch.distributed.fsdp._runtime_utils import _cast_buffers_to_dtype_and_device as _cast_buffers_to_dtype_and_device, _get_orig_buffer_dtypes as _get_orig_buffer_dtypes, _lazy_init as _lazy_init, _reset_flat_param_grad_info_if_needed as _reset_flat_param_grad_info_if_needed
from torch.distributed.fsdp.api import FullStateDictConfig as FullStateDictConfig, ShardingStrategy as ShardingStrategy, StateDictType as StateDictType
from torch.distributed.tensor import DTensor as DTensor
from torch.distributed.utils import _replace_by_prefix as _replace_by_prefix
from typing import Any, no_type_check

logger: Incomplete

def _should_unshard_params(fsdp_state: _FSDPState) -> bool: ...
def _convert_to_wrapped_module_name(module_name: str) -> str: ...
def _param_name_infos(module: nn.Module, fsdp_state: _FSDPState) -> Iterator[tuple[str, str, str]]: ...
def _shared_param_name_infos(module: nn.Module, fsdp_state) -> Iterator[tuple[str, str, str]]: ...
@no_type_check
def _enter_unshard_params_ctx(module, fsdp_state, writeback: bool = False, rank0_only: bool = False, offload_to_cpu: bool = False, with_grads: bool = False) -> None:
    """
    state_dict hooks cannot use the pure context call as the checkpoint flow
    requires to enter the context in the pre-hook but leave the context in the
    post-hook. This API enters the context of ``_unshard_fsdp_state_params``.
    """
@no_type_check
def _exit_unshard_params_ctx(module, fsdp_state) -> None:
    """A helper function to exit ``_unshard_fsdp_state_params`` context."""
def _common_pre_state_dict_hook(module: nn.Module, fsdp_state: _FSDPState) -> None:
    """Performs the pre-state_dict tasks shared by all state_dict types."""
def _common_unshard_pre_state_dict_hook(module: nn.Module, fsdp_state: _FSDPState, offload_to_cpu: bool, rank0_only: bool) -> None:
    """
    Performs the pre-state_dict tasks shared by all state_dict types that require
    ``_unshard_fsdp_state_params()``. FULL_STATE_DICT and SHARDED_STATE_DICT use this hook.
    """
@no_type_check
def _common_unshard_post_state_dict_hook(module, fsdp_state, state_dict, prefix, param_hook):
    """
    The post-state_dict flow that shared by all state_dict types that require
    ``_unshard_fsdp_state_params()``. FULL_STATE_DICT and SHARDED_STATE_DICT use this
    hook.
    """
@no_type_check
def _full_pre_state_dict_hook(fsdp_state, module, *args, **kwargs) -> None:
    """
    Hook that runs before model.state_dict() is called. pre-state_dict hook is
    not actually supported by ``nn.Module``. As a result, this API is called
    from ``_full_post_state_dict_hook()`` to simulate the case. Once pre-state_dict
    is supported in ``nn.Module``, this hook will be registered as a hook in
    ``nn.Module``.
    """
@no_type_check
def _full_post_state_dict_hook(module, fsdp_state, state_dict, prefix):
    """
    Hook that runs after model.state_dict() is called before returning result to
    user. For FSDP, we may have to clone the tensors in state_dict as params go
    back to sharded version after _unshard_fsdp_state_params ends, and also remove
    the ``FSDP_WRAPPED_MODULE`` prefix.
    """
def _full_pre_load_state_dict_hook(module: nn.Module, fsdp_state: _FSDPState, state_dict: dict[str, Any], prefix: str) -> None: ...
def _full_post_load_state_dict_hook(module: nn.Module, fsdp_state: _FSDPState, *args, **kwargs) -> None: ...
def _local_pre_state_dict_hook(fsdp_state: _FSDPState, module: nn.Module, *args, **kwargs) -> None:
    """
    Hook that runs before model.state_dict() is called. Right now, pre-state_dict
    hook is not supported by the PyTorch core. So this API is called from
    `_local_post_state_dict_hook()` to simulate the case.
    """
@no_type_check
def _local_post_state_dict_hook(module, fsdp_state, state_dict, prefix):
    '''
    This hook create a ShardedTensor from the local flat_param and replace
    the state_dict[f"{prefix}{FLAT_PARAM}] with the ShardedTensor. No copy
    will happen. The underlying storage is the same.
    '''
def _local_post_load_state_dict_hook(module: nn.Module, fsdp_state: _FSDPState, *args, **kwargs) -> None: ...
def _local_pre_load_state_dict_hook(module: nn.Module, fsdp_state: _FSDPState, state_dict: dict[str, Any], prefix: str) -> None:
    """
    This hook finds the local flat_param for this FSDP module from the
    state_dict. The flat_param should be a ShardedTensor. This hook converts
    the ShardedTensor to a tensor. No copy happen unless padding is required.
    """
def _sharded_pre_state_dict_hook(fsdp_state: _FSDPState, module: nn.Module, *args, **kwargs) -> None:
    """
    Hook that runs before model.state_dict() is called. Check
    ``_full_pre_load_state_dict_hook`` for the detail.
    """
@no_type_check
def _sharded_post_state_dict_hook(module, fsdp_state, state_dict, prefix):
    """
    The hook replaces the unflattened, unsharded parameter in the state_dict
    with a unflattened, sharded parameter (a ShardedTensor).
    """
@no_type_check
def _sharded_post_load_state_dict_hook(module, fsdp_state, *args, **kwargs) -> None: ...
@no_type_check
def _sharded_pre_load_state_dict_hook(module, fsdp_state, state_dict, prefix) -> None:
    """
    The hook combines the unflattened, sharded parameters (ShardedTensor) to
    a new FlatParameter and shards the new FlatParameter to the local chunk.
    """
@contextlib.contextmanager
def _replace_with_full_state_dict_type(fsdp_state: _FSDPState) -> Generator: ...
@no_type_check
def _post_state_dict_hook(module, state_dict, prefix, *args):
    """
    _post_state_dict_hook() is called after the state_dict() of this
    FSDP module is executed. ``fsdp_state._state_dict_type`` is used to decide
    what postprocessing will be done.
    """
@no_type_check
def _pre_state_dict_hook(module, *args, **kwargs) -> None:
    """
    This is called before the core state dict saving logic of ``module``.
    ``fsdp_state._state_dict_type`` is used to decide what postprocessing will
    be done.
    """
@no_type_check
def _set_use_dtensor(fsdp_state) -> None: ...
@no_type_check
def _pre_load_state_dict_hook(module, state_dict, prefix, *args) -> None:
    """
    This is called before ``module._load_from_state_dict()``.
    ``fsdp_state._state_dict_type`` is used to decide what preprocessing will
    be done.
    """
@no_type_check
def _post_load_state_dict_hook(module, incompatible_keys, *args) -> None: ...
def _register_all_state_dict_hooks(state: _FSDPState):
    """
    Registers pre-save, post-save, pre-load, and post-load state dict hooks.
    """
@no_type_check
def _register_state_dict_hooks_base(state, hook_registration_fn_name, hook, hook_registration_fn_kwargs) -> None:
    """Registers ``hook`` using ``hook_registration_fn``."""
