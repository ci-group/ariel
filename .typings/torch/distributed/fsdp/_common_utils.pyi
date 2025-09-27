import torch
import torch.distributed as dist
import torch.distributed.fsdp._flat_param as flat_param_file
import torch.nn as nn
import weakref
from ._flat_param import FlatParamHandle as FlatParamHandle
from .api import FullOptimStateDictConfig as FullOptimStateDictConfig, FullStateDictConfig as FullStateDictConfig, OptimStateDictConfig as OptimStateDictConfig, ShardingStrategy as ShardingStrategy, StateDictConfig as StateDictConfig, StateDictType as StateDictType
from _typeshed import Incomplete
from collections.abc import Generator, Iterable
from enum import Enum
from torch.distributed._composable_state import _State as _State, _get_module_state as _get_module_state
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import _CHECKPOINT_PREFIX as _CHECKPOINT_PREFIX
from torch.distributed.device_mesh import DeviceMesh as DeviceMesh
from torch.distributed.fsdp._fsdp_extensions import FSDPExtensions as FSDPExtensions
from torch.distributed.utils import _apply_to_tensors as _apply_to_tensors
from torch.utils._mode_utils import no_dispatch as no_dispatch
from typing import Any, Callable, no_type_check

FSDP_WRAPPED_MODULE: str
FSDP_PREFIX: Incomplete
FSDP_FLATTENED: str
_MODULE_TO_INP_DTYPE: weakref.WeakKeyDictionary

class _FSDPDeviceHandle:
    """
    This is a simple abstraction for FSDP computing devices,
    which enables custom backends that implement CUDA-like
    semantics to be integrated with FSDP.
    """
    __backend: Incomplete
    __device: Incomplete
    def __init__(self, device: torch.device, backend: Any = None) -> None: ...
    @classmethod
    def from_device(cls, device: torch.device) -> _FSDPDeviceHandle:
        """
        Return a device handle corresponding to the device, and through this handle,
        operations with the same semantics as CUDA can be performed on the device.
        Just return torch.cuda if the device is cuda to make attribute-access faster.
        Custom backend must first register a module with the same name with {device.type} on torch.
        """
    def __getattr__(self, name: str) -> Any: ...

class _UninitializedDeviceHandle(_FSDPDeviceHandle):
    def __init__(self) -> None: ...
    def __getattribute__(self, name: str) -> Any: ...

class _FSDPState(_State):
    _ignored_modules: set[nn.Module]
    _ignored_params: set[nn.Parameter]
    _ignored_buffer_names: set[str]
    process_group: dist.ProcessGroup | None
    rank: int
    world_size: int
    _device_mesh: DeviceMesh | None
    sharding_strategy: Incomplete
    _use_orig_params: bool
    training_state: Incomplete
    _unshard_params_ctx: dict[nn.Module, Generator]
    _state_dict_type: StateDictType
    _state_dict_config: StateDictConfig
    _optim_state_dict_config: OptimStateDictConfig
    _is_root: bool | None
    _handle: flat_param_file.FlatParamHandle | None
    _fully_sharded_module_to_handle: dict[nn.Module, flat_param_file.FlatParamHandle | None]
    compute_device: torch.device | None
    _gradient_predivide_factor: int
    _gradient_postdivide_factor: int
    _comm_hook: Callable | None
    _comm_hook_state: Any | None
    _unshard_event: torch.Event | None
    _device_handle: _FSDPDeviceHandle
    _all_fsdp_states: list[_FSDPState]
    _all_handles: list[flat_param_file.FlatParamHandle]
    _fsdp_extension: FSDPExtensions | None
    def __init__(self) -> None: ...

def _get_module_fsdp_state(module: nn.Module) -> _FSDPState | None: ...
def _get_module_fsdp_state_if_fully_sharded_module(module: nn.Module) -> _FSDPState | None: ...

class TrainingState(Enum):
    """
    An enum that indicates the state of a ``FullyShardedDataParallel` instance.
    """
    IDLE = ...
    FORWARD_BACKWARD = ...
    SUMMON_FULL_PARAMS = ...

class HandleTrainingState(Enum):
    """
    An enum that indicates the state of a ``FlatParamHandle`.
    """
    IDLE = ...
    FORWARD = ...
    BACKWARD_PRE = ...
    BACKWARD_POST = ...
    SUMMON_FULL_PARAMS = ...

def _is_composable(state: _FSDPState): ...
@no_type_check
def _module_handle(state, module):
    """
    Returns the ``FlatParamHandle`` s corresponding to ``module``. This is
    the handle that contains some parameter in ``module``.
    """
@no_type_check
def _has_fsdp_params(state, module):
    """Returns if ``module`` has parameters managed by FSDP."""
def _get_sharding_strategy(handle):
    """
    Returns the sharding strategy of the handle.
    """
def clean_tensor_name(tensor_name: str) -> str:
    """
    Cleans the parameter or buffer name by removing any module wrapper
    prefixes.
    """
def _set_fsdp_flattened(tensor: torch.Tensor) -> None:
    """
    Sets an attribute on ``tensor`` to mark it as flattened by FSDP. This is to
    avoid re-flattening it during nested construction.
    """
def _is_fsdp_flattened(tensor: torch.Tensor) -> bool:
    """Returns if ``tensor`` has been marked as flattened by FSDP."""
def _named_parameters_with_duplicates(module: nn.Module, **kwargs: Any) -> list[tuple[str, nn.Parameter]]:
    """
    This API is required as some modules overwrite `named_parameters()` but do not support
    `remove_duplicate`.
    """
def _get_param_to_fqns(model: torch.nn.Module, dedup_shared_params: bool = True) -> dict[nn.Parameter, list[str]]:
    '''
    Constructs a mapping from parameter to a list of its "canonical" FQNs. Here,
    we use canonical to mean the fully-qualified name assigned to the parameter
    based on its position in the original nn.Module hierarchy before any wrapper
    or parallelism has been applied to it. This is in contrast to FQNs that may be
    generated after parallelisms or wrappers have been applied to the model.

    Each normal parameter maps to a singleton list containing its FQN, while each
    ``FlatParameter`` maps to a list of its original parameter FQNs, which may
    have length greater than one.  All FQNs are prefixed starting from ``model``.

    In the case where FSDP was applied with ``use_orig_params=True``, there should be no
    ``FlatParameter`` s registered to the model\'s modules and this mapping will only
    contain mappings from ``nn.Parameter`` s to singleton FQN lists.

    It is only in the case where FSDP was applied with ``use_orig_params=False`` where
    a ``FlatParameter`` will be registered in place of the original parameters and there
    will be mappings from each ``FlatParameter`` to lists of FQNs corresponding to the
    original parameters.

    Args:
        model (torch.nn.Module): Root module (which may or may not be a
            :class:`FullyShardedDataParallel` instance).
        dedup_shared_params (bool): For shared parameters, if ``True``, only
            includes the FQNs corresponding to the first encounter of the
            shared parameter in the module traversal; if ``False``, then
            includes the FQNs across all encounters. (Default: ``True``)
    '''
@no_type_check
def _log_post_backward_hook(state, handle, logger) -> None: ...
@no_type_check
def _get_handle_fqns_from_root(state, handle): ...
def _apply_to_modules(root_module: torch.nn.Module, module_fn: Callable, return_fn: Callable, filter_fqns: list[str] | None = None, *args, **kwargs):
    '''
    Performs a pre-order traversal of the modules in the hierarchy rooted at
    ``root_module``, applying ``module_fn`` at each module and finally
    returning a value using ``return_fn``. The traversal constructs the full
    module prefix name (e.g. "module.submodule." just like in model state dict)
    and makes that available to ``module_fn``.

    ``filter_fqns`` is used because some module may have its own prefix similar
    to ``FullyShardedDataParallel`` and the ``named_parameters()`` is overwritten
    to remove the prefix.
    '''
@no_type_check
def _assert_in_training_states(state, training_states) -> None:
    """Asserts that FSDP is in the states ``_training_states``."""
def _get_root_modules(modules: set[nn.Module]) -> set[nn.Module]:
    """
    Returns:
        Set[nn.Module]: The subset of ``modules`` that are root modules (i.e.
        parent-less) with respect to the modules in the set itself. In other
        words, these are the modules in ``modules`` that are not the child of
        any other module in ``modules``.
    """
def _override_module_mixed_precision(root: torch.nn.Module, module_classes_to_override: Iterable[type[nn.Module]], wrap_override_dict: dict[str, Any] = {'mixed_precision': None}) -> set[type[nn.Module]]: ...
def _no_dispatch_record_stream(tensor: torch.Tensor, stream: torch.Stream) -> None: ...
