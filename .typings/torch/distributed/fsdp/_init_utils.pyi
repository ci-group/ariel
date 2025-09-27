import torch
import torch.distributed as dist
import torch.nn as nn
from _typeshed import Incomplete
from collections.abc import Iterable, Iterator
from torch.distributed.algorithms._comm_hooks import default_hooks as default_hooks
from torch.distributed.device_mesh import DeviceMesh as DeviceMesh, _mesh_resources as _mesh_resources
from torch.distributed.distributed_c10d import _get_default_group as _get_default_group
from torch.distributed.fsdp._common_utils import TrainingState as TrainingState, _FSDPDeviceHandle as _FSDPDeviceHandle, _FSDPState as _FSDPState, _get_module_fsdp_state as _get_module_fsdp_state, _is_fsdp_flattened as _is_fsdp_flattened, _named_parameters_with_duplicates as _named_parameters_with_duplicates, clean_tensor_name as clean_tensor_name
from torch.distributed.fsdp._flat_param import FlatParamHandle as FlatParamHandle, FlatParameter as FlatParameter, HandleShardingStrategy as HandleShardingStrategy, _FSDP_USE_FULL_PREC_IN_EVAL as _FSDP_USE_FULL_PREC_IN_EVAL
from torch.distributed.fsdp._limiter_utils import _FreeEventQueue as _FreeEventQueue
from torch.distributed.fsdp.api import BackwardPrefetch as BackwardPrefetch, CPUOffload as CPUOffload, FullOptimStateDictConfig as FullOptimStateDictConfig, FullStateDictConfig as FullStateDictConfig, MixedPrecision as MixedPrecision, ShardingStrategy as ShardingStrategy, StateDictConfig as StateDictConfig, StateDictType as StateDictType
from torch.distributed.fsdp.wrap import _Policy as _Policy
from torch.distributed.tensor.parallel.fsdp import DTensorExtensions as DTensorExtensions
from torch.distributed.utils import _sync_params_and_buffers as _sync_params_and_buffers
from torch.utils._python_dispatch import is_traceable_wrapper_subclass as is_traceable_wrapper_subclass
from torch.utils.hooks import RemovableHandle as RemovableHandle
from typing import Any, Callable, no_type_check

_TORCHDISTX_AVAIL: bool
PARAM_BROADCAST_BUCKET_SIZE: Incomplete
FSDP_SYNCED: str
HybridShardProcessGroupType = tuple[dist.ProcessGroup, dist.ProcessGroup]
ProcessGroupType = dist.ProcessGroup | HybridShardProcessGroupType | None
SHARDING_STRATEGY_MAP: Incomplete
HYBRID_SHARDING_STRATEGIES: Incomplete
NO_RESHARD_AFTER_FORWARD_STRATEGIES: Incomplete

@no_type_check
def _init_process_group_state(state, process_group, sharding_strategy, policy, device_mesh=None): ...
@no_type_check
def _init_process_group_state_for_hybrid_shard(state, process_group, device_mesh): ...
@no_type_check
def _is_valid_hybrid_shard_pg_type(process_group): ...
@no_type_check
def _is_valid_hybrid_shard_device_mesh(device_mesh): ...
@no_type_check
def _init_intra_node_process_group(num_devices_per_node):
    """
    Return a process group across the current node.

    For example, given each row is a distinct node:
    0  1  2  3  4  5  6  7
    8  9 10 11 12 13 14 15
    This API would return an intra-node subgroup across
    [0, 1, ..., 7] or [8, 9, ..., 15] depending on the process's rank.
    For example, rank 3 would get [0, 1, ..., 7].
    """
@no_type_check
def _init_inter_node_process_group(global_process_group, num_devices_per_node):
    """
    Return an inter-node process group where each contained rank has the same local rank.

    For example, given each row is a distinct node:
    0  1  2  3  4  5  6  7
    8  9 10 11 12 13 14 15
    This API would return inter-node process group [0, 8], [1, 9], [2, 10], and so forth
    depending on the process's rank. For example, rank 1 would get [1, 9], rank 5
    would get [5, 13].
    """
def _init_intra_and_inter_node_groups(global_process_group: dist.ProcessGroup, num_devices_per_node: int) -> tuple[dist.ProcessGroup, dist.ProcessGroup]:
    """
    Initialize intra and inter-node process groups and return the ones corresponding to this process's rank.

    This function can be used to initialize process groups for ``HYBRID_SHARD`` or
    ``_HYBRID_SHARD_ZERO2`` in FSDP.
    This function assumes each node has an equal number of CUDA-enabled devices.
    Returns:
        Tuple[dist.ProcessGroup, dist.ProcessGroup]: Intra and inter-node process group.
    """
@no_type_check
def _init_ignored_module_states(state, module, ignored_modules, ignored_states=None): ...
def _check_ignored_states(ignored_states: list[Any], passed_as_ignored_states: bool) -> None:
    """
    Check that the ignored states are uniformly parameters or uniformly modules.

    We may remove this check in the future if we permit mixing.
    """
@no_type_check
def _init_device_handle(state, module, ignored_params, device_id):
    """
    Determine device handle used for initializing FSDP.

    If a device is specified by ``device_id``,
    then returns device handle corresponds to that device type. Otherwise, If the
    module is already on a non-CPU device, then the device type is that non-CPU device type.
    If the module is on CPU or meta, then the device type is the current accelerator device.
    See the :ref:`Accelerators<accelerators>` for details.


    This method will be called once ignored parameters was determined, as the device handle maybe needed
    for other initialization.
    """
@no_type_check
def _init_buffer_state(state, module): ...
@no_type_check
def _init_core_state(state, sharding_strategy, mixed_precision, cpu_offload, limit_all_gathers, use_orig_params, backward_prefetch_limit, forward_prefetch_limit): ...
@no_type_check
def _init_runtime_state(state): ...
@no_type_check
def _init_prefetching_state(state, backward_prefetch, forward_prefetch): ...
@no_type_check
def _init_extension(state, device_mesh=None): ...
@no_type_check
def _init_state_dict_state(state): ...
def _verify_managed_params(module: nn.Module, params: list[nn.Parameter]) -> None:
    """
    Verify if the parameters are accepted by FSDP. The only restriction now
    is that the parameter cannot be a scalar tensor (param.shape == []).
    """
@no_type_check
def _init_param_handle_from_module(state, fully_sharded_module, device_id, param_init_fn, sync_module_states):
    """Initialize a ``FlatParamHandle`` from a module ``fully_sharded_module``."""
@no_type_check
def _init_param_handle_from_params(state, params, fully_sharded_module) -> None: ...
def _get_ignored_modules(root_module: nn.Module, _ignored_modules: Iterable[torch.nn.Module] | None) -> set[nn.Module]:
    """
    Check that ``_ignored_modules`` is an iterable of ``nn.Module`` s without any FSDP instances.

    Return the modules contained in their module
    subtrees as a :class:`set`. Nested FSDP instances are excluded, but their
    already-computed ignored modules are included.

    ``_ignored_modules`` represents the argument passed by the user to FSDP.
    """
def _get_ignored_params(root_module: torch.nn.Module, ignored_modules: set[torch.nn.Module], ignored_parameters: Iterable[torch.nn.Parameter] | None = None) -> set[torch.nn.Parameter]:
    """
    Return the parameters of the modules in ``ignored_modules`` and the parameters in ``ignored_parameters``.

    :class:`FlatParameter` s are excluded from the result.
    """
def _get_ignored_buffer_names(root_module: torch.nn.Module, ignored_modules: set[torch.nn.Module]) -> set[str]:
    """Return the cleaned buffer FQNs in ``ignored_modules``."""
def _get_buffer_names(root_module: nn.Module) -> set[str]:
    """Return the fully prefixed names of all buffers in the module hierarchy rooted at ``root_module`` as a class:`set`."""
def _check_single_device_module(module: nn.Module, ignored_params: set[nn.Parameter], device_id: int | torch.device | None) -> None:
    """
    Raise an error if ``module`` has original parameters on multiple devices, ignoring the parameters in ``ignored_params``.

    Thus, after this method, the
    module must be either fully on the CPU or fully on a non-CPU device.
    """
def _get_device_from_device_id(device_id: int | torch.device | None, rank: int, device_handle: _FSDPDeviceHandle) -> torch.device | None:
    """
    Return a ``torch.device`` for the specified ``device_id``.

    Processes ``device_id`` and returns either the corresponding device or
    ``None`` if ``device_id`` is ``None``.
    """
def _need_to_materialize_module(module: nn.Module, ignored_params: set[nn.Parameter], ignored_modules: set[nn.Module]) -> tuple[bool, bool]:
    """
    Return if ``module`` has parameters on meta device and if ``module`` is using torchdistX deferred initialization.

    At most of the returned bools can
    be ``True``. If either is ``True``, then ``module`` needs to be
    materialized.
    """
def _materialize_with_param_init_fn(root_module: nn.Module, param_init_fn: Callable[[nn.Module], None], ignored_modules: set[nn.Module]) -> None: ...
def _materialize_meta_module(root_module: nn.Module, device_from_device_id: torch.device | None, ignored_modules: set[nn.Module], device_handle: _FSDPDeviceHandle): ...
def _get_modules_to_materialize(root_module: nn.Module, ignored_modules: set[nn.Module]) -> list[nn.Module]: ...
def _move_module_to_device(module: nn.Module, ignored_params: set[nn.Parameter], ignored_buffers: set[torch.Tensor], device_from_device_id: torch.device | None) -> None:
    """
    Move ``module`` depending on ``device_from_device_id`` and its current device.

    This includes moving ignored modules' parameters.

    - If ``device_from_device_id`` is not ``None``, then this moves
    ``module`` to the device.
    - If ``device_from_device_id`` is ``None``, then this does not move
    ``module`` but warns the user if it is on CPU.

    Precondition: ``_check_single_device_module()``.
    """
def _move_states_to_device(params: list[nn.Parameter], buffers: list[torch.Tensor], device_from_device_id: torch.device | None) -> None:
    """
    Move states to the specified device.

    Precondition: ``_check_single_device_module()`` and module's parameters and
    buffers have been materialized if needed.
    """
def _warn_cpu_init() -> None: ...
def _get_compute_device(module: nn.Module, ignored_params: set[nn.Parameter], device_from_device_id: torch.device | None, rank: int, device_handle: _FSDPDeviceHandle) -> torch.device:
    """
    Determine and return this FSDP instance's compute device.

    If the module is already on a non-CPU device, then the compute device is that non-CPU
    device. If the module is on CPU, then the compute device is the current
    device.

    Since this method should be called after materializing the module, any
    non-CPU device should not be meta device. For now, the compute device is
    always a CUDA or CUDA-like device with its explicit index.

    Precondition: ``_check_single_device_module()`` and
    ``_move_module_to_device()``.
    """
def _sync_module_params_and_buffers(module: nn.Module, params: list[nn.Parameter], process_group: dist.ProcessGroup) -> None:
    """
    Synchronize module states (i.e. parameters ``params`` and all not-yet-synced buffers) by broadcasting from rank 0 to all ranks.

    Precondition: ``sync_module_states == True`` and ``self.process_group`` has
    been set.
    """
def _check_module_states_for_sync_module_states(module_states: list[torch.Tensor]) -> None: ...
def _get_orig_params(module: nn.Module, ignored_params: set[nn.Parameter]) -> Iterator[nn.Parameter]:
    """
    Return an iterator over the original parameters in ``module``.

    The iterator does not return
    the parameters in ``ignored_params``, any ``FlatParameter`` s (which may be
    present due to nested FSDP wrapping), or any original parameters already
    flattened (only relevant when ``use_orig_params=True``).
    """
def _check_orig_params_flattened(fsdp_module, ignored_params: set[nn.Parameter]) -> None:
    """
    Check that original parameters in ``fsdp_module`` have been flattened.

    The flattened parameters are made
    invisible to ``named_parameters()`` for the module hierarchy rooted at
    ``fsdp_module``. This should be called as a sanity check after flattening
    the wrapped module's parameters.
    """
def _get_default_comm_hook(sharding_strategy: ShardingStrategy): ...
def _get_default_comm_hook_state(process_group: dist.ProcessGroup) -> default_hooks.DefaultState: ...
