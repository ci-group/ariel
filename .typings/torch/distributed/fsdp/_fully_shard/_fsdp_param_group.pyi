import contextlib
import torch
import torch.distributed as dist
import torch.nn as nn
from ._fsdp_api import CPUOffloadPolicy as CPUOffloadPolicy, MixedPrecisionPolicy as MixedPrecisionPolicy, OffloadPolicy as OffloadPolicy
from ._fsdp_collectives import AllGatherResult as AllGatherResult, foreach_all_gather as foreach_all_gather, foreach_all_gather_copy_out as foreach_all_gather_copy_out, foreach_reduce as foreach_reduce
from ._fsdp_common import FSDPMeshInfo as FSDPMeshInfo, HSDPMeshInfo as HSDPMeshInfo, TrainingState as TrainingState, compiled_autograd_enabled as compiled_autograd_enabled
from ._fsdp_param import FSDPParam as FSDPParam, ParamModuleInfo as ParamModuleInfo, ShardedState as ShardedState
from _typeshed import Incomplete
from torch.distributed.device_mesh import _get_device_handle as _get_device_handle
from torch.distributed.fsdp._common_utils import _named_parameters_with_duplicates as _named_parameters_with_duplicates
from torch.distributed.tensor import Shard as Shard
from torch.profiler import record_function as record_function
from torch.utils._pytree import tree_flatten as tree_flatten, tree_unflatten as tree_unflatten
from torch.utils.hooks import RemovableHandle as RemovableHandle
from typing import Any, Callable, NamedTuple

logger: Incomplete
_ModuleToHandleDict = dict[nn.Module, RemovableHandle]

class FSDPCommContext:
    """This has the communication state shared across FSDP states/parameter groups."""
    device_handle: Incomplete
    all_gather_copy_in_stream: Incomplete
    all_gather_stream: Incomplete
    reduce_scatter_stream: Incomplete
    all_reduce_stream: Incomplete
    all_gather_state: AllGatherState | None
    reduce_scatter_state: ReduceScatterState | None
    post_forward_order: list[FSDPParamGroup]
    def lazy_init(self, device: torch.device): ...
    def get_all_gather_streams(self, async_op: bool, training_state: TrainingState) -> tuple[torch.Stream, torch.Stream]: ...

class AllGatherState(NamedTuple):
    all_gather_result: AllGatherResult
    event: torch.Event | None

class ReduceScatterState(NamedTuple):
    reduce_scatter_input: torch.Tensor
    event: torch.Event | None

class AllReduceState(NamedTuple):
    all_reduce_input: torch.Tensor
    event: torch.Event | None

class FSDPParamGroup:
    """This class represents a parameter group to communicate together."""
    _orig_dtype: torch.dtype | None
    _reduce_dtype: torch.dtype | None
    modules: Incomplete
    fsdp_params: Incomplete
    mesh_info: Incomplete
    post_forward_mesh_info: Incomplete
    device: Incomplete
    device_handle: Incomplete
    mp_policy: Incomplete
    offload_policy: Incomplete
    _training_state: Incomplete
    _sharded_state: Incomplete
    _module_fqn: str | None
    _reset_sharded_params: bool
    _module_to_pre_save_state_dict_hook_handle: _ModuleToHandleDict
    _module_to_pre_load_state_dict_hook_handle: _ModuleToHandleDict
    _all_reduce_hook: Callable[[torch.Tensor], None] | None
    _all_reduce_hook_stream: torch.cuda.Stream | None
    comm_ctx: Incomplete
    _post_forward_indices: list[int]
    reduce_grads: bool
    all_reduce_grads: bool
    reshard_after_backward: bool
    gradient_divide_factor: float | None
    force_sum_reduction_for_comms: bool
    unshard_async_op: bool
    unshard_in_backward: bool
    allocate_memory_from_process_group: bool
    _all_gather_result: AllGatherResult | None
    _post_reduce_event: torch.Event | None
    _reshard_after_forward_event: torch.Event | None
    _partial_reduce_output: torch.Tensor | None
    _all_reduce_state: AllReduceState | None
    def __init__(self, params: list[nn.Parameter], modules: tuple[nn.Module, ...], mesh_info: FSDPMeshInfo, post_forward_mesh_info: FSDPMeshInfo | None, device: torch.device, shard_placement_fn: Callable[[nn.Parameter], Shard | None] | None, mp_policy: MixedPrecisionPolicy, offload_policy: OffloadPolicy) -> None: ...
    def _init_mp_dtypes(self) -> None: ...
    def lazy_init(self) -> None: ...
    def unshard(self, async_op: bool = False): ...
    def wait_for_unshard(self) -> None:
        """
        1. In forward with implicit prefetching, to overlap the current copy-out
        with the next all-gather, we save a reference to the current all-gather
        result to free after the next copy-out.
        2. Otherwise (explicit prefetching or in backward), we free the
        all-gather result immediately after the current copy-out since we can
        already overlap the current copy-out with the previous reduce-scatter.
        """
    def _wait_all_gather_streams_on_event(self, event: torch.Event | None): ...
    def reshard(self) -> None: ...
    def pre_forward(self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]: ...
    def post_forward(self, module: nn.Module, input: Any, output: Any): ...
    def _record_post_forward(self) -> None: ...
    def pre_backward(self, default_prefetch: bool, *unused: Any): ...
    def post_backward(self, *unused: Any): ...
    def finalize_backward(self) -> None: ...
    def _wait_for_post_backward(self) -> None: ...
    def _backward_prefetch(self) -> None: ...
    @staticmethod
    def _prefetch_unshard(target_fsdp_param_group: FSDPParamGroup, pass_type: str) -> None: ...
    def _to_sharded(self) -> None: ...
    def _to_sharded_post_forward(self) -> None: ...
    def _to_unsharded(self) -> None: ...
    @property
    def is_sharded(self) -> bool: ...
    @property
    def is_sharded_post_forward(self) -> bool: ...
    @property
    def is_unsharded(self) -> bool: ...
    @contextlib.contextmanager
    def use_training_state(self, training_state: TrainingState): ...
    def _register_post_backward_hook(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]: ...
    def _register_state_dict_hooks(self) -> None: ...
    @property
    def _reshard_after_forward(self) -> bool: ...
    @property
    def _use_post_forward_mesh(self) -> bool: ...
    @property
    def _is_hsdp(self) -> bool: ...
    @property
    def _all_gather_process_group(self) -> dist.ProcessGroup: ...
    @property
    def _reduce_scatter_process_group(self) -> dist.ProcessGroup: ...
    @property
    def _all_reduce_process_group(self) -> dist.ProcessGroup: ...
    def _with_fqn(self, label: str) -> str: ...
    def __repr__(self) -> str: ...
    def _validate_no_meta_params(self) -> None: ...
    def _validate_cpu_offload_params(self) -> None: ...

def _get_param_module_infos(params: list[nn.Parameter], modules: tuple[nn.Module, ...]) -> list[ParamModuleInfo]:
    """
    Shared parameter: lin1.weight = lin2.weight
    Shared module: mlp.lin1 = mlp.lin2
    We do not remove duplicates when traversing both modules and parameters to
    find shared modules' parameters and shared parameters within a module.
    """

class RegisterPostBackwardFunction(torch.autograd.Function):
    @staticmethod
    def _assert_not_tracing_fsdp() -> None: ...
    @staticmethod
    def forward(ctx, param_group: FSDPParamGroup, *inputs: torch.Tensor): ...
    @staticmethod
    def backward(ctx, *grads: torch.Tensor): ...
