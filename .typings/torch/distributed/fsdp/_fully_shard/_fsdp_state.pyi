import torch
import torch.nn as nn
from ._fsdp_api import MixedPrecisionPolicy as MixedPrecisionPolicy
from ._fsdp_common import TrainingState as TrainingState, _cast_fp_tensor as _cast_fp_tensor, compiled_autograd_enabled as compiled_autograd_enabled, detect_compiled_autograd as detect_compiled_autograd
from ._fsdp_param import FSDPParam as FSDPParam
from ._fsdp_param_group import FSDPCommContext as FSDPCommContext, FSDPParamGroup as FSDPParamGroup
from _typeshed import Incomplete
from collections.abc import Sequence
from torch._logging import warning_once as warning_once
from torch.autograd import Variable as Variable
from torch.autograd.graph import _MultiHandle as _MultiHandle
from torch.distributed._composable_state import _State as _State, _get_module_state as _get_module_state, _insert_module_state as _insert_module_state
from torch.distributed.device_mesh import _get_device_handle as _get_device_handle
from torch.distributed.utils import _apply_to_tensors as _apply_to_tensors, _to_kwargs as _to_kwargs
from torch.utils._pytree import tree_flatten as tree_flatten
from typing import Any, Callable

logger: Incomplete

class FSDPStateContext:
    """This has state shared across FSDP states."""
    all_states: list[FSDPState]
    iter_forward_root: FSDPState | None
    post_backward_final_callback_queued: bool
    is_last_backward: bool
    post_optim_event: torch.Event | None
    def __init__(self) -> None: ...

def disable_if_config_true(func): ...

class FSDPState(_State):
    _fsdp_param_group: FSDPParamGroup | None
    _is_root: bool | None
    _state_ctx: Incomplete
    _comm_ctx: Incomplete
    _training_state: TrainingState
    _states_to_forward_prefetch: list[FSDPState]
    _states_to_backward_prefetch: list[FSDPState]
    _modules_to_run_forward: set[nn.Module]
    _auto_reshard_after_forward: bool | None
    def __init__(self) -> None: ...
    _modules: Incomplete
    _device: Incomplete
    _device_handle: Incomplete
    _mp_policy: Incomplete
    _pre_forward_hook_handle: Incomplete
    _post_forward_hook_handle: Incomplete
    def init(self, modules: tuple[nn.Module, ...], device: torch.device, mp_policy: MixedPrecisionPolicy, auto_reshard_after_forward: bool) -> None: ...
    def _root_pre_forward(self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]: ...
    def _lazy_init(self) -> None:
        """
        Lazy initialization represents when all modules' parallelisms have
        finalized (e.g. FSDP has been applied to all desired modules). This
        means that we can determine which state is the root, and we do so by
        the 1st state to run forward.
        """
    def _init_shared_state(self) -> None: ...
    def _init_fqns(self) -> None:
        """Sets module and parameter FQN attributes for debugging."""
    @disable_if_config_true
    def _pre_forward(self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]: ...
    @disable_if_config_true
    def _post_forward(self, module: nn.Module, input: Any, output: Any) -> Any: ...
    def _pre_backward(self, grad: torch.Tensor) -> torch.Tensor: ...
    def _root_post_backward_final_callback(self) -> None: ...
    def _finalize_backward(self) -> None: ...
    def _register_pre_backward_hook(self, output: Any) -> Any: ...
    def _register_root_post_backward_final_callback(self) -> None: ...

def _get_module_fsdp_state(module: nn.Module) -> FSDPState | None: ...
def _register_group_forward_hooks(modules: Sequence[nn.Module], pre_hook: Callable, post_hook: Callable, modules_to_run: set[nn.Module]):
    """
    Registers group forward pre and post-hooks. The pre-hook runs upon the
    first module pre-forward, and the post-hook runs upon the last. If at least
    one module does not run forward, then the post-hook does not run.
    """
