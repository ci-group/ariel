import torch
from _typeshed import Incomplete
from enum import Enum
from torch import nn, optim
from torch.distributed._tools.mem_tracker import MemTracker, _RefType, _State
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
from torch.utils.weak import WeakIdKeyDictionary
from typing import Any, Callable, NamedTuple, TypeVar
from typing_extensions import ParamSpec, TypeVarTuple, Unpack

__all__ = ['FSDPMemTracker']

_P = ParamSpec('_P')
_R = TypeVar('_R')
_Ts = TypeVarTuple('_Ts')

class _FSDPRefType(_RefType):
    """
    Enumerates categories of memory usage in FSDP modules, including parameters, gradients, activations,
    and optimizer states.

    Attributes:
        SHARDED_PARAM (str): Memory usage of sharded parameters.
        UNSHARDED_PARAM (str): Memory usage of unsharded parameters.
        SHARDED_GRAD (str): Memory usage of sharded gradients corresponding to the sharded parameters.
        UNSHARDED_GRAD (str): Memory usage of unsharded gradients corresponding to the unsharded parameters.
        ACT (str): Memory usage of activations and tensors from forward and AC recomputation.
        TEMP (str): Memory usage of temporary tensors during the backward pass including gradients of activations.
        ALL_GATHER (str): Memory usage of all_gather output tensor.
        REDUCE_SCATTER (str): Memory usage of reduce_scatter input tensor.
        OPT (str): Memory usage of tensors storing optimizer states.
        INP (str): Memory usage of input tensors.
    """
    SHARDED_PARAM = 'Sharded Param'
    UNSHARDED_PARAM = 'Unsharded Param'
    BUFFER = 'Buffer'
    SHARDED_GRAD = 'Sharded Grad'
    UNSHARDED_GRAD = 'Unsharded Grad'
    ACT = 'Activation'
    TEMP = 'Temp'
    ALL_GATHER = 'All Gather'
    REDUCE_SCATTER = 'Reduce Scatter'
    OPT = 'OptState'
    INP = 'Inputs'

class _SavedFSDPMethods(NamedTuple):
    pre_backward: Callable
    post_backward: Callable

class _FSDPModState(_State):
    """
    Enumerates the states of FSDP modules during the forward and backward passes.
    """
    BEF_PRE_FW = 'Before Pre-Forward'
    AFT_PRE_FW = 'After Pre-Forward'
    BEF_POST_FW = 'Before Post-Forward'
    AFT_POST_FW = 'After Post-Forward'
    BEF_PRE_BW = 'Before Pre-Backward'
    AFT_PRE_BW = 'After Pre-Backward'
    BEF_POST_BW = 'Before Post-Backward'
    AFT_POST_BW = 'After Post-Backward'
    PRE_FW_AC = 'Pre-Forward AC'
    POST_FW_AC = 'Post-Forward AC'
    PEAK_FW = 'Peak Forward'
    PEAK_BW = 'Peak Backward'

class _FSDPModMemStats:
    """
    A class to store the memory statistics of an FSDP module.

    Args:
        mod_fqn (str): The fully qualified name of the FSDP module.

    Attributes:
        snapshots (Dict[_FSDPModState, Dict[torch.device, Dict[str, int]]]): A dictionary of memory snapshots
        of the module at different states as defined by ``_FSDPModState``. Each key is a device, and
        each value is another dictionary with keys as memory reference types defined by ``_FSDPRefType`` and
        values as the memory consumed in bytes.

    """
    mod_fqn: Incomplete
    local_peak: dict[torch.device, int]
    snapshots: dict[_FSDPModState, list[dict[torch.device, dict[str, int]]]]
    def __init__(self, mod_fqn: str) -> None: ...

class _FSDPState(Enum):
    PRE_FW = ...
    FW = ...
    POST_FW = ...
    PRE_BW = ...
    BW = ...
    POST_BW = ...

class FSDPMemTracker(MemTracker):
    '''
    A ``TorchDispatchMode`` based context manager that extends ``torch.distributed._tools.mem_tracker.MemTracker`` to track
    and categorize the peak memory and module-wise memory usage of FSDP modules.

    It tracks the peak memory usage across all the devices of all the FSDP modules in the module tree and categorizes
    the tensor memory usage as defined by ``_FSDPRefType``. Further, it captures memory `snapshots` at different stages of
    the module execution defined by ``_FSDPModState``.

    Attributes:
        memory_tracking: A weakref key dictionary to store the memory statistics of each module. Each key is a reference
        to a module, and each value is a ``_FSDPModMemStats`` object that stores the memory statistics of the module.

    Args:
        mod (torch.nn.Module): The root FSDP module to be tracked.
        optm (torch.optim.Optimizer, optional): The optimizer to be tracked.

    Note: Please refer to ``torch.distributed._tools.mem_tracker.MemTracker`` to learn about the limitations.

    Example usage

    .. code-block:: python

        module = ...
        optimizer = ...
        inp = ...
        fmt = FSDPMemTracker(module, optimizer)
        fmt.track_inputs((inp,))
        with fmt:
            optimizer.zero_grad()
            loss = module(inp)
            print("After Forward:")
            fmt.display_snapshot("current")
            loss.backward()
            optimizer.step()
        fmt.display_snapshot("peak")
        fmt.display_modulewise_snapshots(depth=3, units="MB")

    '''
    _root_mod: Incomplete
    _optm: Incomplete
    _fsdp_mod_to_saved_methods: WeakIdKeyDictionary
    _fsdp_state: _FSDPState
    _ref_class: type[_RefType]
    def __init__(self, mod: torch.nn.Module, optm: torch.optim.Optimizer | None = None) -> None: ...
    def _instrument_fsdp_sharded_params_grads(self, fsdp_param_group: FSDPParamGroup) -> None: ...
    _ac_mod: Incomplete
    _in_ac: bool
    def _fsdp_state_pre_forward(self, fsdp_mod: FSDPModule, orig_fsdp_state_pre_fw: Callable[_P, tuple[tuple[Unpack[_Ts]], dict[str, Any]]]) -> Callable[_P, tuple[tuple[Unpack[_Ts]], dict[str, Any]]]: ...
    def _fsdp_state_post_forward(self, fsdp_mod: FSDPModule, orig_fsdp_state_post_fw: Callable[_P, _R]) -> Callable[_P, _R]: ...
    def _fsdp_param_group_pre_backward(self, fsdp_mod: FSDPModule, orig_fsdp_param_group_pre_backward: Callable[_P, Any]) -> Callable[_P, None]: ...
    def _fsdp_param_group_post_backward(self, fsdp_mod: FSDPModule, orig_fsdp_param_group_post_backward: Callable[_P, Any]) -> Callable[_P, None]: ...
    def _instrument_fsdp_module(self) -> None: ...
    _in_opt: bool
    _optimizer_hook_handles: Incomplete
    def _instrument_optimizer(self) -> None: ...
    def _register_module_and_optimizer_hooks(self) -> None: ...
    def _deregister_module_and_optimizer_hooks(self) -> None: ...
    def track_inputs(self, inputs: tuple[Any, ...]) -> None:
        """
        This is used to track the input tensors to the model and annotate them as ``Inputs``.
        Args:
            inputs (Tuple[Any]): A tuple containing the input data. This can include tensors
                        as well as other data types. Only tensors will be tracked.
        """
    def track_external(self, *external: nn.Module | optim.Optimizer | torch.Tensor) -> None:
        """This is no-op for ``FSDPMemTracker``"""
    _peak_mem_snap: Incomplete
    _peak_mem: Incomplete
    def __enter__(self) -> FSDPMemTracker: ...
    def __exit__(self, *args: Any) -> None: ...
    def __torch_dispatch__(self, func, types, args=..., kwargs=None): ...
