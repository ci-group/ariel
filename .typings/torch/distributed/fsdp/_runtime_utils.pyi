import torch
import torch.nn as nn
from _typeshed import Incomplete
from enum import Enum
from torch.autograd import Variable as Variable
from torch.autograd.graph import register_multi_grad_hook as register_multi_grad_hook
from torch.distributed.algorithms._comm_hooks import LOW_PRECISION_HOOKS as LOW_PRECISION_HOOKS
from torch.distributed.fsdp._common_utils import TrainingState as TrainingState, _FSDPState as _FSDPState, _assert_in_training_states as _assert_in_training_states, _get_module_fsdp_state as _get_module_fsdp_state, _is_composable as _is_composable, _log_post_backward_hook as _log_post_backward_hook, _no_dispatch_record_stream as _no_dispatch_record_stream, clean_tensor_name as clean_tensor_name
from torch.distributed.fsdp._flat_param import FlatParamHandle as FlatParamHandle, FlatParameter as FlatParameter, HandleShardingStrategy as HandleShardingStrategy, HandleTrainingState as HandleTrainingState, RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES as RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES
from torch.distributed.fsdp._init_utils import HYBRID_SHARDING_STRATEGIES as HYBRID_SHARDING_STRATEGIES
from torch.distributed.fsdp.api import BackwardPrefetch as BackwardPrefetch
from torch.distributed.utils import _apply_to_tensors as _apply_to_tensors, _cast_forward_inputs as _cast_forward_inputs, _p_assert as _p_assert, _to_kwargs as _to_kwargs
from typing import Any, no_type_check

logger: Incomplete
HOMOGENEOUS_ATTR_NAMES: Incomplete

class _PrefetchMode(Enum):
    BACKWARD = ...
    FORWARD = ...

def _get_fsdp_root_states_with_modules(module: nn.Module) -> tuple[list[_FSDPState], list[nn.Module]]:
    """
    Returns a tuple containing:
    1. A list of the root ``_FSDPState`` instances in the module tree rooted at
    ``module`` without any duplicates and following the ``module.modules()``
    traversal order (which is assumed to be depth-first).
    2. A corresponding list of the root modules owning the states in the first
    list.

    This is similar to :func:`_get_fsdp_states_with_modules` except that we
    must call :func:`_is_fsdp_root` to force a lazy initialization to determine
    the FSDP root in case lazy initialization has not yet happened.
    """
def _get_fsdp_root_states(module: nn.Module) -> list[_FSDPState]:
    """See :func:`_get_fsdp_root_states_with_modules`."""
def _is_fsdp_root(state: _FSDPState, module: nn.Module) -> bool:
    """
    Returns if ``state`` corresponds to that of an FSDP root.

    For the wrapper code path, ``state`` and ``module`` should be the same. For
    the non-wrapper code path, ``state`` should be ``module`` 's state.
    """
@no_type_check
def _lazy_init(state, root_module):
    """
    Performs initialization lazily, typically right before the first forward
    pass. The laziness is needed to ensure that the parameter device/dtype and
    the FSDP hierarchy have finalized. This method's actual logic only runs on
    the root FSDP instance, which performs initialization for all non-root FSDP
    instances to avoid partial initialization.

    For the non-composable code path, ``state`` and ``root_module`` should be
    the same, namely the FSDP instance itself.
    """
def _check_flat_params_on_expected_device(state: _FSDPState, module: nn.Module):
    """
    Checks that all ``FlatParameter``s in ``module`` 's tree managed by
    ``state`` are on the expected device for *lazy initialization*.
    """
@no_type_check
def _share_state_and_init_handle_attrs(root_state, root_module) -> None:
    """
    Shares data structure state from the ``root_state`` to all FSDP states in
    ``root_module`` 's module tree, and initializes handle attributes. These
    are done together to require a single loop over the states.
    """
@no_type_check
def _init_streams(state) -> None:
    """
    Initializes CUDA streams for overlapping communication, computation, and
    data transfers. The streams should be shared across FSDP instances.
    """
@no_type_check
def _unshard(state, handle, unshard_stream, pre_unshard_stream) -> None:
    """
    Unshards the handles in ``handles``. If the handles are in
    :meth:`summon_full_params` and are using mixed precision, then they are
    forced to full precision.

    Postcondition: handle's ``FlatParameter`` 's data is the padded
    unsharded flat parameter on the compute device.
    """
@no_type_check
def _reshard(state, handle, free_unsharded_flat_param) -> None:
    """
    Reshards the handle. ``free_unsharded_flat_param`` indicates whether to
    free the handle's padded unsharded flat parameter.
    """
def _unshard_grads(handle: FlatParamHandle | None) -> None: ...
def _reshard_grads(handle: FlatParamHandle | None) -> None: ...
@no_type_check
def _pre_forward(state, handle, unshard_fn, module, args, kwargs):
    """
    Runs the pre-forward logic. This includes an opportunity to unshard
    currently sharded parameters such as those for the current forward and
    registering post-backward hooks for these current parameters. This function
    also converts forward ``args`` and ``kwargs`` to the given precision.

    Args:
        handles (List[FlatParamHandle]): Handles giving the parameters used in
            the current forward.
        unshard_fn (Optional[Callable]): A callable to unshard any currently
            sharded parameters or ``None`` to not do any unsharding.
        module (nn.Module): Module whose forward this method runs right before;
            expected by the hook signature.
        args (Tuple[Any, ...]): Module forward ``args``.
        kwargs (Dict[str, Any]): Module forward ``kwargs``.
    """
@no_type_check
def _pre_forward_unshard(state, handle) -> None:
    """Unshards parameters in the pre-forward."""
@no_type_check
def _post_forward(state, handle, reshard_fn, module, input, output):
    """
    Runs the post-forward logic. This includes an opportunity to reshard
    currently unsharded parameters such as those used in the current forward
    and registering pre-backward hooks on the forward outputs.

    Args:
        handles (List[FlatParamHandle]): Handles giving the parameters used in
            the current forward.
        reshard_fn (Optional[Callable]): A callable to reshard any currently
            unsharded parameters (e.g. from the current forward) or ``None`` to
            not do any resharding.
        module (nn.Module): Module whose forward just ran, which should be a
            fully sharded module (see [Note: Fully Sharded Module]); expected
            by the hook signature.
        input (Any): Unused; expected by the hook signature.
        output (Any): Forward pass output; pre-backward hooks are registered on
            the tensors that require gradients in this output.

    Postcondition: Each ``FlatParameter`` 's data points to the sharded flat
    parameter.
    """
@no_type_check
def _post_forward_reshard(state, handle) -> None:
    """Reshards parameters in the post-forward."""
@no_type_check
def _root_pre_forward(state, module, args, kwargs):
    """
    Runs pre-forward logic specific to the root FSDP instance, which should run
    before any individual module's pre-forward. This starts with an attempt at
    lazy initialization (which only runs non-vacuously once). Otherwise, if
    this is called on a non-root FSDP instance, then it returns directly.

    Args:
        module (nn.Module): Module for which this logic tries to run. It may or
            may not be the root. If not, then this method does not do anything.
    """
@no_type_check
def _root_cast_forward_input(state, module, args, kwargs): ...
@no_type_check
def _pre_backward_hook(state, module, handle, grad, *unused):
    """
    Prepares ``_handle`` 's ``FlatParameter`` s for gradient computation.

    Args:
        module (nn.Module): Fully sharded module (see [Note: Fully Sharded
            Module]).
    """
@no_type_check
def _post_backward_hook(state, handle, flat_param, *unused) -> None:
    """
    Reduce-scatters the gradient of ``handle`` 's ``FlatParameter``.

    Precondition: The ``FlatParameter`` 's ``.grad`` attribute contains the
    unsharded gradient for the local batch.

    Postcondition:
    - If using ``NO_SHARD``, then the ``.grad`` attribute is the reduced
    unsharded gradient.
    - Otherwise, the ``_saved_grad_shard`` attribute is the reduced sharded
    gradient (accumulating with any existing gradient).
    """
def _post_backward_reshard_only_hook(state: _FSDPState, handle: FlatParamHandle, *unused: Any) -> None: ...
def _post_backward_reshard(state: _FSDPState, handle: FlatParamHandle, *unused: Any) -> None: ...
@no_type_check
def _should_free_in_backward(state, handle):
    """
    Returns whether FSDP should free the unsharded flat parameter in the
    post-backward or not.
    """
@no_type_check
def _reduce_grad(state, handle) -> None:
    """
    For sharded strategies, this runs gradient reduction, sharded gradient
    accumulation if needed, and the post-reduction callback.
    """
@no_type_check
def _get_reduce_scatter_tensors(state, unsharded_grad):
    """
    Returns the input and output tensors to reduce-scatter, respectively.
    """
@no_type_check
def _accumulate_sharded_grad(state, handle, sharded_grad):
    """
    Accumulates the reduce-scattered sharded gradient with any existing sharded
    gradient if needed, returning the gradient to offload (if CPU offloading is
    enabled).
    """
@no_type_check
def _reduce_grad_no_shard(state, handle) -> None:
    """
    For no-shard, this runs gradient reduction (which directly covers any
    gradient accumulation implicitly) and the post-reduction callback.
    """
@no_type_check
def _post_reduce_grad_callback(state, handle, grad_to_offload) -> None:
    """
    This callback captures any logic to run after the gradient reduction
    finishes. Currently, this offloads the gradient to CPU if CPU offloading is
    enabled and uses sharded gradient views if ``use_orig_params=True``.
    """
@no_type_check
def _offload_grad(state, handle, grad_to_offload) -> None: ...
@no_type_check
def _post_backward_use_sharded_grad_views(handle) -> None: ...
def _div_if_needed(tensor: torch.Tensor, div_factor: float) -> None: ...
@no_type_check
def _cast_grad_to_param_dtype(state, sharded_grad, param) -> None:
    """
    Casts ``sharded_grad`` back to the full parameter dtype so that the
    optimizer step runs with that dtype. This performs an actual cast if
    1. parameters were in reduced precision during the forward since then
    gradients would be in that reduced precision, or
    2. parameters were not in reduced precision but gradients were in
    reduced precision for communication.
    However, if a low precision communication hook is registered, then this
    dtype cast happens in the hook instead.
    """
def _check_grad_to_accumulate(new_sharded_grad: torch.Tensor, accumulated_grad: torch.Tensor) -> None: ...
@no_type_check
def _low_precision_hook_enabled(state): ...
@no_type_check
def _post_backward_final_callback(state, module) -> None:
    """
    This waits for the post-backward to finish and performs some final cleanup.
    This runs at the end of the entire backward pass and should only be called
    on the root FSDP instance.
    """
@no_type_check
def _catch_all_reshard(state) -> None:
    """
    Reshards the parameters that may not have been resharded in the
    post-backward hook. This can happen when a module's output is used in the
    forward pass, meaning that its pre-backward hook runs (unsharding the
    parameter), but the post-backward hook does not run because the output was
    not jused in the loss computation corresponding to this backward pass.
    """
@no_type_check
def _finalize_params(state) -> None:
    """Finalizes the parameters before the next iteration."""
@no_type_check
def _prefetch_handle(state, current_handle, prefetch_mode) -> None:
    """
    Prefetches the next handles if needed (without synchronization). An empty
    handles key cannot prefetch.
    """
@no_type_check
def _get_handle_to_prefetch(state, current_handle):
    '''
    Returns a :class:`list` of the handles keys to prefetch for the next
    module(s), where ``current_handle`` represents the current module.

    "Prefetching" refers to running the unshard logic early (without
    synchronization), and the "next" modules depend on the recorded execution
    order and the current training state.
    '''
def _get_training_state(handle: FlatParamHandle) -> HandleTrainingState:
    """Returns the training state of the handles in ``handle``."""
@no_type_check
def _register_pre_forward_hook(state, module) -> None:
    """
    Registers a pre-forward hook on ``module``.
    """
@no_type_check
def _register_post_forward_hook(state, module) -> None:
    """
    Registers a post-forward hook on ``module``. Even if the module has no
    handles, we should register the hook since it will register the module's
    pre-backward hook.
    """
@no_type_check
def _register_root_pre_forward_hook(state, module) -> None:
    """
    Registers root pre-forward hook on ``module``, which should be the local
    FSDP root.

    NOTE: For the current composable FSDP design, we have each application of
    ``fully_shard()`` to a module to indicate that that module is the local
    FSDP root. We may remove this assumption in the future, in which case we
    will need to register this root pre-forward hook on any candidate module
    that may be the local FSDP root.
    """
@no_type_check
def _register_pre_backward_hooks(state, module, outputs, handle):
    """
    Registers pre-backward hooks on the tensors that require gradients in the
    forward pass outputs ``outputs``, which were computed using the
    ``FlatParameter`` s of ``handles``.

    Args:
        module (nn.Module): Fully sharded module (see [Note: Fully Sharded
            Module]).

    Returns:
        Forward pass outputs with pre-backward hooks registered to tensors that
        require gradients.
    """
def _register_post_backward_hook(state: _FSDPState, handle: FlatParamHandle | None) -> None:
    """
    Registers post-backward hooks on the ``FlatParameter`` s'
    ``AccumulateGrad`` objects to reshard and to reduce-scatter gradients.

    The ``AccumulateGrad`` object represents the last function that finalizes
    the ``FlatParameter`` 's gradient, so it only runs after its entire
    gradient computation has finished.

    We register the post-backward hook only once in the *first* forward that a
    ``FlatParameter`` participates in. This relies on the ``AccumulateGrad``
    object being preserved through multiple forwards.

    NOTE: We follow this heuristic to prefer the *first* forward to target the
    parameter mixed precision case, where there are *separate*
    ``AccumulateGrad`` objects across the different forwards. (Without
    parameter mixed precision, the ``AccumulateGrad`` objects are the same.) If
    we instead prefer the *last* forward, then the hook runs early.
    """
def _register_post_backward_reshard_only_hook(state: _FSDPState, handle: FlatParamHandle | None, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    """
    Registers post-backward hooks to reshard flat parameters that do not
    require gradient. We register these using multi-post-grad hooks on the
    input activations to ensure that all gradients that may depend on the
    parameters have been computed before resharding.
    """
@no_type_check
def _register_post_backward_final_callback(state, module) -> None:
    """
    Registers the post-backward final callback that runs at the end of the
    backward pass. This should be called from the root FSDP instance at the
    beginning of the pre-backward.
    """
def _wait_for_computation_stream(computation_stream: torch.Stream, unshard_stream: torch.Stream, pre_unshard_stream: torch.Stream):
    """
    Has the unshard and pre-unshard streams wait for the computation stream.
    For example, this should be called in the FSDP root's pre-forward to
    respect optimizer step computation.
    """
def _reset_flat_param_grad_info_if_needed(handles: list[FlatParamHandle]):
    """
    Clears the original parameters' gradients if needed. This method's CPU
    overhead is minimal, so we may call it throughout FSDP methods, which serve
    as callsites to free the gradient memory earlier.
    """
@no_type_check
def _get_buffers_and_dtypes_for_computation(state, root_module):
    """
    Returns all buffers in the module tree rooted at ``root_module`` and a
    corresponding list of the buffer dtypes for computation. Each buffer dtype
    is either ``None`` if buffer mixed precision is not enabled or the buffer
    low precision dtype otherwise.
    """
@no_type_check
def _get_orig_buffer_dtypes(state, buffer_names):
    """
    Returns the original buffer types of the given buffer names.
    """
def _cast_buffers_to_dtype_and_device(buffers: list[torch.Tensor], buffer_dtypes: list[torch.dtype | None], device: torch.device) -> None:
    """
    Casts ``buffers`` to the dtypes given by ``buffer_dtypes`` and moves them
    to ``device``. If an element in ``buffer_dtypes`` is ``None``, then the
    corresponding buffer is only moved to ``device``.
    """
