import torch
import torch.nn as nn
from ._fsdp_api import MixedPrecisionPolicy, OffloadPolicy
from ._fsdp_param_group import FSDPParamGroup
from ._fsdp_state import FSDPState
from _typeshed import Incomplete
from torch.distributed.tensor import DeviceMesh, Shard
from typing import Any, Callable, overload

__all__ = ['fully_shard', 'FSDPModule', 'UnshardHandle', 'register_fsdp_forward_method']

@overload
def fully_shard(module: nn.Module, *, mesh: DeviceMesh | None = ..., reshard_after_forward: bool | int = ..., shard_placement_fn: Callable[[nn.Parameter], Shard | None] | None = ..., mp_policy: MixedPrecisionPolicy = ..., offload_policy: OffloadPolicy = ..., ignored_params: set[nn.Parameter] | None = ...) -> FSDPModule: ...
@overload
def fully_shard(module: list[nn.Module], *, mesh: DeviceMesh | None = ..., reshard_after_forward: bool | int = ..., shard_placement_fn: Callable[[nn.Parameter], Shard | None] | None = ..., mp_policy: MixedPrecisionPolicy = ..., offload_policy: OffloadPolicy = ..., ignored_params: set[nn.Parameter] | None = ...) -> list[FSDPModule]: ...

class FSDPModule:
    def __new__(cls, *args, **kwargs):
        """
        Override ``__new__`` to remove the FSDP class and directly construct
        the original class for cases like indexing into a container module.
        """
    def reshard(self) -> None:
        """
        Reshards the module's parameters, freeing the unsharded parameters if
        they are allocated and registering the sharded parameters to the
        module. This method is *not* recursive.
        """
    def unshard(self, async_op: bool = False) -> UnshardHandle | None:
        """
        Unshards the module's parameters by allocating memory and all-gathering
        the parameters. This method is *not* recursive. The unshard follows the
        :class:`MixedPrecisionPolicy`, so it will all-gather following
        ``param_dtype`` if set.

        Args:
            async_op (bool): If ``True``, then returns a :class:`UnshardHandle`
                that has a :meth:`wait` method to wait on the unshard op. If
                ``False``, then returns ``None`` and waits on the handle inside
                this function.

        .. note:: If ``async_op=True``, then FSDP will wait on the pending
            unshard in the module's pre-forward for the user. The user only
            needs to call :meth:`wait` explicitly if the wait should happen
            before pre-forward.
        """
    def set_is_last_backward(self, is_last_backward: bool) -> None:
        """
        Sets whether the next backward is the last one. On the last backward,
        FSDP waits on pending gradient reduction and clears internal data
        data structures for backward prefetching. This can be useful for
        microbatching.
        """
    def set_requires_gradient_sync(self, requires_gradient_sync: bool, *, recurse: bool = True) -> None:
        """
        Sets if the module should sync gradients. This can be used to implement
        gradient accumulation *without communication*. For HSDP, this controls
        both reduce-scatter and all-reduce together. This is the equivalence of
        `no_sync` in FSDP1.

        Args:
            requires_gradient_sync (bool): Whether to reduce gradients for the
                module's parameters.
            recurse (bool): Whether to set for all FSDP submodules or just the
                passed-in module.
        """
    def set_requires_all_reduce(self, requires_all_reduce: bool, *, recurse: bool = True) -> None:
        """
        Sets if the module should all-reduce gradients. This can be used to
        implement gradient accumulation with only reduce-scatter but not
        all-reduce for HSDP.
        """
    def set_reshard_after_forward(self, reshard_after_forward: bool, recurse: bool = True) -> None:
        """
        Sets if the module should reshard parameters after forward. This can be
        used to change the ``reshard_after_forward`` FSDP arg at runtime. For
        example, this can be used to set the FSDP root module's value to
        ``True`` (since it is otherwise specially set to ``False``), or it can
        set an FSDP module's value to ``False`` for running evals and set back
        to ``True`` for training.

        Args:
            reshard_after_forward (bool): Whether to reshard parameters after
                forward.
            recurse (bool): Whether to set for all FSDP submodules or just the
                passed-in module.
        """
    def set_reshard_after_backward(self, reshard_after_backward: bool, *, recurse: bool = True) -> None:
        """
        Sets if the module should reshard parameters after backward. This can
        be used during gradient accumulation to trade off higher memory for
        reduced communication since the unsharded parameters do not need to be
        re-all-gathered before the next forward.

        Args:
            reshard_after_backward (bool): Whether to reshard parameters after
                backward.
            recurse (bool): Whether to set for all FSDP submodules or just the
                passed-in module.
        """
    def set_modules_to_forward_prefetch(self, modules: list[FSDPModule]) -> None:
        """
        Sets the FSDP modules for which this FSDP module should explicitly
        prefetch all-gathers in forward. The prefetching runs after this
        module's all-gather copy-out.

        Passing a singleton list containing the next FSDP module gives the same
        all-gather overlap behavior as the default overlap behavior, except the
        prefetched all-gather is issued earlier from the CPU. Passing a list
        with at least length two is required for more aggressive overlap and
        will use more reserved memory.

        Args:
            modules (List[FSDPModule]): FSDP modules to prefetch.
        """
    def set_modules_to_backward_prefetch(self, modules: list[FSDPModule]) -> None:
        """
        Sets the FSDP modules for which this FSDP module should explicitly
        prefetch all-gathers in backward. This overrides the default backward
        pretching implementation that prefetches the next FSDP module based on
        the reverse post-forward order.

        Passing a singleton list containing the previous FSDP module gives the
        same all-gather overlap behavior as the default overlap behavior.
        Passing a list with at least length two is required for more aggressive
        overlap and will use more reserved memory.

        Args:
            modules (List[FSDPModule]): FSDP modules to prefetch.
        """
    def set_all_reduce_hook(self, hook: Callable[[torch.Tensor], None], *, stream: torch.cuda.Stream | None = None):
        """
        Args:
            hook (Callable[[torch.Tensor], None]): User-defined all-reduce hook
                with expected signature ``hook(reduce_output: torch.Tensor) -> None``
                where ``reduce_output`` is the reduce-scatter output if only
                using FSDP or the all-reduce output if using native HSDP.
            stream (Optional[torch.cuda.Stream]): Stream to run the all-reduce
                hook in. This should only be set if not using native HSDP. If
                using native HSDP, the hook will run in the internally defined
                all-reduce stream used by the native HSDP all-reduce.
        """
    def set_post_optim_event(self, event: torch.Event) -> None:
        """
        Sets a post-optimizer-step event for the root FSDP module to wait the
        all-gather streams on.

        By default, the root FSDP module waits the all-gather streams on the
        current stream to ensure that the optimizer step has finished before
        all-gathering. However, this may introduce false dependencies if
        there is unrelated computation after the optimizer step. This API
        allows the user to provide their own event to wait on. After the root
        waits on the event, the event is discarded, so this API should be
        called with a new event each iteration.

        Args:
            event (torch.Event): Event recorded after the optimizer step
                to wait all-gather streams on.
        """
    def set_reduce_scatter_divide_factor(self, factor: float) -> None:
        """Use :py:meth:`set_gradient_divide_factor` instead"""
    def set_gradient_divide_factor(self, factor: float) -> None:
        """
        Sets a custom divide factor for the gradient reduction. This might use
        a custom reduce op using NCCL's PreMulSum, which allows multiplying by
        the factor before reduction.

        Args:
            factor (float): Custom divide factor.
        """
    def set_force_sum_reduction_for_comms(self, enable: bool) -> None:
        '''
        Sets whether to require the low-level collective communication
        primitives to exclusively use "sum"-type reductions, even if it comes
        at the cost of separate additional pre- or post-scaling operations.
        This is needed for example because NCCL currently supports zero-copy
        transfers only for this kind of collectives.

        NB: for MTIA devices, this is always implicitly enabled.

        NB: if `set_all_reduce_hook` is used under FSDP setup, the caller needs
        to ensure the custom all-reduce across FSDP units follow this strategy
        as well, as FSDP can no longer automatically handle that.

        Args:
            enable (bool): Whether to only ever use ReduceOp.SUM for comms.
        '''
    def set_unshard_in_backward(self, unshard_in_backward: bool) -> None:
        """
        Sets whether the FSDP module's parameters need to be unsharded in
        backward. This can be used in expert cases when the user knows that all
        parameters in this FSDP module's parameter group are not needed for
        backward computation (e.g. embedding).
        """
    def set_allocate_memory_from_process_group_for_comm(self, enable: bool) -> None:
        """
        Sets whether the temporary staging buffers used to send and receive data
        over collective communications should be allocated using the custom
        optimized allocator provided by the ProcessGroup itself (if any). This
        might allow the ProcessGroup to be more efficient. For example, when
        using NCCL, this enables it to leverage zero-copy transfers over SHARP
        (for NVLink and/or InfiniBand).

        Args:
            enable (bool): Whether to turn on ProcessGroup allocation.
        """
    def _set_unshard_async_op(self, async_op: bool):
        """
        Sets whether to use ``async_op=True`` or ``False`` for the pre-forward
        and pre-backward unshard op. This defaults to ``False`` but can be set
        to ``True`` with this method.

        Setting this to ``True`` allows the all-gather allocations to happen in
        the default stream, avoiding inter-stream memory fragmentation.
        However, you must use explicit prefetching (e.g. via :meth:`unshard`)
        in forward to still get overlap, and the pre-all-gather ops like dtype
        casting and copy-in will not overlap with compute.
        """
    def _get_fsdp_state(self) -> FSDPState: ...
    def _apply(self, *args: Any, **kwargs: Any) -> Any: ...

class UnshardHandle:
    """
    A handle to wait on a :meth:`FSDPModule.unshard` op.
    """
    def wait(self) -> None:
        """
        Waits on the unshard op. This ensures that the current stream can use
        the unsharded parameters, which are now registered to the module.
        """

class _UnshardHandleImpl(UnshardHandle):
    _fsdp_param_group: Incomplete
    def __init__(self, fsdp_param_group: FSDPParamGroup | None) -> None: ...
    def wait(self) -> None: ...

def register_fsdp_forward_method(module: nn.Module, method_name: str) -> None:
    """
    Registers a method on ``module`` to be considered a forward method for
    FSDP.

    FSDP all-gathers parameters pre-forward and optionally frees parameters
    post-forward (depending on ``reshard_after_forward``). FSDP only knows to
    do this for :meth:`nn.Module.forward` by default. This function patches a
    user-specified method to run the pre/post-forward hooks before/after the
    method, respectively. If ``module`` is not an :class:`FSDPModule`, then
    this is a no-op.

    Args:
        module (nn.Module): Module to register the forward method on.
        method_name (str): Name of the forward method.
    """
