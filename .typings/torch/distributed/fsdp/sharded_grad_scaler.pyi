import torch
from _typeshed import Incomplete
from collections.abc import Iterable
from torch.amp.grad_scaler import GradScaler as GradScaler, OptState as OptState, _MultiDeviceReplicator as _MultiDeviceReplicator
from torch.distributed.distributed_c10d import ProcessGroup as ProcessGroup
from typing import Any, overload

logger: Incomplete

def _refresh_per_optimizer_state() -> dict[str, Any]: ...
def _is_supported_device(tensor: torch.Tensor) -> bool: ...

class _GeneralMultiDeviceReplicator(_MultiDeviceReplicator):
    '''
    Lazily serves tensor to request device. This class extends
    _MultiDeviceReplicator to allow support for "cpu" as a device.
    '''
    master: Incomplete
    _per_device_tensors: dict[torch.device, torch.Tensor]
    def __init__(self, master_tensor: torch.Tensor) -> None: ...

class ShardedGradScaler(GradScaler):
    """
    ShardedGradScaler helps perform gradient scaling in a shard aware manner. It extends
    functionality from GradScaler:
    * Supports Pytorch DDP and FSDP implementations
    * Support CPU offloaded tensors (as used in fully sharded data parallel[FSDP])
    * Supports the custom Mixed Precision loss dtype (fp16, bf16) that FSDP returns
    * Sync inf/nan for scaled gradient tensors on any torch.device (where tensors are placed) across
    nodes

    Example::

        # Creates a ShardedGradScaler once at the beginning of training.
        scaler = ShardedGradScaler()

        for epoch in epochs:
            for input, target in data:
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target)

                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                scaler.scale(loss).backward()

                # scaler.step() first unscales gradients of the optimizer's params.
                # If gradients don't contain infs/NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                scaler.step(optimizer)

                # Updates the scale for next iteration.
                scaler.update()

    See :class:`GradScaler` for explanation of scaling/unscaling and more use cases.

    Args:
        init_scale (float, optional, default=2.**16):  Initial scale factor.
        growth_factor (float, optional, default=2.0):  Factor by which the scale is multiplied during
            :meth:`update` if no inf/NaN gradients occur for ``growth_interval`` consecutive iterations.
        backoff_factor (float, optional, default=0.5):  Factor by which the scale is multiplied during
            :meth:`update` if inf/NaN gradients occur in an iteration.
        growth_interval (int, optional, default=2000):  Number of consecutive iterations without inf/NaN gradients
            that must occur for the scale to be multiplied by ``growth_factor``.
        enabled (bool, optional):  If ``False``, disables gradient scaling. :meth:`step` simply
            invokes the underlying ``optimizer.step()``, and other methods become no-ops.
            Default: ``True``
        process_group (ProcessGroup, optional, default=torch.distributed.group.WORLD):
            process group for sharding
    """
    process_group: Incomplete
    _per_optimizer_states: Incomplete
    def __init__(self, device: str = 'cuda', init_scale: float = ..., backoff_factor: float = 0.5, growth_factor: float = 2.0, growth_interval: int = 2000, enabled: bool = True, process_group: ProcessGroup | None = ...) -> None: ...
    @overload
    def scale(self, outputs: torch.Tensor) -> torch.Tensor: ...
    @overload
    def scale(self, outputs: list[torch.Tensor]) -> list[torch.Tensor]: ...
    @overload
    def scale(self, outputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]: ...
    @overload
    def scale(self, outputs: Iterable[torch.Tensor]) -> Iterable[torch.Tensor]: ...
    def _unscale_grads_(self, optimizer: torch.optim.Optimizer, inv_scale: torch.Tensor, found_inf: torch.Tensor, allow_fp16: bool = True) -> dict[torch.device, torch.Tensor]: ...
    def unscale_(self, optimizer: torch.optim.Optimizer) -> None: ...
    _growth_tracker: Incomplete
    def _amp_update_scale_cpu_(self, found_inf: torch.Tensor) -> None:
        """
        If found_inf is 1.0 (True), then scale is multiplied by backoff_factor and growth_tracker is set to zero.
        Otherwise, scale is multiplied by the growth factor when the growth interval is reached.
        """
    def update(self, new_scale: float | torch.Tensor | None = None) -> None:
        """
        Updates the scale factor.
        If any optimizer steps were skipped the scale is multiplied by ``backoff_factor``
        to reduce it. If ``growth_interval`` unskipped iterations occurred consecutively,
        the scale is multiplied by ``growth_factor`` to increase it.
        Passing ``new_scale`` sets the new scale value manually. (``new_scale`` is not
        used directly, it's used to fill GradScaler's internal scale tensor. So if
        ``new_scale`` was a tensor, later in-place changes to that tensor will not further
        affect the scale GradScaler uses internally.)
        Args:
            new_scale (float or :class:`torch.Tensor`, optional, default=None):  New scale factor.
        .. warning::
            :meth:`update` should only be called at the end of the iteration, after ``scaler.step(optimizer)`` has
            been invoked for all optimizers used this iteration.
        """
