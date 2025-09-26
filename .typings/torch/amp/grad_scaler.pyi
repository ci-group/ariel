import torch
from _typeshed import Incomplete
from collections.abc import Iterable
from enum import Enum
from typing import Any, overload

__all__ = ['OptState', 'GradScaler']

class _MultiDeviceReplicator:
    """Lazily serves copies of a tensor to requested devices.

    Copies are cached per-device.
    """
    master: Incomplete
    _per_device_tensors: dict[torch.device, torch.Tensor]
    def __init__(self, master_tensor: torch.Tensor) -> None: ...
    def get(self, device: torch.device) -> torch.Tensor: ...

class OptState(Enum):
    READY = 0
    UNSCALED = 1
    STEPPED = 2

class GradScaler:
    '''An instance ``scaler`` of :class:`GradScaler`.

    Helps perform the steps of gradient scaling
    conveniently.

    * ``scaler.scale(loss)`` multiplies a given loss by ``scaler``\'s current scale factor.
    * ``scaler.step(optimizer)`` safely unscales gradients and calls ``optimizer.step()``.
    * ``scaler.update()`` updates ``scaler``\'s scale factor.

    Example::

        # Creates a GradScaler once at the beginning of training.
        scaler = GradScaler()

        for epoch in epochs:
            for input, target in data:
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target)

                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                scaler.scale(loss).backward()

                # scaler.step() first unscales gradients of the optimizer\'s params.
                # If gradients don\'t contain infs/NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                scaler.step(optimizer)

                # Updates the scale for next iteration.
                scaler.update()

    See the :ref:`Automatic Mixed Precision examples<amp-examples>` for usage
    (along with autocasting) in more complex cases like gradient clipping, gradient accumulation, gradient penalty,
    and multiple losses/optimizers.

    ``scaler`` dynamically estimates the scale factor each iteration.  To minimize gradient underflow,
    a large scale factor should be used.  However, ``float16`` values can "overflow" (become inf or NaN) if
    the scale factor is too large.  Therefore, the optimal scale factor is the largest factor that can be used
    without incurring inf or NaN gradient values.
    ``scaler`` approximates the optimal scale factor over time by checking the gradients for infs and NaNs during every
    ``scaler.step(optimizer)`` (or optional separate ``scaler.unscale_(optimizer)``, see :meth:`unscale_`).

    * If infs/NaNs are found, ``scaler.step(optimizer)`` skips the underlying ``optimizer.step()`` (so the params
      themselves remain uncorrupted) and ``update()`` multiplies the scale by ``backoff_factor``.

    * If no infs/NaNs are found, ``scaler.step(optimizer)`` runs the underlying ``optimizer.step()`` as usual.
      If ``growth_interval`` unskipped iterations occur consecutively, ``update()`` multiplies the scale by
      ``growth_factor``.

    The scale factor often causes infs/NaNs to appear in gradients for the first few iterations as its
    value calibrates.  ``scaler.step`` will skip the underlying ``optimizer.step()`` for these
    iterations.  After that, step skipping should occur rarely (once every few hundred or thousand iterations).

    Args:
        device (str, optional, default="cuda"): Device type to use. Possible values are: \'cuda\' and \'cpu\'.
            The type is the same as the `type` attribute of a :class:`torch.device`.
            Thus, you may obtain the device type of a tensor using `Tensor.device.type`.
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
    '''
    _device: Incomplete
    _enabled: Incomplete
    _init_scale: Incomplete
    _scale: torch.Tensor | None
    _growth_factor: Incomplete
    _backoff_factor: Incomplete
    _growth_interval: Incomplete
    _init_growth_tracker: int
    _growth_tracker: torch.Tensor | None
    _per_optimizer_states: dict[int, dict[str, Any]]
    def __init__(self, device: str = 'cuda', init_scale: float = ..., growth_factor: float = 2.0, backoff_factor: float = 0.5, growth_interval: int = 2000, enabled: bool = True) -> None: ...
    def _check_scale_growth_tracker(self, funcname: str) -> tuple[torch.Tensor, torch.Tensor]: ...
    def _lazy_init_scale_growth_tracker(self, dev: torch.device) -> None: ...
    @overload
    def scale(self, outputs: torch.Tensor) -> torch.Tensor: ...
    @overload
    def scale(self, outputs: list[torch.Tensor]) -> list[torch.Tensor]: ...
    @overload
    def scale(self, outputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]: ...
    @overload
    def scale(self, outputs: Iterable[torch.Tensor]) -> Iterable[torch.Tensor]: ...
    def _unscale_grads_(self, optimizer: torch.optim.Optimizer, inv_scale: torch.Tensor, found_inf: torch.Tensor, allow_fp16: bool) -> dict[torch.device, torch.Tensor]: ...
    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        '''
        Divides ("unscales") the optimizer\'s gradient tensors by the scale factor.

        :meth:`unscale_` is optional, serving cases where you need to
        :ref:`modify or inspect gradients<working-with-unscaled-gradients>`
        between the backward pass(es) and :meth:`step`.
        If :meth:`unscale_` is not called explicitly,  gradients will be unscaled  automatically during :meth:`step`.

        Simple example, using :meth:`unscale_` to enable clipping of unscaled gradients::

            ...
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()

        Args:
            optimizer (torch.optim.Optimizer):  Optimizer that owns the gradients to be unscaled.

        .. note::
            :meth:`unscale_` does not incur a CPU-GPU sync.

        .. warning::
            :meth:`unscale_` should only be called once per optimizer per :meth:`step` call,
            and only after all gradients for that optimizer\'s assigned parameters have been accumulated.
            Calling :meth:`unscale_` twice for a given optimizer between each :meth:`step` triggers a RuntimeError.

        .. warning::
            :meth:`unscale_` may unscale sparse gradients out of place, replacing the ``.grad`` attribute.
        '''
    def _maybe_opt_step(self, optimizer: torch.optim.Optimizer, optimizer_state: dict[str, Any], *args: Any, **kwargs: Any) -> float | None: ...
    def step(self, optimizer: torch.optim.Optimizer, *args: Any, **kwargs: Any) -> float | None:
        """Invoke ``unscale_(optimizer)`` followed by parameter update, if gradients are not infs/NaN.

        :meth:`step` carries out the following two operations:

        1.  Internally invokes ``unscale_(optimizer)`` (unless :meth:`unscale_` was explicitly called for ``optimizer``
            earlier in the iteration).  As part of the :meth:`unscale_`, gradients are checked for infs/NaNs.
        2.  If no inf/NaN gradients are found, invokes ``optimizer.step()`` using the unscaled
            gradients.  Otherwise, ``optimizer.step()`` is skipped to avoid corrupting the params.

        ``*args`` and ``**kwargs`` are forwarded to ``optimizer.step()``.

        Returns the return value of ``optimizer.step(*args, **kwargs)``.

        Args:
            optimizer (torch.optim.Optimizer):  Optimizer that applies the gradients.
            args:  Any arguments.
            kwargs:  Any keyword arguments.

        .. warning::
            Closure use is not currently supported.
        """
    def update(self, new_scale: float | torch.Tensor | None = None) -> None:
        """Update the scale factor.

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

        .. warning::
            For performance reasons, we do not check the scale factor value to avoid synchronizations,
            so the scale factor is not guaranteed to be above 1. If the scale falls below 1 and/or
            you are seeing NaNs in your gradients or loss, something is likely wrong. For example,
            bf16-pretrained models are often incompatible with AMP/fp16 due to differing dynamic ranges.
        """
    def _get_scale_async(self) -> torch.Tensor | None: ...
    def get_scale(self) -> float:
        """Return a Python float containing the current scale, or 1.0 if scaling is disabled.

        .. warning::
            :meth:`get_scale` incurs a CPU-GPU sync.
        """
    def get_growth_factor(self) -> float:
        """Return a Python float containing the scale growth factor."""
    def set_growth_factor(self, new_factor: float) -> None:
        """Set a new scale growth factor.

        Args:
            new_scale (float):  Value to use as the new scale growth factor.
        """
    def get_backoff_factor(self) -> float:
        """Return a Python float containing the scale backoff factor."""
    def set_backoff_factor(self, new_factor: float) -> None:
        """Set a new scale backoff factor.

        Args:
            new_scale (float):  Value to use as the new scale backoff factor.
        """
    def get_growth_interval(self) -> int:
        """Return a Python int containing the growth interval."""
    def set_growth_interval(self, new_interval: int) -> None:
        """Set a new growth interval.

        Args:
            new_interval (int):  Value to use as the new growth interval.
        """
    def _get_growth_tracker(self) -> int: ...
    def is_enabled(self) -> bool:
        """Return a bool indicating whether this instance is enabled."""
    def state_dict(self) -> dict[str, Any]:
        '''Return the state of the scaler as a :class:`dict`.

        It contains five entries:

        * ``"scale"`` - a Python float containing the current scale
        * ``"growth_factor"`` - a Python float containing the current growth factor
        * ``"backoff_factor"`` - a Python float containing the current backoff factor
        * ``"growth_interval"`` - a Python int containing the current growth interval
        * ``"_growth_tracker"`` - a Python int containing the number of recent consecutive unskipped steps.

        If this instance is not enabled, returns an empty dict.

        .. note::
           If you wish to checkpoint the scaler\'s state after a particular iteration, :meth:`state_dict`
           should be called after :meth:`update`.
        '''
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the scaler state.

        If this instance is disabled, :meth:`load_state_dict` is a no-op.

        Args:
           state_dict(dict): scaler state.  Should be an object returned from a call to :meth:`state_dict`.
        """
    def __getstate__(self) -> dict[str, Any]: ...
    def __setstate__(self, state: dict[str, Any]) -> None: ...
    def _check_inf_per_device(self, optimizer: torch.optim.Optimizer) -> dict[str, Any]: ...
    def _found_inf_per_device(self, optimizer: torch.optim.Optimizer) -> dict[str, Any]: ...
