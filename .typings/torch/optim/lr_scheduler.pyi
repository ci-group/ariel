import types
from .optimizer import Optimizer
from _typeshed import Incomplete
from collections.abc import Iterable, Sequence
from typing import Any, Callable, Literal, SupportsFloat, TypedDict
from typing_extensions import Self, override

__all__ = ['LambdaLR', 'MultiplicativeLR', 'StepLR', 'MultiStepLR', 'ConstantLR', 'LinearLR', 'ExponentialLR', 'SequentialLR', 'CosineAnnealingLR', 'ChainedScheduler', 'ReduceLROnPlateau', 'CyclicLR', 'CosineAnnealingWarmRestarts', 'OneCycleLR', 'PolynomialLR', 'LRScheduler']

class LRScheduler:
    """Adjusts the learning rate during optimization."""
    _get_lr_called_within_step: bool
    _is_initial: bool
    optimizer: Incomplete
    base_lrs: list[float]
    last_epoch: Incomplete
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1) -> None: ...
    _step_count: int
    def _initial_step(self) -> None:
        """Initialize step counts and perform a step."""
    def state_dict(self) -> dict[str, Any]:
        """Return the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
    def load_state_dict(self, state_dict: dict[str, Any]):
        """Load the scheduler's state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
    def get_last_lr(self) -> list[float]:
        """Return last computed learning rate by current scheduler."""
    def get_lr(self) -> list[float]:
        """Compute learning rate using chainable form of the scheduler."""
    _last_lr: list[float]
    def step(self, epoch: int | None = None) -> None:
        """Perform a step."""

class _LRScheduler(LRScheduler): ...

class _enable_get_lr_call:
    o: Incomplete
    def __init__(self, o: LRScheduler) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, type: type[BaseException] | None, value: BaseException | None, traceback: types.TracebackType | None) -> None: ...

class _initial_mode:
    o: Incomplete
    def __init__(self, o: LRScheduler) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, type: type[BaseException] | None, value: BaseException | None, traceback: types.TracebackType | None) -> None: ...

class LambdaLR(LRScheduler):
    """Sets the initial learning rate.

    The learning rate of each parameter group is set to the initial lr
    times a given function. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in optimizer.param_groups.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer has two groups.
        >>> num_epochs = 100
        >>> lambda1 = lambda epoch: epoch // 30
        >>> lambda2 = lambda epoch: 0.95**epoch
        >>> scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
        >>> for epoch in range(num_epochs):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
        >>>
        >>> # Alternatively, you can use a single lambda function for all groups.
        >>> scheduler = LambdaLR(opt, lr_lambda=lambda epoch: epoch // 30)
        >>> for epoch in range(num_epochs):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/LambdaLR.png
    """
    optimizer: Incomplete
    lr_lambdas: list[Callable[[int], float]]
    def __init__(self, optimizer: Optimizer, lr_lambda: Callable[[int], float] | list[Callable[[int], float]], last_epoch: int = -1) -> None: ...
    @override
    def state_dict(self) -> dict[str, Any]:
        """Return the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The learning rate lambda functions will only be saved if they are callable objects
        and not if they are functions or lambdas.

        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.
        """
    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the scheduler's state.

        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
    @override
    def get_lr(self) -> list[float]:
        """Compute learning rate."""

class MultiplicativeLR(LRScheduler):
    """Multiply the learning rate of each parameter group by the factor given in the specified function.

    When last_epoch=-1, set initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in optimizer.param_groups.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # xdoctest: +SKIP
        >>> lmbda = lambda epoch: 0.95
        >>> scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/MultiplicativeLR.png
    """
    optimizer: Incomplete
    lr_lambdas: list[Callable[[int], float]]
    def __init__(self, optimizer: Optimizer, lr_lambda: Callable[[int], float] | list[Callable[[int], float]], last_epoch: int = -1) -> None: ...
    @override
    def state_dict(self) -> dict[str, Any]:
        """Return the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The learning rate lambda functions will only be saved if they are callable objects
        and not if they are functions or lambdas.
        """
    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the scheduler's state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
    @override
    def get_lr(self) -> list[float]:
        """Compute the learning rate of each parameter group."""

class StepLR(LRScheduler):
    """Decays the learning rate of each parameter group by gamma every step_size epochs.

    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60
        >>> # lr = 0.0005   if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/StepLR.png
    """
    step_size: Incomplete
    gamma: Incomplete
    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1) -> None: ...
    @override
    def get_lr(self) -> list[float]:
        """Compute the learning rate of each parameter group."""
    def _get_closed_form_lr(self) -> list[float]: ...

class MultiStepLR(LRScheduler):
    """Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones.

    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/MultiStepLR.png
    """
    milestones: Incomplete
    gamma: Incomplete
    def __init__(self, optimizer: Optimizer, milestones: Iterable[int], gamma: float = 0.1, last_epoch: int = -1) -> None: ...
    @override
    def get_lr(self) -> list[float]:
        """Compute the learning rate of each parameter group."""
    def _get_closed_form_lr(self): ...

class ConstantLR(LRScheduler):
    """Multiply the learning rate of each parameter group by a small constant factor.

    The multiplication is done until the number of epoch reaches a pre-defined milestone: total_iters.
    Notice that such multiplication of the small constant factor can
    happen simultaneously with other changes to the learning rate from outside this scheduler.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        factor (float): The number we multiply learning rate until the milestone. Default: 1./3.
        total_iters (int): The number of steps that the scheduler multiplies the learning rate by the factor.
            Default: 5.
        last_epoch (int): The index of the last epoch. Default: -1.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.025   if epoch == 0
        >>> # lr = 0.025   if epoch == 1
        >>> # lr = 0.025   if epoch == 2
        >>> # lr = 0.025   if epoch == 3
        >>> # ...
        >>> # lr = 0.05    if epoch >= 40
        >>> scheduler = ConstantLR(optimizer, factor=0.5, total_iters=40)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/ConstantLR.png
    """
    factor: Incomplete
    total_iters: Incomplete
    def __init__(self, optimizer: Optimizer, factor: float = ..., total_iters: int = 5, last_epoch: int = -1) -> None: ...
    @override
    def get_lr(self) -> list[float]:
        """Compute the learning rate of each parameter group."""
    def _get_closed_form_lr(self): ...

class LinearLR(LRScheduler):
    """Decays the learning rate of each parameter group by linearly changing small multiplicative factor.

    The multiplication is done until the number of epoch reaches a pre-defined milestone: total_iters.
    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        start_factor (float): The number we multiply learning rate in the first epoch.
            The multiplication factor changes towards end_factor in the following epochs.
            Default: 1./3.
        end_factor (float): The number we multiply learning rate at the end of linear changing
            process. Default: 1.0.
        total_iters (int): The number of iterations that multiplicative factor reaches to 1.
            Default: 5.
        last_epoch (int): The index of the last epoch. Default: -1.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.003687  if epoch == 0
        >>> # lr = 0.004875  if epoch == 1
        >>> # lr = 0.006062  if epoch == 2
        >>> # lr = 0.00725   if epoch == 3
        >>> # ...
        >>> # lr = 0.05      if epoch >= 40
        >>> scheduler = LinearLR(optimizer, start_factor=0.05, total_iters=40)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/LinearLR.png
    """
    start_factor: Incomplete
    end_factor: Incomplete
    total_iters: Incomplete
    def __init__(self, optimizer: Optimizer, start_factor: float = ..., end_factor: float = 1.0, total_iters: int = 5, last_epoch: int = -1) -> None: ...
    @override
    def get_lr(self) -> list[float]:
        """Compute the learning rate."""
    def _get_closed_form_lr(self): ...

class ExponentialLR(LRScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch.

    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # xdoctest: +SKIP
        >>> scheduler = ExponentialLR(optimizer, gamma=0.95)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/ExponentialLR.png
    """
    gamma: Incomplete
    def __init__(self, optimizer: Optimizer, gamma: float, last_epoch: int = -1) -> None: ...
    @override
    def get_lr(self) -> list[float]:
        """Compute the learning rate of each parameter group."""
    def _get_closed_form_lr(self): ...

class SequentialLR(LRScheduler):
    """Contains a list of schedulers expected to be called sequentially during the optimization process.

    Specifically, the schedulers will be called according to the milestone points, which should provide exact
    intervals by which each scheduler should be called at a given epoch.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        schedulers (list): List of chained schedulers.
        milestones (list): List of integers that reflects milestone points.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.005     if epoch == 0
        >>> # lr = 0.005     if epoch == 1
        >>> # lr = 0.005     if epoch == 2
        >>> # ...
        >>> # lr = 0.05      if epoch == 20
        >>> # lr = 0.045     if epoch == 21
        >>> # lr = 0.0405    if epoch == 22
        >>> scheduler1 = ConstantLR(optimizer, factor=0.1, total_iters=20)
        >>> scheduler2 = ExponentialLR(optimizer, gamma=0.9)
        >>> scheduler = SequentialLR(
        ...     optimizer,
        ...     schedulers=[scheduler1, scheduler2],
        ...     milestones=[20],
        ... )
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/SequentialLR.png
    """
    _schedulers: Incomplete
    _milestones: Incomplete
    last_epoch: Incomplete
    optimizer: Incomplete
    _last_lr: Incomplete
    def __init__(self, optimizer: Optimizer, schedulers: list[LRScheduler], milestones: list[int], last_epoch: int = -1) -> None: ...
    def recursive_undo(self, sched=None) -> None:
        """
        Recursively undo any step performed by the initialisation of
        schedulers.
        """
    def step(self) -> None:
        """Perform a step."""
    @override
    def state_dict(self) -> dict[str, Any]:
        """Return the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The wrapped scheduler states will also be saved.
        """
    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the scheduler's state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """

class PolynomialLR(LRScheduler):
    '''Decays the learning rate of each parameter group using a polynomial function in the given total_iters.

    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_iters (int): The number of steps that the scheduler decays the learning rate. Default: 5.
        power (float): The power of the polynomial. Default: 1.0.

    Example:
        >>> # xdoctest: +SKIP("undefined vars")
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.0490   if epoch == 0
        >>> # lr = 0.0481   if epoch == 1
        >>> # lr = 0.0472   if epoch == 2
        >>> # ...
        >>> # lr = 0.0      if epoch >= 50
        >>> scheduler = PolynomialLR(optimizer, total_iters=50, power=0.9)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/PolynomialLR.png
    '''
    total_iters: Incomplete
    power: Incomplete
    def __init__(self, optimizer: Optimizer, total_iters: int = 5, power: float = 1.0, last_epoch: int = -1) -> None: ...
    @override
    def get_lr(self) -> list[float]:
        """Compute the learning rate."""
    def _get_closed_form_lr(self): ...

class CosineAnnealingLR(LRScheduler):
    """
    Set the learning rate of each parameter group using a cosine annealing schedule.

    The learning rate is updated recursively using:

    .. math::
        \\eta_{t+1} = \\eta_{\\min} + (\\eta_t - \\eta_{\\min}) \\cdot
        \\frac{1 + \\cos\\left(\\frac{(T_{cur}+1) \\pi}{T_{max}}\\right)}
            {1 + \\cos\\left(\\frac{T_{cur} \\pi}{T_{max}}\\right)}

    This implements a recursive approximation of the closed-form schedule proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_:

    .. math::
        \\eta_t = \\eta_{\\min} + \\frac{1}{2}(\\eta_{\\max} - \\eta_{\\min}) \\left(
            1 + \\cos\\left(\\frac{T_{cur} \\pi}{T_{max}}\\right) \\right)

    where:

    - :math:`\\eta_t` is the learning rate at step :math:`t`
    - :math:`T_{cur}` is the number of epochs since the last restart
    - :math:`T_{max}` is the maximum number of epochs in a cycle

    Note:
        Although SGDR includes periodic restarts, this implementation performs cosine annealing
        **without restarts**, so :math:`T_{cur} = t` and increases monotonically with each call
        to :meth:`step`.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of the last epoch. Default: -1.

    .. _SGDR\\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983

    Example:
        >>> # xdoctest: +SKIP
        >>> num_epochs = 100
        >>> scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        >>> for epoch in range(num_epochs):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/CosineAnnealingLR.png
    """
    T_max: Incomplete
    eta_min: Incomplete
    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0, last_epoch: int = -1) -> None: ...
    @override
    def get_lr(self) -> list[float]:
        """Retrieve the learning rate of each parameter group."""
    def _get_closed_form_lr(self) -> list[float]: ...

class ChainedScheduler(LRScheduler):
    """Chains a list of learning rate schedulers.

    Takes in a sequence of chainable learning rate schedulers and calls their
    step() functions consecutively in just one call to step().

    Args:
        schedulers (sequence): sequence of chained schedulers.
        optimizer (Optimizer, optional): Wrapped optimizer. Default: None.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05      if epoch == 0
        >>> # lr = 0.0450    if epoch == 1
        >>> # lr = 0.0405    if epoch == 2
        >>> # ...
        >>> # lr = 0.00675   if epoch == 19
        >>> # lr = 0.06078   if epoch == 20
        >>> # lr = 0.05470   if epoch == 21
        >>> scheduler1 = ConstantLR(optimizer, factor=0.1, total_iters=20)
        >>> scheduler2 = ExponentialLR(optimizer, gamma=0.9)
        >>> scheduler = ChainedScheduler([scheduler1, scheduler2], optimizer=optimizer)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/ChainedScheduler.png
    """
    _schedulers: Incomplete
    optimizer: Incomplete
    _last_lr: Incomplete
    def __init__(self, schedulers: Sequence[LRScheduler], optimizer: Optimizer | None = None) -> None: ...
    def step(self) -> None:
        """Perform a step."""
    @override
    def state_dict(self) -> dict[str, Any]:
        """Return the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The wrapped scheduler states will also be saved.
        """
    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the scheduler's state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """

class ReduceLROnPlateau(LRScheduler):
    '''Reduce learning rate when a metric has stopped improving.

    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a \'patience\' number
    of epochs, the learning rate is reduced.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: \'min\'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): The number of allowed epochs with no improvement after
            which the learning rate will be reduced.
            For example, consider the case of having no patience (`patience = 0`).
            In the first epoch, a baseline is established and is always considered good as there\'s no previous baseline.
            In the second epoch, if the performance is worse than the baseline,
            we have what is considered an intolerable epoch.
            Since the count of intolerable epochs (1) is greater than the patience level (0),
            the learning rate is reduced at the end of this epoch.
            From the third epoch onwards, the learning rate continues to be reduced at the end of each epoch
            if the performance is worse than the baseline. If the performance improves or remains the same,
            the learning rate is not adjusted.
            Default: 10.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in \'max\'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: \'rel\'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.

    Example:
        >>> # xdoctest: +SKIP
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = ReduceLROnPlateau(optimizer, "min")
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>> # Note that step should be called after validate()
        >>>     scheduler.step(val_loss)

    .. image:: ../scripts/lr_scheduler_images/ReduceLROnPlateau.png
    '''
    factor: Incomplete
    optimizer: Incomplete
    default_min_lr: Incomplete
    min_lrs: Incomplete
    patience: Incomplete
    cooldown: Incomplete
    eps: Incomplete
    last_epoch: int
    _last_lr: Incomplete
    def __init__(self, optimizer: Optimizer, mode: Literal['min', 'max'] = 'min', factor: float = 0.1, patience: int = 10, threshold: float = 0.0001, threshold_mode: Literal['rel', 'abs'] = 'rel', cooldown: int = 0, min_lr: list[float] | float = 0, eps: float = 1e-08) -> None: ...
    best: Incomplete
    cooldown_counter: int
    num_bad_epochs: int
    def _reset(self) -> None:
        """Reset num_bad_epochs counter and cooldown counter."""
    def step(self, metrics: SupportsFloat, epoch=None) -> None:
        """Perform a step."""
    def _reduce_lr(self, epoch) -> None: ...
    @property
    def in_cooldown(self): ...
    def is_better(self, a, best): ...
    mode_worse: Incomplete
    mode: Incomplete
    threshold: Incomplete
    threshold_mode: Incomplete
    def _init_is_better(self, mode, threshold, threshold_mode) -> None: ...
    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the scheduler's state."""

class CyclicLR(LRScheduler):
    '''Sets the learning rate of each parameter group according to cyclical learning rate policy (CLR).

    The policy cycles the learning rate between two boundaries with a constant frequency,
    as detailed in the paper `Cyclical Learning Rates for Training Neural Networks`_.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.

    Cyclical learning rate policy changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.

    This class has three built-in policies, as put forth in the paper:

    * "triangular": A basic triangular cycle without amplitude scaling.
    * "triangular2": A basic triangular cycle that scales initial amplitude by half each cycle.
    * "exp_range": A cycle that scales initial amplitude by :math:`\\text{gamma}^{\\text{cycle iterations}}`
      at each cycle iteration.

    This implementation was adapted from the github repo: `bckenstler/CLR`_

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for each parameter group.
        max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size_up (int): Number of training iterations in the
            increasing half of a cycle. Default: 2000
        step_size_down (int): Number of training iterations in the
            decreasing half of a cycle. If step_size_down is None,
            it is set to step_size_up. Default: None
        mode (str): One of {triangular, triangular2, exp_range}.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
            Default: \'triangular\'
        gamma (float): Constant in \'exp_range\' scaling function:
            gamma**(cycle iterations)
            Default: 1.0
        scale_fn (function): Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            If specified, then \'mode\' is ignored.
            Default: None
        scale_mode (str): {\'cycle\', \'iterations\'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle).
            Default: \'cycle\'
        cycle_momentum (bool): If ``True``, momentum is cycled inversely
            to learning rate between \'base_momentum\' and \'max_momentum\'.
            Default: True
        base_momentum (float or list): Lower momentum boundaries in the cycle
            for each parameter group. Note that momentum is cycled inversely
            to learning rate; at the peak of a cycle, momentum is
            \'base_momentum\' and learning rate is \'max_lr\'.
            Default: 0.8
        max_momentum (float or list): Upper momentum boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_momentum - base_momentum).
            The momentum at any cycle is the difference of max_momentum
            and some scaling of the amplitude; therefore
            base_momentum may not actually be reached depending on
            scaling function. Note that momentum is cycled inversely
            to learning rate; at the start of a cycle, momentum is \'max_momentum\'
            and learning rate is \'base_lr\'
            Default: 0.9
        last_epoch (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_epoch=-1, the schedule is started from the beginning.
            Default: -1

    Example:
        >>> # xdoctest: +SKIP
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.lr_scheduler.CyclicLR(
        ...     optimizer,
        ...     base_lr=0.01,
        ...     max_lr=0.1,
        ...     step_size_up=10,
        ... )
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>         scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/CyclicLR.png

    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    '''
    optimizer: Incomplete
    max_lrs: Incomplete
    total_size: Incomplete
    step_ratio: Incomplete
    mode: Incomplete
    gamma: Incomplete
    _scale_fn_ref: Callable[[float], float]
    _scale_fn_custom: Incomplete
    scale_mode: Incomplete
    cycle_momentum: Incomplete
    use_beta1: Incomplete
    base_momentums: Incomplete
    max_momentums: Incomplete
    base_lrs: Incomplete
    def __init__(self, optimizer: Optimizer, base_lr: float | list[float], max_lr: float | list[float], step_size_up: int = 2000, step_size_down: int | None = None, mode: Literal['triangular', 'triangular2', 'exp_range'] = 'triangular', gamma: float = 1.0, scale_fn: Callable[[float], float] | None = None, scale_mode: Literal['cycle', 'iterations'] = 'cycle', cycle_momentum: bool = True, base_momentum: float = 0.8, max_momentum: float = 0.9, last_epoch: int = -1) -> None: ...
    def _init_scale_fn(self) -> None: ...
    def scale_fn(self, x) -> float:
        """Get the scaling policy."""
    @staticmethod
    def _triangular_scale_fn(x: float) -> float: ...
    @staticmethod
    def _triangular2_scale_fn(x: float) -> float: ...
    @staticmethod
    def _exp_range_scale_fn(gamma: float, x: float) -> float: ...
    @override
    def get_lr(self) -> list[float]:
        """Calculate the learning rate at batch index.

        This function treats `self.last_epoch` as the last batch index.

        If `self.cycle_momentum` is ``True``, this function has a side effect of
        updating the optimizer's momentum.
        """
    @override
    def state_dict(self) -> dict[str, Any]: ...
    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the scheduler's state."""

class CosineAnnealingWarmRestarts(LRScheduler):
    """Set the learning rate of each parameter group using a cosine annealing schedule.

    The :math:`\\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:

    .. math::
        \\eta_t = \\eta_{min} + \\frac{1}{2}(\\eta_{max} - \\eta_{min})\\left(1 +
        \\cos\\left(\\frac{T_{cur}}{T_{i}}\\pi\\right)\\right)

    When :math:`T_{cur}=T_{i}`, set :math:`\\eta_t = \\eta_{min}`.
    When :math:`T_{cur}=0` after restart, set :math:`\\eta_t=\\eta_{max}`.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations until the first restart.
        T_mult (int, optional): A factor by which :math:`T_{i}` increases after a restart. Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        last_epoch (int, optional): The index of the last epoch. Default: -1.

    .. _SGDR\\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983

    Example:
        >>> # xdoctest: +SKIP
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        >>> scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        ...     optimizer, T_0=20
        ... )
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/CosineAnnealingWarmRestarts.png
    """
    T_0: Incomplete
    T_i: Incomplete
    T_mult: Incomplete
    eta_min: Incomplete
    T_cur: Incomplete
    def __init__(self, optimizer: Optimizer, T_0: int, T_mult: int = 1, eta_min: float = 0.0, last_epoch: int = -1) -> None: ...
    @override
    def get_lr(self) -> list[float]:
        """Compute the initial learning rate."""
    last_epoch: Incomplete
    _last_lr: Incomplete
    @override
    def step(self, epoch=None) -> None:
        '''Step could be called after every batch update.

        Example:
            >>> # xdoctest: +SKIP("Undefined vars")
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> iters = len(dataloader)
            >>> for epoch in range(20):
            >>>     for i, sample in enumerate(dataloader):
            >>>         inputs, labels = sample[\'inputs\'], sample[\'labels\']
            >>>         optimizer.zero_grad()
            >>>         outputs = net(inputs)
            >>>         loss = criterion(outputs, labels)
            >>>         loss.backward()
            >>>         optimizer.step()
            >>>         scheduler.step(epoch + i / iters)

        This function can be called in an interleaved way.

        Example:
            >>> # xdoctest: +SKIP("Undefined vars")
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> for epoch in range(20):
            >>>     scheduler.step()
            >>> scheduler.step(26)
            >>> scheduler.step()  # scheduler.step(27), instead of scheduler(20)
        '''

class _SchedulePhase(TypedDict):
    end_step: float
    start_lr: str
    end_lr: str
    start_momentum: str
    end_momentum: str

class OneCycleLR(LRScheduler):
    '''Sets the learning rate of each parameter group according to the 1cycle learning rate policy.

    The 1cycle policy anneals the learning rate from an initial learning rate to some maximum
    learning rate and then from that maximum learning rate to some minimum learning rate much
    lower than the initial learning rate.
    This policy was initially described in the paper `Super-Convergence:
    Very Fast Training of Neural Networks Using Large Learning Rates`_.

    The 1cycle learning rate policy changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.

    This scheduler is not chainable.

    Note also that the total number of steps in the cycle can be determined in one
    of two ways (listed in order of precedence):

    #. A value for total_steps is explicitly provided.
    #. A number of epochs (epochs) and a number of steps per epoch
       (steps_per_epoch) are provided.
       In this case, the number of total steps is inferred by
       total_steps = epochs * steps_per_epoch

    You must either provide a value for total_steps or provide a value for both
    epochs and steps_per_epoch.

    The default behaviour of this scheduler follows the fastai implementation of 1cycle, which
    claims that "unpublished work has shown even better results by using only two phases". To
    mimic the behaviour of the original paper instead, set ``three_phase=True``.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group.
        total_steps (int): The total number of steps in the cycle. Note that
            if a value is not provided here, then it must be inferred by providing
            a value for epochs and steps_per_epoch.
            Default: None
        epochs (int): The number of epochs to train for. This is used along
            with steps_per_epoch in order to infer the total number of steps in the cycle
            if a value for total_steps is not provided.
            Default: None
        steps_per_epoch (int): The number of steps per epoch to train for. This is
            used along with epochs in order to infer the total number of steps in the
            cycle if a value for total_steps is not provided.
            Default: None
        pct_start (float): The percentage of the cycle (in number of steps) spent
            increasing the learning rate.
            Default: 0.3
        anneal_strategy (str): {\'cos\', \'linear\'}
            Specifies the annealing strategy: "cos" for cosine annealing, "linear" for
            linear annealing.
            Default: \'cos\'
        cycle_momentum (bool): If ``True``, momentum is cycled inversely
            to learning rate between \'base_momentum\' and \'max_momentum\'.
            Default: True
        base_momentum (float or list): Lower momentum boundaries in the cycle
            for each parameter group. Note that momentum is cycled inversely
            to learning rate; at the peak of a cycle, momentum is
            \'base_momentum\' and learning rate is \'max_lr\'.
            Default: 0.85
        max_momentum (float or list): Upper momentum boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_momentum - base_momentum).
            Note that momentum is cycled inversely
            to learning rate; at the start of a cycle, momentum is \'max_momentum\'
            and learning rate is \'base_lr\'
            Default: 0.95
        div_factor (float): Determines the initial learning rate via
            initial_lr = max_lr/div_factor
            Default: 25
        final_div_factor (float): Determines the minimum learning rate via
            min_lr = initial_lr/final_div_factor
            Default: 1e4
        three_phase (bool): If ``True``, use a third phase of the schedule to annihilate the
            learning rate according to \'final_div_factor\' instead of modifying the second
            phase (the first two phases will be symmetrical about the step indicated by
            \'pct_start\').
        last_epoch (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_epoch=-1, the schedule is started from the beginning.
            Default: -1

    Example:
        >>> # xdoctest: +SKIP
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
        >>> scheduler = torch.optim.lr_scheduler.OneCycleLR(
        ...     optimizer, max_lr=0.01, steps_per_epoch=len(data_loader), epochs=10
        ... )
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>         optimizer.step()
        >>>         scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/OneCycleLR.png

    .. _Super-Convergence\\: Very Fast Training of Neural Networks Using Large Learning Rates:
        https://arxiv.org/abs/1708.07120
    '''
    optimizer: Incomplete
    total_steps: Incomplete
    _schedule_phases: list[_SchedulePhase]
    _anneal_func_type: Incomplete
    cycle_momentum: Incomplete
    use_beta1: Incomplete
    def __init__(self, optimizer: Optimizer, max_lr: float | list[float], total_steps: int | None = None, epochs: int | None = None, steps_per_epoch: int | None = None, pct_start: float = 0.3, anneal_strategy: Literal['cos', 'linear'] = 'cos', cycle_momentum: bool = True, base_momentum: float | list[float] = 0.85, max_momentum: float | list[float] = 0.95, div_factor: float = 25.0, final_div_factor: float = 10000.0, three_phase: bool = False, last_epoch: int = -1) -> None: ...
    def _anneal_func(self, *args, **kwargs): ...
    @staticmethod
    def _annealing_cos(start, end, pct):
        """Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."""
    @staticmethod
    def _annealing_linear(start, end, pct):
        """Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."""
    @override
    def get_lr(self) -> list[float]:
        """Compute the learning rate of each parameter group."""
