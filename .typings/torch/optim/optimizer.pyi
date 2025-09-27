import torch
from _typeshed import Incomplete
from collections import OrderedDict, defaultdict
from collections.abc import Hashable, Iterable
from torch.utils._foreach_utils import Indices, TensorListList
from torch.utils.hooks import RemovableHandle
from typing import Any, Callable, TypeVar, overload
from typing_extensions import ParamSpec, Self, TypeAlias

__all__ = ['Optimizer', 'register_optimizer_step_pre_hook', 'register_optimizer_step_post_hook']

_T = TypeVar('_T')
_P = ParamSpec('_P')
Args: TypeAlias = tuple[Any, ...]
Kwargs: TypeAlias = dict[str, Any]
StateDict: TypeAlias = dict[str, Any]
DeviceDict = dict[torch.device | None, torch.Tensor]
DeviceDtypeDict = dict[tuple[torch.device, torch.dtype] | None, torch.Tensor]

class _RequiredParameter:
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self) -> str: ...

def register_optimizer_step_pre_hook(hook: GlobalOptimizerPreHook) -> RemovableHandle:
    """Register a pre hook common to all optimizers.

    The hook should have the following signature::

        hook(optimizer, args, kwargs) -> None or modified args and kwargs

    Args:
        hook (Callable): A user defined hook which is registered on all optimizers.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
def register_optimizer_step_post_hook(hook: GlobalOptimizerPostHook) -> RemovableHandle:
    """Register a post hook common to all optimizers.

    The hook should have the following signature::

        hook(optimizer, args, kwargs) -> None

    Args:
        hook (Callable): A user defined hook which is registered on all optimizers.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
ParamsT: TypeAlias = Iterable[torch.Tensor] | Iterable[dict[str, Any]] | Iterable[tuple[str, torch.Tensor]]
R = TypeVar('R')
T = TypeVar('T')

class Optimizer:
    """Base class for all optimizers.

    .. warning::
        Parameters need to be specified as collections that have a deterministic
        ordering that is consistent between runs. Examples of objects that don't
        satisfy those properties are sets and iterators over values of dictionaries.

    Args:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
    """
    OptimizerPreHook: TypeAlias = Callable[[Self, Args, Kwargs], tuple[Args, Kwargs] | None]
    OptimizerPostHook: TypeAlias = Callable[[Self, Args, Kwargs], None]
    _optimizer_step_pre_hooks: dict[int, OptimizerPreHook]
    _optimizer_step_post_hooks: dict[int, OptimizerPostHook]
    _optimizer_state_dict_pre_hooks: OrderedDict[int, Callable[[Optimizer], None]]
    _optimizer_state_dict_post_hooks: OrderedDict[int, Callable[[Optimizer, StateDict], StateDict | None]]
    _optimizer_load_state_dict_pre_hooks: OrderedDict[int, Callable[[Optimizer, StateDict], StateDict | None]]
    _optimizer_load_state_dict_post_hooks: OrderedDict[int, Callable[[Optimizer], None]]
    defaults: Incomplete
    state: defaultdict[torch.Tensor, Any]
    param_groups: list[dict[str, Any]]
    _warned_capturable_if_run_uncaptured: bool
    def __init__(self, params: ParamsT, defaults: dict[str, Any]) -> None: ...
    def __getstate__(self) -> dict[str, Any]: ...
    def __setstate__(self, state: dict[str, Any]) -> None: ...
    def __repr__(self) -> str: ...
    def _cuda_graph_capture_health_check(self) -> None: ...
    def _optimizer_step_code(self) -> None:
        """Entry point for `torch.profile.profiler`.

        When python tracing is enabled the profiler will hook into this
        function at the CPython level to inspect the optimizer's parameters and
        param groups. It is called it after `step()` since many optimizers
        lazily initialize state.

        This is a workaround due to lack of a proper step hook on the optimizer,
        and will be removed if it exists.
        """
    @staticmethod
    def profile_hook_step(func: Callable[_P, R]) -> Callable[_P, R]: ...
    @staticmethod
    def _group_tensors_by_device_and_dtype(tensorlistlist: TensorListList, with_indices: bool = False) -> dict[tuple[None, None], tuple[TensorListList, Indices]] | dict[tuple[torch.device, torch.dtype], tuple[TensorListList, Indices]]:
        """Group a list of lists of tensors by device and dtype.

        Skips this step if we are compiling since this will occur during inductor lowering.
        """
    _zero_grad_profile_name: Incomplete
    def _patch_step_function(self) -> None: ...
    def register_step_pre_hook(self, hook: OptimizerPreHook) -> RemovableHandle:
        """Register an optimizer step pre hook which will be called before optimizer step.

        It should have the following signature::

            hook(optimizer, args, kwargs) -> None or modified args and kwargs

        The ``optimizer`` argument is the optimizer instance being used. If
        args and kwargs are modified by the pre-hook, then the transformed
        values are returned as a tuple containing the new_args and new_kwargs.

        Args:
            hook (Callable): The user defined hook to be registered.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
    def register_step_post_hook(self, hook: OptimizerPostHook) -> RemovableHandle:
        """Register an optimizer step post hook which will be called after optimizer step.

        It should have the following signature::

            hook(optimizer, args, kwargs) -> None

        The ``optimizer`` argument is the optimizer instance being used.

        Args:
            hook (Callable): The user defined hook to be registered.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
    def register_state_dict_pre_hook(self, hook: Callable[[Optimizer], None], prepend: bool = False) -> RemovableHandle:
        """Register a state dict pre-hook which will be called before :meth:`~torch.optim.Optimizer.state_dict` is called.

        It should have the following signature::

            hook(optimizer) -> None

        The ``optimizer`` argument is the optimizer instance being used.
        The hook will be called with argument ``self`` before calling ``state_dict`` on ``self``.
        The registered hook can be used to perform pre-processing before the ``state_dict``
        call is made.

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If True, the provided pre ``hook`` will be fired before
                all the already registered pre-hooks on ``state_dict``. Otherwise,
                the provided ``hook`` will be fired after all the already registered
                pre-hooks. (default: False)

        Returns:
            :class:`torch.utils.hooks.RemoveableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
    def register_state_dict_post_hook(self, hook: Callable[[Optimizer, StateDict], StateDict | None], prepend: bool = False) -> RemovableHandle:
        """Register a state dict post-hook which will be called after :meth:`~torch.optim.Optimizer.state_dict` is called.

        It should have the following signature::

            hook(optimizer, state_dict) -> state_dict or None

        The hook will be called with arguments ``self`` and ``state_dict`` after generating
        a ``state_dict`` on ``self``. The hook may modify the state_dict inplace or optionally
        return a new one. The registered hook can be used to perform post-processing
        on the ``state_dict`` before it is returned.

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If True, the provided post ``hook`` will be fired before
                all the already registered post-hooks on ``state_dict``. Otherwise,
                the provided ``hook`` will be fired after all the already registered
                post-hooks. (default: False)

        Returns:
            :class:`torch.utils.hooks.RemoveableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
    @torch._disable_dynamo
    def state_dict(self) -> StateDict:
        """Return the state of the optimizer as a :class:`dict`.

        It contains two entries:

        * ``state``: a Dict holding current optimization state. Its content
            differs between optimizer classes, but some common characteristics
            hold. For example, state is saved per parameter, and the parameter
            itself is NOT saved. ``state`` is a Dictionary mapping parameter ids
            to a Dict with state corresponding to each parameter.
        * ``param_groups``: a List containing all parameter groups where each
            parameter group is a Dict. Each parameter group contains metadata
            specific to the optimizer, such as learning rate and weight decay,
            as well as a List of parameter IDs of the parameters in the group.
            If a param group was initialized with ``named_parameters()`` the names
            content will also be saved in the state dict.

        NOTE: The parameter IDs may look like indices but they are just IDs
        associating state with param_group. When loading from a state_dict,
        the optimizer will zip the param_group ``params`` (int IDs) and the
        optimizer ``param_groups`` (actual ``nn.Parameter`` s) in order to
        match state WITHOUT additional verification.

        A returned state dict might look something like:

        .. code-block:: text

            {
                'state': {
                    0: {'momentum_buffer': tensor(...), ...},
                    1: {'momentum_buffer': tensor(...), ...},
                    2: {'momentum_buffer': tensor(...), ...},
                    3: {'momentum_buffer': tensor(...), ...}
                },
                'param_groups': [
                    {
                        'lr': 0.01,
                        'weight_decay': 0,
                        ...
                        'params': [0]
                        'param_names' ['param0']  (optional)
                    },
                    {
                        'lr': 0.001,
                        'weight_decay': 0.5,
                        ...
                        'params': [1, 2, 3]
                        'param_names': ['param1', 'layer.weight', 'layer.bias'] (optional)
                    }
                ]
            }

        """
    @staticmethod
    def _process_value_according_to_param_policy(param: torch.Tensor, value: torch.Tensor, param_id: int, param_groups: list[dict[Any, Any]], key: Hashable = None) -> torch.Tensor: ...
    def register_load_state_dict_pre_hook(self, hook: Callable[[Optimizer, StateDict], StateDict | None], prepend: bool = False) -> RemovableHandle:
        """Register a load_state_dict pre-hook which will be called before
        :meth:`~torch.optim.Optimizer.load_state_dict` is called. It should have the
        following signature::

            hook(optimizer, state_dict) -> state_dict or None

        The ``optimizer`` argument is the optimizer instance being used and the
        ``state_dict`` argument is a shallow copy of the ``state_dict`` the user
        passed in to ``load_state_dict``. The hook may modify the state_dict inplace
        or optionally return a new one. If a state_dict is returned, it will be used
        to be loaded into the optimizer.

        The hook will be called with argument ``self`` and ``state_dict`` before
        calling ``load_state_dict`` on ``self``. The registered hook can be used to
        perform pre-processing before the ``load_state_dict`` call is made.

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If True, the provided pre ``hook`` will be fired before
                all the already registered pre-hooks on ``load_state_dict``. Otherwise,
                the provided ``hook`` will be fired after all the already registered
                pre-hooks. (default: False)

        Returns:
            :class:`torch.utils.hooks.RemoveableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
    def register_load_state_dict_post_hook(self, hook: Callable[[Optimizer], None], prepend: bool = False) -> RemovableHandle:
        """Register a load_state_dict post-hook which will be called after
        :meth:`~torch.optim.Optimizer.load_state_dict` is called. It should have the
        following signature::

            hook(optimizer) -> None

        The ``optimizer`` argument is the optimizer instance being used.

        The hook will be called with argument ``self`` after calling
        ``load_state_dict`` on ``self``. The registered hook can be used to
        perform post-processing after ``load_state_dict`` has loaded the
        ``state_dict``.

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If True, the provided post ``hook`` will be fired before
                all the already registered post-hooks on ``load_state_dict``. Otherwise,
                the provided ``hook`` will be fired after all the already registered
                post-hooks. (default: False)

        Returns:
            :class:`torch.utils.hooks.RemoveableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
    @torch._disable_dynamo
    def load_state_dict(self, state_dict: StateDict) -> None:
        '''Load the optimizer state.

        Args:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.

        .. warning::
            Make sure this method is called after initializing :class:`torch.optim.lr_scheduler.LRScheduler`,
            as calling it beforehand will overwrite the loaded learning rates.

        .. note::
            The names of the parameters (if they exist under the "param_names" key of each param group
            in :meth:`state_dict`) will not affect the loading process.
            To use the parameters\' names for custom cases (such as when the parameters in the loaded state dict
            differ from those initialized in the optimizer),
            a custom ``register_load_state_dict_pre_hook`` should be implemented to adapt the loaded dict
            accordingly.
            If ``param_names`` exist in loaded state dict ``param_groups`` they will be saved and override
            the current names, if present, in the optimizer state. If they do not exist in loaded state dict,
            the optimizer ``param_names`` will remain unchanged.

        Example:
            >>> # xdoctest: +SKIP
            >>> model = torch.nn.Linear(10, 10)
            >>> optim = torch.optim.SGD(model.parameters(), lr=3e-4)
            >>> scheduler1 = torch.optim.lr_scheduler.LinearLR(
            ...     optim,
            ...     start_factor=0.1,
            ...     end_factor=1,
            ...     total_iters=20,
            ... )
            >>> scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            ...     optim,
            ...     T_max=80,
            ...     eta_min=3e-5,
            ... )
            >>> lr = torch.optim.lr_scheduler.SequentialLR(
            ...     optim,
            ...     schedulers=[scheduler1, scheduler2],
            ...     milestones=[20],
            ... )
            >>> lr.load_state_dict(torch.load("./save_seq.pt"))
            >>> # now load the optimizer checkpoint after loading the LRScheduler
            >>> optim.load_state_dict(torch.load("./save_optim.pt"))

        '''
    @torch._disable_dynamo
    def zero_grad(self, set_to_none: bool = True) -> None:
        """Reset the gradients of all optimized :class:`torch.Tensor` s.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                This will in general have lower memory footprint, and can modestly improve performance.
                However, it changes certain behaviors. For example:
                1. When the user tries to access a gradient and perform manual ops on it,
                a None attribute or a Tensor full of 0s will behave differently.
                2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\\ s
                are guaranteed to be None for params that did not receive a gradient.
                3. ``torch.optim`` optimizers have a different behavior if the gradient is 0 or None
                (in one case it does the step with a gradient of 0 and in the other it skips
                the step altogether).
        """
    @overload
    def step(self, closure: None = None) -> None: ...
    @overload
    def step(self, closure: Callable[[], float]) -> float: ...
    @torch._disable_dynamo
    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """Add a param group to the :class:`Optimizer` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.

        Args:
            param_group (dict): Specifies what Tensors should be optimized along with group
                specific optimization options.
        """
