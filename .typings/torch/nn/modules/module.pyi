import weakref
from _typeshed import Incomplete
from collections.abc import Iterator, Mapping
from torch import Tensor, device, dtype
from torch._prims_common import DeviceLikeType
from torch.nn.parameter import Parameter
from torch.utils.hooks import RemovableHandle
from typing import Any, Callable, NamedTuple, TypeVar, overload
from typing_extensions import Self

__all__ = ['register_module_forward_pre_hook', 'register_module_forward_hook', 'register_module_full_backward_pre_hook', 'register_module_backward_hook', 'register_module_full_backward_hook', 'register_module_buffer_registration_hook', 'register_module_module_registration_hook', 'register_module_parameter_registration_hook', 'Module']

_grad_t = tuple[Tensor, ...] | Tensor
T = TypeVar('T', bound='Module')

class _IncompatibleKeys(NamedTuple('IncompatibleKeys', [('missing_keys', Incomplete), ('unexpected_keys', Incomplete)])):
    __slots__: Incomplete
    def __repr__(self) -> str: ...
    __str__ = __repr__

class _WrappedHook:
    hook: Callable
    with_module: bool
    module: weakref.ReferenceType[Module]
    def __init__(self, hook: Callable, module: Module | None = None) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def __getstate__(self) -> dict: ...
    def __setstate__(self, state: dict): ...

def register_module_buffer_registration_hook(hook: Callable[..., None]) -> RemovableHandle:
    """Register a buffer registration hook common to all modules.

    .. warning ::

        This adds global state to the `nn.Module` module

    The hook will be called every time :func:`register_buffer` is invoked.
    It should have the following signature::

        hook(module, name, buffer) -> None or new buffer

    The hook can modify the input or return a single modified value in the hook.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
def register_module_module_registration_hook(hook: Callable[..., None]) -> RemovableHandle:
    """Register a module registration hook common to all modules.

    .. warning ::

        This adds global state to the `nn.Module` module

    The hook will be called every time :func:`register_module` is invoked.
    It should have the following signature::

        hook(module, name, submodule) -> None or new submodule

    The hook can modify the input or return a single modified value in the hook.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
def register_module_parameter_registration_hook(hook: Callable[..., None]) -> RemovableHandle:
    """Register a parameter registration hook common to all modules.

    .. warning ::

        This adds global state to the `nn.Module` module

    The hook will be called every time :func:`register_parameter` is invoked.
    It should have the following signature::

        hook(module, name, param) -> None or new parameter

    The hook can modify the input or return a single modified value in the hook.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
def register_module_forward_pre_hook(hook: Callable[..., None]) -> RemovableHandle:
    """Register a forward pre-hook common to all modules.

    .. warning ::

        This adds global state to the `nn.module` module
        and it is only intended for debugging/profiling purposes.

    The hook will be called every time before :func:`forward` is invoked.
    It should have the following signature::

        hook(module, input) -> None or modified input

    The input contains only the positional arguments given to the module.
    Keyword arguments won't be passed to the hooks and only to the ``forward``.
    The hook can modify the input. User can either return a tuple or a
    single modified value in the hook. We will wrap the value into a tuple
    if a single value is returned(unless that value is already a tuple).

    This hook has precedence over the specific module hooks registered with
    ``register_forward_pre_hook``.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
def register_module_forward_hook(hook: Callable[..., None], *, with_kwargs: bool = False, always_call: bool = False) -> RemovableHandle:
    """Register a global forward hook for all the modules.

    .. warning ::

        This adds global state to the `nn.module` module
        and it is only intended for debugging/profiling purposes.

    The hook will be called every time after :func:`forward` has computed an output.
    It should have the following signature::

        hook(module, input, output) -> None or modified output

    The input contains only the positional arguments given to the module.
    Keyword arguments won't be passed to the hooks and only to the ``forward``.
    You can optionally modify the output of the module by returning a new value
    that will replace the output from the :func:`forward` function.

    Parameters:
        hook (Callable): The user defined hook to be registered.
        always_call (bool): If ``True`` the ``hook`` will be run regardless of
            whether an exception is raised while calling the Module.
            Default: ``False``
    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``

    This hook will be executed before specific module hooks registered with
    ``register_forward_hook``.
    """
def register_module_backward_hook(hook: Callable[[Module, _grad_t, _grad_t], None | _grad_t]) -> RemovableHandle:
    """Register a backward hook common to all the modules.

    This function is deprecated in favor of
    :func:`torch.nn.modules.module.register_module_full_backward_hook`
    and the behavior of this function will change in future versions.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``

    """
def register_module_full_backward_pre_hook(hook: Callable[[Module, _grad_t], None | _grad_t]) -> RemovableHandle:
    """Register a backward pre-hook common to all the modules.

    .. warning ::
        This adds global state to the `nn.module` module
        and it is only intended for debugging/profiling purposes.

    Hooks registered using this function behave in the same way as those
    registered by :meth:`torch.nn.Module.register_full_backward_pre_hook`.
    Refer to its documentation for more details.

    Hooks registered using this function will be called before hooks registered
    using :meth:`torch.nn.Module.register_full_backward_pre_hook`.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``

    """
def register_module_full_backward_hook(hook: Callable[[Module, _grad_t, _grad_t], None | _grad_t]) -> RemovableHandle:
    """Register a backward hook common to all the modules.

    .. warning ::
        This adds global state to the `nn.module` module
        and it is only intended for debugging/profiling purposes.

    Hooks registered using this function behave in the same way as those
    registered by :meth:`torch.nn.Module.register_full_backward_hook`.
    Refer to its documentation for more details.

    Hooks registered using this function will be called before hooks registered
    using :meth:`torch.nn.Module.register_full_backward_hook`.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``

    """

class Module:
    """Base class for all neural network modules.

    Your models should also subclass this class.

    Modules can also contain other Modules, allowing them to be nested in
    a tree structure. You can assign the submodules as regular attributes::

        import torch.nn as nn
        import torch.nn.functional as F


        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))

    Submodules assigned in this way will be registered, and will also have their
    parameters converted when you call :meth:`to`, etc.

    .. note::
        As per the example above, an ``__init__()`` call to the parent class
        must be made before assignment on the child.

    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    """
    dump_patches: bool
    _version: int
    training: bool
    _parameters: dict[str, Parameter | None]
    _buffers: dict[str, Tensor | None]
    _non_persistent_buffers_set: set[str]
    _backward_pre_hooks: dict[int, Callable]
    _backward_hooks: dict[int, Callable]
    _is_full_backward_hook: bool | None
    _forward_hooks: dict[int, Callable]
    _forward_hooks_with_kwargs: dict[int, bool]
    _forward_hooks_always_called: dict[int, bool]
    _forward_pre_hooks: dict[int, Callable]
    _forward_pre_hooks_with_kwargs: dict[int, bool]
    _state_dict_hooks: dict[int, Callable]
    _load_state_dict_pre_hooks: dict[int, Callable]
    _state_dict_pre_hooks: dict[int, Callable]
    _load_state_dict_post_hooks: dict[int, Callable]
    _modules: dict[str, Module | None]
    call_super_init: bool
    _compiled_call_impl: Callable | None
    def __init__(self, *args, **kwargs) -> None:
        """Initialize internal Module state, shared by both nn.Module and ScriptModule."""
    forward: Callable[..., Any]
    def register_buffer(self, name: str, tensor: Tensor | None, persistent: bool = True) -> None:
        '''Add a buffer to the module.

        This is typically used to register a buffer that should not be
        considered a model parameter. For example, BatchNorm\'s ``running_mean``
        is not a parameter, but is part of the module\'s state. Buffers, by
        default, are persistent and will be saved alongside parameters. This
        behavior can be changed by setting :attr:`persistent` to ``False``. The
        only difference between a persistent buffer and a non-persistent buffer
        is that the latter will not be a part of this module\'s
        :attr:`state_dict`.

        Buffers can be accessed as attributes using given names.

        Args:
            name (str): name of the buffer. The buffer can be accessed
                from this module using the given name
            tensor (Tensor or None): buffer to be registered. If ``None``, then operations
                that run on buffers, such as :attr:`cuda`, are ignored. If ``None``,
                the buffer is **not** included in the module\'s :attr:`state_dict`.
            persistent (bool): whether the buffer is part of this module\'s
                :attr:`state_dict`.

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> self.register_buffer(\'running_mean\', torch.zeros(num_features))

        '''
    def register_parameter(self, name: str, param: Parameter | None) -> None:
        """Add a parameter to the module.

        The parameter can be accessed as an attribute using given name.

        Args:
            name (str): name of the parameter. The parameter can be accessed
                from this module using the given name
            param (Parameter or None): parameter to be added to the module. If
                ``None``, then operations that run on parameters, such as :attr:`cuda`,
                are ignored. If ``None``, the parameter is **not** included in the
                module's :attr:`state_dict`.
        """
    def add_module(self, name: str, module: Module | None) -> None:
        """Add a child module to the current module.

        The module can be accessed as an attribute using the given name.

        Args:
            name (str): name of the child module. The child module can be
                accessed from this module using the given name
            module (Module): child module to be added to the module.
        """
    def register_module(self, name: str, module: Module | None) -> None:
        """Alias for :func:`add_module`."""
    def get_submodule(self, target: str) -> Module:
        '''Return the submodule given by ``target`` if it exists, otherwise throw an error.

        For example, let\'s say you have an ``nn.Module`` ``A`` that
        looks like this:

        .. code-block:: text

            A(
                (net_b): Module(
                    (net_c): Module(
                        (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
                    )
                    (linear): Linear(in_features=100, out_features=200, bias=True)
                )
            )

        (The diagram shows an ``nn.Module`` ``A``. ``A`` which has a nested
        submodule ``net_b``, which itself has two submodules ``net_c``
        and ``linear``. ``net_c`` then has a submodule ``conv``.)

        To check whether or not we have the ``linear`` submodule, we
        would call ``get_submodule("net_b.linear")``. To check whether
        we have the ``conv`` submodule, we would call
        ``get_submodule("net_b.net_c.conv")``.

        The runtime of ``get_submodule`` is bounded by the degree
        of module nesting in ``target``. A query against
        ``named_modules`` achieves the same result, but it is O(N) in
        the number of transitive modules. So, for a simple check to see
        if some submodule exists, ``get_submodule`` should always be
        used.

        Args:
            target: The fully-qualified string name of the submodule
                to look for. (See above example for how to specify a
                fully-qualified string.)

        Returns:
            torch.nn.Module: The submodule referenced by ``target``

        Raises:
            AttributeError: If at any point along the path resulting from
                the target string the (sub)path resolves to a non-existent
                attribute name or an object that is not an instance of ``nn.Module``.
        '''
    def set_submodule(self, target: str, module: Module, strict: bool = False) -> None:
        '''
        Set the submodule given by ``target`` if it exists, otherwise throw an error.

        .. note::
            If ``strict`` is set to ``False`` (default), the method will replace an existing submodule
            or create a new submodule if the parent module exists. If ``strict`` is set to ``True``,
            the method will only attempt to replace an existing submodule and throw an error if
            the submodule does not exist.

        For example, let\'s say you have an ``nn.Module`` ``A`` that
        looks like this:

        .. code-block:: text

            A(
                (net_b): Module(
                    (net_c): Module(
                        (conv): Conv2d(3, 3, 3)
                    )
                    (linear): Linear(3, 3)
                )
            )

        (The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested
        submodule ``net_b``, which itself has two submodules ``net_c``
        and ``linear``. ``net_c`` then has a submodule ``conv``.)

        To override the ``Conv2d`` with a new submodule ``Linear``, you
        could call ``set_submodule("net_b.net_c.conv", nn.Linear(1, 1))``
        where ``strict`` could be ``True`` or ``False``

        To add a new submodule ``Conv2d`` to the existing ``net_b`` module,
        you would call ``set_submodule("net_b.conv", nn.Conv2d(1, 1, 1))``.

        In the above if you set ``strict=True`` and call
        ``set_submodule("net_b.conv", nn.Conv2d(1, 1, 1), strict=True)``, an AttributeError
        will be raised because ``net_b`` does not have a submodule named ``conv``.

        Args:
            target: The fully-qualified string name of the submodule
                to look for. (See above example for how to specify a
                fully-qualified string.)
            module: The module to set the submodule to.
            strict: If ``False``, the method will replace an existing submodule
                or create a new submodule if the parent module exists. If ``True``,
                the method will only attempt to replace an existing submodule and throw an error
                if the submodule doesn\'t already exist.

        Raises:
            ValueError: If the ``target`` string is empty or if ``module`` is not an instance of ``nn.Module``.
            AttributeError: If at any point along the path resulting from
                the ``target`` string the (sub)path resolves to a non-existent
                attribute name or an object that is not an instance of ``nn.Module``.
        '''
    def get_parameter(self, target: str) -> Parameter:
        """Return the parameter given by ``target`` if it exists, otherwise throw an error.

        See the docstring for ``get_submodule`` for a more detailed
        explanation of this method's functionality as well as how to
        correctly specify ``target``.

        Args:
            target: The fully-qualified string name of the Parameter
                to look for. (See ``get_submodule`` for how to specify a
                fully-qualified string.)

        Returns:
            torch.nn.Parameter: The Parameter referenced by ``target``

        Raises:
            AttributeError: If the target string references an invalid
                path or resolves to something that is not an
                ``nn.Parameter``
        """
    def get_buffer(self, target: str) -> Tensor:
        """Return the buffer given by ``target`` if it exists, otherwise throw an error.

        See the docstring for ``get_submodule`` for a more detailed
        explanation of this method's functionality as well as how to
        correctly specify ``target``.

        Args:
            target: The fully-qualified string name of the buffer
                to look for. (See ``get_submodule`` for how to specify a
                fully-qualified string.)

        Returns:
            torch.Tensor: The buffer referenced by ``target``

        Raises:
            AttributeError: If the target string references an invalid
                path or resolves to something that is not a
                buffer
        """
    def get_extra_state(self) -> Any:
        """Return any extra state to include in the module's state_dict.

        Implement this and a corresponding :func:`set_extra_state` for your module
        if you need to store extra state. This function is called when building the
        module's `state_dict()`.

        Note that extra state should be picklable to ensure working serialization
        of the state_dict. We only provide backwards compatibility guarantees
        for serializing Tensors; other objects may break backwards compatibility if
        their serialized pickled form changes.

        Returns:
            object: Any extra state to store in the module's state_dict
        """
    def set_extra_state(self, state: Any) -> None:
        """Set extra state contained in the loaded `state_dict`.

        This function is called from :func:`load_state_dict` to handle any extra state
        found within the `state_dict`. Implement this function and a corresponding
        :func:`get_extra_state` for your module if you need to store extra state within its
        `state_dict`.

        Args:
            state (dict): Extra state from the `state_dict`
        """
    def _apply(self, fn, recurse: bool = True): ...
    def apply(self, fn: Callable[[Module], None]) -> Self:
        """Apply ``fn`` recursively to every submodule (as returned by ``.children()``) as well as self.

        Typical use includes initializing the parameters of a model
        (see also :ref:`nn-init-doc`).

        Args:
            fn (:class:`Module` -> None): function to be applied to each submodule

        Returns:
            Module: self

        Example::

            >>> @torch.no_grad()
            >>> def init_weights(m):
            >>>     print(m)
            >>>     if type(m) == nn.Linear:
            >>>         m.weight.fill_(1.0)
            >>>         print(m.weight)
            >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
            >>> net.apply(init_weights)
            Linear(in_features=2, out_features=2, bias=True)
            Parameter containing:
            tensor([[1., 1.],
                    [1., 1.]], requires_grad=True)
            Linear(in_features=2, out_features=2, bias=True)
            Parameter containing:
            tensor([[1., 1.],
                    [1., 1.]], requires_grad=True)
            Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            )

        """
    def cuda(self, device: int | device | None = None) -> Self:
        """Move all model parameters and buffers to the GPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing the optimizer if the module will
        live on GPU while being optimized.

        .. note::
            This method modifies the module in-place.

        Args:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
    def ipu(self, device: int | device | None = None) -> Self:
        """Move all model parameters and buffers to the IPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing the optimizer if the module will
        live on IPU while being optimized.

        .. note::
            This method modifies the module in-place.

        Arguments:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
    def xpu(self, device: int | device | None = None) -> Self:
        """Move all model parameters and buffers to the XPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on XPU while being optimized.

        .. note::
            This method modifies the module in-place.

        Arguments:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
    def mtia(self, device: int | device | None = None) -> Self:
        """Move all model parameters and buffers to the MTIA.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing the optimizer if the module will
        live on MTIA while being optimized.

        .. note::
            This method modifies the module in-place.

        Arguments:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
    def cpu(self) -> Self:
        """Move all model parameters and buffers to the CPU.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
    def type(self, dst_type: dtype | str) -> Self:
        """Casts all parameters and buffers to :attr:`dst_type`.

        .. note::
            This method modifies the module in-place.

        Args:
            dst_type (type or string): the desired type

        Returns:
            Module: self
        """
    def float(self) -> Self:
        """Casts all floating point parameters and buffers to ``float`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
    def double(self) -> Self:
        """Casts all floating point parameters and buffers to ``double`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
    def half(self) -> Self:
        """Casts all floating point parameters and buffers to ``half`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
    def bfloat16(self) -> Self:
        """Casts all floating point parameters and buffers to ``bfloat16`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
    def to_empty(self, *, device: DeviceLikeType | None, recurse: bool = True) -> Self:
        """Move the parameters and buffers to the specified device without copying storage.

        Args:
            device (:class:`torch.device`): The desired device of the parameters
                and buffers in this module.
            recurse (bool): Whether parameters and buffers of submodules should
                be recursively moved to the specified device.

        Returns:
            Module: self
        """
    @overload
    def to(self, device: DeviceLikeType | None = ..., dtype: dtype | None = ..., non_blocking: bool = ...) -> Self: ...
    @overload
    def to(self, dtype: dtype, non_blocking: bool = ...) -> Self: ...
    @overload
    def to(self, tensor: Tensor, non_blocking: bool = ...) -> Self: ...
    def register_full_backward_pre_hook(self, hook: Callable[[Module, _grad_t], None | _grad_t], prepend: bool = False) -> RemovableHandle:
        """Register a backward pre-hook on the module.

        The hook will be called every time the gradients for the module are computed.
        The hook should have the following signature::

            hook(module, grad_output) -> tuple[Tensor] or None

        The :attr:`grad_output` is a tuple. The hook should
        not modify its arguments, but it can optionally return a new gradient with
        respect to the output that will be used in place of :attr:`grad_output` in
        subsequent computations. Entries in :attr:`grad_output` will be ``None`` for
        all non-Tensor arguments.

        For technical reasons, when this hook is applied to a Module, its forward function will
        receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
        of each Tensor returned by the Module's forward function.

        .. warning ::
            Modifying inputs inplace is not allowed when using backward hooks and
            will raise an error.

        Args:
            hook (Callable): The user-defined hook to be registered.
            prepend (bool): If true, the provided ``hook`` will be fired before
                all existing ``backward_pre`` hooks on this
                :class:`torch.nn.Module`. Otherwise, the provided
                ``hook`` will be fired after all existing ``backward_pre`` hooks
                on this :class:`torch.nn.Module`. Note that global
                ``backward_pre`` hooks registered with
                :func:`register_module_full_backward_pre_hook` will fire before
                all hooks registered by this method.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``

        """
    def register_backward_hook(self, hook: Callable[[Module, _grad_t, _grad_t], None | _grad_t]) -> RemovableHandle:
        """Register a backward hook on the module.

        This function is deprecated in favor of :meth:`~torch.nn.Module.register_full_backward_hook` and
        the behavior of this function will change in future versions.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``

        """
    def register_full_backward_hook(self, hook: Callable[[Module, _grad_t, _grad_t], None | _grad_t], prepend: bool = False) -> RemovableHandle:
        """Register a backward hook on the module.

        The hook will be called every time the gradients with respect to a module are computed, and its firing rules are as follows:

            1. Ordinarily, the hook fires when the gradients are computed with respect to the module inputs.
            2. If none of the module inputs require gradients, the hook will fire when the gradients are computed
               with respect to module outputs.
            3. If none of the module outputs require gradients, then the hooks will not fire.

        The hook should have the following signature::

            hook(module, grad_input, grad_output) -> tuple(Tensor) or None

        The :attr:`grad_input` and :attr:`grad_output` are tuples that contain the gradients
        with respect to the inputs and outputs respectively. The hook should
        not modify its arguments, but it can optionally return a new gradient with
        respect to the input that will be used in place of :attr:`grad_input` in
        subsequent computations. :attr:`grad_input` will only correspond to the inputs given
        as positional arguments and all kwarg arguments are ignored. Entries
        in :attr:`grad_input` and :attr:`grad_output` will be ``None`` for all non-Tensor
        arguments.

        For technical reasons, when this hook is applied to a Module, its forward function will
        receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
        of each Tensor returned by the Module's forward function.

        .. warning ::
            Modifying inputs or outputs inplace is not allowed when using backward hooks and
            will raise an error.

        Args:
            hook (Callable): The user-defined hook to be registered.
            prepend (bool): If true, the provided ``hook`` will be fired before
                all existing ``backward`` hooks on this
                :class:`torch.nn.Module`. Otherwise, the provided
                ``hook`` will be fired after all existing ``backward`` hooks on
                this :class:`torch.nn.Module`. Note that global
                ``backward`` hooks registered with
                :func:`register_module_full_backward_hook` will fire before
                all hooks registered by this method.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``

        """
    def _get_backward_hooks(self):
        """Return the backward hooks for use in the call function.

        It returns two lists, one with the full backward hooks and one with the non-full
        backward hooks.
        """
    def _get_backward_pre_hooks(self): ...
    def _maybe_warn_non_full_backward_hook(self, inputs, result, grad_fn) -> None: ...
    def register_forward_pre_hook(self, hook: Callable[[T, tuple[Any, ...]], Any | None] | Callable[[T, tuple[Any, ...], dict[str, Any]], tuple[Any, dict[str, Any]] | None], *, prepend: bool = False, with_kwargs: bool = False) -> RemovableHandle:
        """Register a forward pre-hook on the module.

        The hook will be called every time before :func:`forward` is invoked.


        If ``with_kwargs`` is false or not specified, the input contains only
        the positional arguments given to the module. Keyword arguments won't be
        passed to the hooks and only to the ``forward``. The hook can modify the
        input. User can either return a tuple or a single modified value in the
        hook. We will wrap the value into a tuple if a single value is returned
        (unless that value is already a tuple). The hook should have the
        following signature::

            hook(module, args) -> None or modified input

        If ``with_kwargs`` is true, the forward pre-hook will be passed the
        kwargs given to the forward function. And if the hook modifies the
        input, both the args and kwargs should be returned. The hook should have
        the following signature::

            hook(module, args, kwargs) -> None or a tuple of modified input and kwargs

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If true, the provided ``hook`` will be fired before
                all existing ``forward_pre`` hooks on this
                :class:`torch.nn.Module`. Otherwise, the provided
                ``hook`` will be fired after all existing ``forward_pre`` hooks
                on this :class:`torch.nn.Module`. Note that global
                ``forward_pre`` hooks registered with
                :func:`register_module_forward_pre_hook` will fire before all
                hooks registered by this method.
                Default: ``False``
            with_kwargs (bool): If true, the ``hook`` will be passed the kwargs
                given to the forward function.
                Default: ``False``

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
    def register_forward_hook(self, hook: Callable[[T, tuple[Any, ...], Any], Any | None] | Callable[[T, tuple[Any, ...], dict[str, Any], Any], Any | None], *, prepend: bool = False, with_kwargs: bool = False, always_call: bool = False) -> RemovableHandle:
        """Register a forward hook on the module.

        The hook will be called every time after :func:`forward` has computed an output.

        If ``with_kwargs`` is ``False`` or not specified, the input contains only
        the positional arguments given to the module. Keyword arguments won't be
        passed to the hooks and only to the ``forward``. The hook can modify the
        output. It can modify the input inplace but it will not have effect on
        forward since this is called after :func:`forward` is called. The hook
        should have the following signature::

            hook(module, args, output) -> None or modified output

        If ``with_kwargs`` is ``True``, the forward hook will be passed the
        ``kwargs`` given to the forward function and be expected to return the
        output possibly modified. The hook should have the following signature::

            hook(module, args, kwargs, output) -> None or modified output

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If ``True``, the provided ``hook`` will be fired
                before all existing ``forward`` hooks on this
                :class:`torch.nn.Module`. Otherwise, the provided
                ``hook`` will be fired after all existing ``forward`` hooks on
                this :class:`torch.nn.Module`. Note that global
                ``forward`` hooks registered with
                :func:`register_module_forward_hook` will fire before all hooks
                registered by this method.
                Default: ``False``
            with_kwargs (bool): If ``True``, the ``hook`` will be passed the
                kwargs given to the forward function.
                Default: ``False``
            always_call (bool): If ``True`` the ``hook`` will be run regardless of
                whether an exception is raised while calling the Module.
                Default: ``False``

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
    def _slow_forward(self, *input, **kwargs): ...
    def _wrapped_call_impl(self, *args, **kwargs): ...
    def _call_impl(self, *args, **kwargs): ...
    __call__: Callable[..., Any]
    def __getstate__(self): ...
    def __setstate__(self, state) -> None: ...
    def __getattr__(self, name: str) -> Tensor | Module: ...
    def __setattr__(self, name: str, value: Tensor | Module) -> None: ...
    def __delattr__(self, name) -> None: ...
    def _register_state_dict_hook(self, hook):
        """Register a post-hook for the :meth:`~torch.nn.Module.state_dict` method.

        It should have the following signature::
            hook(module, state_dict, prefix, local_metadata) -> None or state_dict

        The registered hooks can modify the ``state_dict`` inplace or return a new one.
        If a new ``state_dict`` is returned, it will only be respected if it is the root
        module that :meth:`~nn.Module.state_dict` is called from.
        """
    def register_state_dict_post_hook(self, hook):
        """Register a post-hook for the :meth:`~torch.nn.Module.state_dict` method.

        It should have the following signature::
            hook(module, state_dict, prefix, local_metadata) -> None

        The registered hooks can modify the ``state_dict`` inplace.
        """
    def register_state_dict_pre_hook(self, hook):
        """Register a pre-hook for the :meth:`~torch.nn.Module.state_dict` method.

        It should have the following signature::
            hook(module, prefix, keep_vars) -> None

        The registered hooks can be used to perform pre-processing before the ``state_dict``
        call is made.
        """
    def _save_to_state_dict(self, destination, prefix, keep_vars) -> None:
        """Save module state to the `destination` dictionary.

        The `destination` dictionary will contain the state
        of the module, but not its descendants. This is called on every
        submodule in :meth:`~torch.nn.Module.state_dict`.

        In rare cases, subclasses can achieve class-specific behavior by
        overriding this method with custom logic.

        Args:
            destination (dict): a dict where state will be stored
            prefix (str): the prefix for parameters and buffers used in this
                module
        """
    T_destination = TypeVar('T_destination', bound=dict[str, Any])
    @overload
    def state_dict(self, *, destination: T_destination, prefix: str = ..., keep_vars: bool = ...) -> T_destination: ...
    @overload
    def state_dict(self, *, prefix: str = ..., keep_vars: bool = ...) -> dict[str, Any]: ...
    def _register_load_state_dict_pre_hook(self, hook, with_module: bool = False):
        """See :meth:`~torch.nn.Module.register_load_state_dict_pre_hook` for details.

        A subtle difference is that if ``with_module`` is set to ``False``, then the
        hook will not take the ``module`` as the first argument whereas
        :meth:`~torch.nn.Module.register_load_state_dict_pre_hook` always takes the
        ``module`` as the first argument.

        Arguments:
            hook (Callable): Callable hook that will be invoked before
                loading the state dict.
            with_module (bool, optional): Whether or not to pass the module
                instance to the hook as the first parameter.
        """
    def register_load_state_dict_pre_hook(self, hook):
        """Register a pre-hook to be run before module's :meth:`~nn.Module.load_state_dict` is called.

        It should have the following signature::
            hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None  # noqa: B950

        Arguments:
            hook (Callable): Callable hook that will be invoked before
                loading the state dict.
        """
    def register_load_state_dict_post_hook(self, hook):
        """Register a post-hook to be run after module's :meth:`~nn.Module.load_state_dict` is called.

        It should have the following signature::
            hook(module, incompatible_keys) -> None

        The ``module`` argument is the current module that this hook is registered
        on, and the ``incompatible_keys`` argument is a ``NamedTuple`` consisting
        of attributes ``missing_keys`` and ``unexpected_keys``. ``missing_keys``
        is a ``list`` of ``str`` containing the missing keys and
        ``unexpected_keys`` is a ``list`` of ``str`` containing the unexpected keys.

        The given incompatible_keys can be modified inplace if needed.

        Note that the checks performed when calling :func:`load_state_dict` with
        ``strict=True`` are affected by modifications the hook makes to
        ``missing_keys`` or ``unexpected_keys``, as expected. Additions to either
        set of keys will result in an error being thrown when ``strict=True``, and
        clearing out both missing and unexpected keys will avoid an error.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None:
        '''Copy parameters and buffers from :attr:`state_dict` into only this module, but not its descendants.

        This is called on every submodule
        in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
        For state dicts without metadata, :attr:`local_metadata` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at `local_metadata.get("version", None)`.
        Additionally, :attr:`local_metadata` can also contain the key
        `assign_to_params_buffers` that indicates whether keys should be
        assigned their corresponding tensor in the state_dict.

        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
            it can be modified.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            local_metadata (dict): a dict containing the metadata for this module.
                See
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (list of str): if ``strict=True``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=True``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`
        '''
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        """Copy parameters and buffers from :attr:`state_dict` into this module and its descendants.

        If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~torch.nn.Module.state_dict` function.

        .. warning::
            If :attr:`assign` is ``True`` the optimizer must be created after
            the call to :attr:`load_state_dict` unless
            :func:`~torch.__future__.get_swap_module_params_on_conversion` is ``True``.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
            assign (bool, optional): When set to ``False``, the properties of the tensors
                in the current module are preserved whereas setting it to ``True`` preserves
                properties of the Tensors in the state dict. The only
                exception is the ``requires_grad`` field of :class:`~torch.nn.Parameter`
                for which the value from the module is preserved. Default: ``False``

        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * ``missing_keys`` is a list of str containing any keys that are expected
                    by this module but missing from the provided ``state_dict``.
                * ``unexpected_keys`` is a list of str containing the keys that are not
                    expected by this module but present in the provided ``state_dict``.

        Note:
            If a parameter or buffer is registered as ``None`` and its corresponding key
            exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
            ``RuntimeError``.
        """
    def _named_members(self, get_members_fn, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True):
        """Help yield various names + members of modules."""
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        '''Return an iterator over module parameters.

        This is typically passed to an optimizer.

        Args:
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Yields:
            Parameter: module parameter

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for param in model.parameters():
            >>>     print(type(param), param.size())
            <class \'torch.Tensor\'> (20L,)
            <class \'torch.Tensor\'> (20L, 1L, 5L, 5L)

        '''
    def named_parameters(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> Iterator[tuple[str, Parameter]]:
        '''Return an iterator over module parameters, yielding both the name of the parameter as well as the parameter itself.

        Args:
            prefix (str): prefix to prepend to all parameter names.
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.
            remove_duplicate (bool, optional): whether to remove the duplicated
                parameters in the result. Defaults to True.

        Yields:
            (str, Parameter): Tuple containing the name and parameter

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for name, param in self.named_parameters():
            >>>     if name in [\'bias\']:
            >>>         print(param.size())

        '''
    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        '''Return an iterator over module buffers.

        Args:
            recurse (bool): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module.

        Yields:
            torch.Tensor: module buffer

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for buf in model.buffers():
            >>>     print(type(buf), buf.size())
            <class \'torch.Tensor\'> (20L,)
            <class \'torch.Tensor\'> (20L, 1L, 5L, 5L)

        '''
    def named_buffers(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> Iterator[tuple[str, Tensor]]:
        '''Return an iterator over module buffers, yielding both the name of the buffer as well as the buffer itself.

        Args:
            prefix (str): prefix to prepend to all buffer names.
            recurse (bool, optional): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module. Defaults to True.
            remove_duplicate (bool, optional): whether to remove the duplicated buffers in the result. Defaults to True.

        Yields:
            (str, torch.Tensor): Tuple containing the name and buffer

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for name, buf in self.named_buffers():
            >>>     if name in [\'running_var\']:
            >>>         print(buf.size())

        '''
    def children(self) -> Iterator['Module']:
        """Return an iterator over immediate children modules.

        Yields:
            Module: a child module
        """
    def named_children(self) -> Iterator[tuple[str, 'Module']]:
        '''Return an iterator over immediate children modules, yielding both the name of the module as well as the module itself.

        Yields:
            (str, Module): Tuple containing a name and child module

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for name, module in model.named_children():
            >>>     if name in [\'conv4\', \'conv5\']:
            >>>         print(module)

        '''
    def modules(self) -> Iterator['Module']:
        """Return an iterator over all modules in the network.

        Yields:
            Module: a module in the network

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

        Example::

            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.modules()):
            ...     print(idx, '->', m)

            0 -> Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            )
            1 -> Linear(in_features=2, out_features=2, bias=True)

        """
    def named_modules(self, memo: set['Module'] | None = None, prefix: str = '', remove_duplicate: bool = True):
        """Return an iterator over all modules in the network, yielding both the name of the module as well as the module itself.

        Args:
            memo: a memo to store the set of modules already added to the result
            prefix: a prefix that will be added to the name of the module
            remove_duplicate: whether to remove the duplicated module instances in the result
                or not

        Yields:
            (str, Module): Tuple of name and module

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

        Example::

            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.named_modules()):
            ...     print(idx, '->', m)

            0 -> ('', Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            ))
            1 -> ('0', Linear(in_features=2, out_features=2, bias=True))

        """
    def train(self, mode: bool = True) -> Self:
        """Set the module in training mode.

        This has an effect only on certain modules. See the documentation of
        particular modules for details of their behaviors in training/evaluation
        mode, i.e., whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
    def eval(self) -> Self:
        """Set the module in evaluation mode.

        This has an effect only on certain modules. See the documentation of
        particular modules for details of their behaviors in training/evaluation
        mode, i.e. whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.

        See :ref:`locally-disable-grad-doc` for a comparison between
        `.eval()` and several similar mechanisms that may be confused with it.

        Returns:
            Module: self
        """
    def requires_grad_(self, requires_grad: bool = True) -> Self:
        """Change if autograd should record operations on parameters in this module.

        This method sets the parameters' :attr:`requires_grad` attributes
        in-place.

        This method is helpful for freezing part of the module for finetuning
        or training parts of a model individually (e.g., GAN training).

        See :ref:`locally-disable-grad-doc` for a comparison between
        `.requires_grad_()` and several similar mechanisms that may be confused with it.

        Args:
            requires_grad (bool): whether autograd should record operations on
                                  parameters in this module. Default: ``True``.

        Returns:
            Module: self
        """
    def zero_grad(self, set_to_none: bool = True) -> None:
        """Reset gradients of all model parameters.

        See similar function under :class:`torch.optim.Optimizer` for more context.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                See :meth:`torch.optim.Optimizer.zero_grad` for details.
        """
    def share_memory(self) -> Self:
        """See :meth:`torch.Tensor.share_memory_`."""
    def _get_name(self): ...
    def extra_repr(self) -> str:
        """Return the extra representation of the module.

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
    def __repr__(self) -> str: ...
    def __dir__(self): ...
    def _replicate_for_data_parallel(self): ...
    def compile(self, *args, **kwargs) -> None:
        """
        Compile this Module's forward using :func:`torch.compile`.

        This Module's `__call__` method is compiled and all arguments are passed as-is
        to :func:`torch.compile`.

        See :func:`torch.compile` for details on the arguments for this function.
        """
