import abc
import contextlib
import torch
from _typeshed import Incomplete
from collections.abc import Generator, Iterable, MutableMapping, Sequence
from torch._ops import OpOverload
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.hooks import RemovableHandle
from typing import Any, Callable, Literal, NamedTuple
from typing_extensions import TypeAlias

__all__ = ['saved_tensors_hooks', 'save_on_cpu', 'disable_saved_tensors_hooks', 'register_multi_grad_hook', 'allow_mutation_on_saved_tensors', 'Node', 'GradientEdge', 'get_gradient_edge', 'increment_version']

class Node(abc.ABC, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def name(self) -> str:
        """Return the name.

        Example::

            >>> import torch
            >>> a = torch.tensor([0., 0., 0.], requires_grad=True)
            >>> b = a.clone()
            >>> assert isinstance(b.grad_fn, torch.autograd.graph.Node)
            >>> print(b.grad_fn.name())
            CloneBackward0
        """
    @property
    @abc.abstractmethod
    def next_functions(self) -> tuple[tuple[Node | None, int], ...]: ...
    @abc.abstractmethod
    def metadata(self) -> dict:
        """Return the metadata."""
    @property
    @abc.abstractmethod
    def _input_metadata(self) -> list[Any]: ...
    @abc.abstractmethod
    def _register_hook_dict(self, tensor: torch.Tensor) -> None: ...
    @abc.abstractmethod
    def register_hook(self, fn: Callable[..., Any]) -> RemovableHandle:
        """Register a backward hook.

        The hook will be called every time a gradient with respect to the
        Node is computed. The hook should have the following signature::

            hook(grad_inputs: Tuple[Tensor], grad_outputs: Tuple[Tensor]) -> Tuple[Tensor] or None


        The hook should not modify its argument, but it can optionally return
        a new gradient which will be used in place of :attr:`grad_inputs`.

        This function returns a handle with a method ``handle.remove()``
        that removes the hook from the module.

        .. note::
            See :ref:`backward-hooks-execution` for more information on how when this hook
            is executed, and how its execution is ordered relative to other hooks.

        .. note::
            In the rare case where the hook is registered while the Node has already
            begun execution, there is no longer any guarantee on :attr:`grad_outputs`
            content (it might be as usual or empty depending on other factors). The
            hook can still optionally return a new gradient to be used in place of
            :attr:`grad_inputs` independent of :attr:`grad_outputs`.

        Example::

            >>> import torch
            >>> a = torch.tensor([0., 0., 0.], requires_grad=True)
            >>> b = a.clone()
            >>> assert isinstance(b.grad_fn, torch.autograd.graph.Node)
            >>> handle = b.grad_fn.register_hook(lambda gI, gO: (gO[0] * 2,))
            >>> b.sum().backward(retain_graph=True)
            >>> print(a.grad)
            tensor([2., 2., 2.])
            >>> handle.remove() # Removes the hook
            >>> a.grad = None
            >>> b.sum().backward(retain_graph=True)
            >>> print(a.grad)
            tensor([1., 1., 1.])
        """
    @abc.abstractmethod
    def register_prehook(self, fn: Callable[..., Any]) -> RemovableHandle:
        """Register a backward pre-hook.

        The hook will be called every time a gradient with respect to the
        Node is computed. The hook should have the following signature::

            hook(grad_outputs: Tuple[Tensor]) -> Tuple[Tensor] or None

        The hook should not modify its argument, but it can optionally return
        a new gradient which will be used in place of :attr:`grad_outputs`.

        This function returns a handle with a method ``handle.remove()``
        that removes the hook from the module.

        .. note::
            See :ref:`backward-hooks-execution` for more information on how when this hook
            is executed, and how its execution is ordered relative to other hooks.

        Example::

            >>> a = torch.tensor([0., 0., 0.], requires_grad=True)
            >>> b = a.clone()
            >>> assert isinstance(b.grad_fn, torch.autograd.graph.Node)
            >>> handle = b.grad_fn.register_prehook(lambda gI: (gI[0] * 2,))
            >>> b.sum().backward(retain_graph=True)
            >>> print(a.grad)
            tensor([2., 2., 2.])
            >>> handle.remove()
            >>> a.grad = None
            >>> b.sum().backward(retain_graph=True)
            >>> print(a.grad)
            tensor([1., 1., 1.])
        """
    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool: ...

class GradientEdge(NamedTuple):
    """Object representing a given gradient edge within the autograd graph.

    To get the gradient edge where a given Tensor gradient will be computed,
    you can do ``edge = autograd.graph.get_gradient_edge(tensor)``.
    """
    node: Node
    output_nr: int

def get_gradient_edge(tensor: torch.Tensor) -> GradientEdge:
    """Get the gradient edge for computing the gradient of the given Tensor.

    In particular, it is equivalent to call
    ``g = autograd.grad(loss, input)`` and ``g = autograd.grad(loss, get_gradient_edge(input))``.
    """
def increment_version(tensor: torch.Tensor | Iterable[torch.Tensor]) -> None:
    """Update autograd metadata tracking whether the given Tensor was modified in place.

    This is to enable more accurate error checking within the autograd engine.
    It is already done automatically by PyTorch functions and within custom Function
    when mark_dirty() is called appropriately so you only need to call this explicitly
    if you are doing inplace operation on the Tensor data in a way that Pytorch doesn't
    know about. For example a custom kernel that reads the Tensor data_ptr and modifies
    the memory inplace based on this pointer. Can accept either a tensor, or a list of tensors.

    Note that incrementing the version counter multiple times for a single inplace operation
    is not problematic.

    Note that if you pass in tensor constructed under torch.inference_mode(),
    we will not bump its version counter (because your tensor does not have one).
    """

class saved_tensors_hooks:
    '''Context-manager that sets a pair of pack / unpack hooks for saved tensors.

    Use this context-manager to define how intermediary results of an operation
    should be packed before saving, and unpacked on retrieval.

    In that context, the ``pack_hook`` function will be called everytime an
    operation saves a tensor for backward (this includes intermediary results
    saved using
    :func:`~torch.autograd.function._ContextMethodMixin.save_for_backward` but
    also those recorded by a PyTorch-defined operation). The output of
    ``pack_hook`` is then stored in the computation graph instead of the
    original tensor.

    The ``unpack_hook`` is called when the saved tensor needs to be accessed,
    namely when executing :func:`torch.Tensor.backward()` or
    :func:`torch.autograd.grad()`. It takes as argument the *packed* object
    returned by ``pack_hook`` and should return a tensor which has the same
    content as the original tensor (passed as input to the corresponding
    ``pack_hook``).

    The hooks should have the following signatures:

        pack_hook(tensor: Tensor) -> Any

        unpack_hook(Any) -> Tensor

    where the return value of ``pack_hook`` is a valid input to ``unpack_hook``.

    In general, you want ``unpack_hook(pack_hook(t))`` to be equal to ``t`` in terms
    of value, size, dtype and device.

    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> def pack_hook(x):
        ...     print("Packing", x)
        ...     return x.detach()
        >>>
        >>> def unpack_hook(x):
        ...     print("Unpacking", x)
        ...     return x
        >>>
        >>> a = torch.ones(5, requires_grad=True)
        >>> b = torch.ones(5, requires_grad=True) * 2
        >>> with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        ...     y = a * b
        Packing tensor([1., 1., 1., 1., 1.], requires_grad=True)
        Packing tensor([2., 2., 2., 2., 2.], grad_fn=<MulBackward0>)
        >>> y.sum().backward()
        Unpacking tensor([1., 1., 1., 1., 1.], requires_grad=True)
        Unpacking tensor([2., 2., 2., 2., 2.], grad_fn=<MulBackward0>)

    .. warning ::
        Performing an inplace operation on the input to either hooks may lead
        to undefined behavior.

    .. warning ::
        Only one pair of hooks is allowed at a time. When recursively nesting this
        context-manager, only the inner-most pair of hooks will be applied.

    .. warning ::
        To avoid reference cycle, the return value of ``pack_hook`` cannot hold a
        reference to the input tensor. For example, use `lambda x: x.detach()`
        instead of `lambda x: x` as the pack hook.
    '''
    pack_hook: Incomplete
    unpack_hook: Incomplete
    def __init__(self, pack_hook: Callable[[torch.Tensor], Any], unpack_hook: Callable[[Any], torch.Tensor]) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *args: object) -> None: ...

class save_on_cpu(saved_tensors_hooks):
    '''Context manager under which tensors saved by the forward pass will be stored on cpu, then retrieved for backward.

    When performing operations within this context manager, intermediary
    results saved in the graph during the forward pass will be moved to CPU,
    then copied back to the original device when needed for the backward pass.
    If the graph was already on CPU, no tensor copy is performed.

    Use this context-manager to trade compute for GPU memory usage (e.g.
    when your model doesn\'t fit in GPU memory during training).

    Args:
        pin_memory (bool): If ``True`` tensors will be saved to CPU pinned memory
                           during packing and copied to GPU asynchronously during unpacking.
                           Defaults to ``False``.
                           Also see :ref:`cuda-memory-pinning`.


    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> a = torch.randn(5, requires_grad=True, device="cuda")
        >>> b = torch.randn(5, requires_grad=True, device="cuda")
        >>> c = torch.randn(5, requires_grad=True, device="cuda")
        >>>
        >>> def f(a, b, c):
        ...     prod_1 = a * b           # a and b are saved on GPU
        ...     with torch.autograd.graph.save_on_cpu():
        ...         prod_2 = prod_1 * c  # prod_1 and c are saved on CPU
        ...     y = prod_2 * a           # prod_2 and a are saved on GPU
        ...     return y
        >>>
        >>> y = f(a, b, c)
        >>> del a, b, c  # for illustration only
        >>> # the content of a, b, and prod_2 are still alive on GPU
        >>> # the content of prod_1 and c only live on CPU
        >>> y.sum().backward()  # all CPU tensors are moved back to GPU, for backward
        >>> # all intermediary tensors are released (deleted) after the call to backward
    '''
    def __init__(self, pin_memory: bool = False, device_type: str = 'cuda') -> None: ...

@contextlib.contextmanager
def disable_saved_tensors_hooks(error_message: str) -> Generator[None, None, None]:
    '''Context-manager that disables the saved tensors default hooks feature.

    Useful for if you are creating a feature that does not work with saved
    tensors default hooks.

    Args:
        error_message (str): When saved tensors default hooks are used when they
                             have been are disabled, a RuntimeError with this
                             error message gets raised.

    Example::

        >>> # xdoctest: +SKIP(failing)
        >>> message = "saved tensors default hooks are disabled"
        >>> with torch.autograd.graph.disable_saved_tensors_hooks(message):
        ...     # Raises RuntimeError: saved tensors default hooks are disabled
        ...     with torch.autograd.graph.save_on_cpu():
        ...         pass
    '''

class _MultiHandle(RemovableHandle):
    handles: tuple[RemovableHandle, ...]
    def __init__(self, handles: tuple[RemovableHandle, ...]) -> None: ...
    def remove(self) -> None: ...
    def __getstate__(self) -> tuple[RemovableHandle, ...]: ...
    def __setstate__(self, state: tuple[RemovableHandle, ...]) -> None: ...

def register_multi_grad_hook(tensors: Sequence[torch.Tensor], fn: Callable[[Sequence[torch.Tensor | None]], None] | Callable[[torch.Tensor], None], *, mode: Literal['all', 'any'] = 'all') -> RemovableHandle:
    '''Register a multi-grad backward hook.

    There are two supported modes: ``"all"`` and ``"any"``.

    Under the ``"all"`` mode, the hook will be called after gradients with respect to every tensor in
    :attr:`tensors` have been computed. If a tensor is in :attr:`tensors` but
    is not part of the graph, or if a tensor is not needed to compute the gradients
    for any ``inputs`` specified for the current ``.backward()`` or ``.grad()`` call,
    this tensor will be ignored and the hook will not wait for its gradient to be
    computed.

    After every non-ignored tensor\'s gradient has been computed, :attr:`fn` will be
    called with those gradients. ``None`` will be passed for tensors that did not
    have their gradients computed.

    Under the ``"any"`` mode, the hook will be called after the first gradient
    with respect to a tensor in :attr:`tensors` has been computed. The hook
    will be called with that gradient as its argument.

    The hook should not modify its arguments.

    This function returns a handle with a method ``handle.remove()`` that removes the hook.

    .. note::
        See :ref:`backward-hooks-execution` for more information on how when this hook
        is executed, and how its execution is ordered relative to other hooks.

    Example::

        >>> import torch
        >>>
        >>> a = torch.rand(2, 3, requires_grad=True)
        >>> b = torch.rand(2, 3, requires_grad=True)
        >>> c = a * b
        >>> d = a * b
        >>>
        >>> def fn(grads):
        ...     print([g is not None for g in grads])
        ...
        >>> torch.autograd.graph.register_multi_grad_hook((a, b, c, d), fn)
        >>>
        >>> c.sum().backward(retain_graph=True)
        [True, True, True, False]
        >>> c.sum().backward(inputs=(a,), retain_graph=True)
        [True, False, True, False]
        >>>
    '''
_TID: TypeAlias = tuple[int, int, int]
_SID: TypeAlias = tuple[int, int]

class _Handle: ...

class _swap_with_cloned(saved_tensors_hooks):
    def __init__(self, ctx: _AllowMutationOnSavedContext) -> None: ...

class _CloneArgBeforeMutateMode(TorchDispatchMode):
    ctx: Incomplete
    def __init__(self, ctx: _AllowMutationOnSavedContext) -> None: ...
    def __torch_dispatch__(self, func: OpOverload, types: Iterable[type], args: tuple[Any, ...] = (), kwargs: dict[Any, Any] | None = None) -> Any: ...

class _AllowMutationOnSavedContext:
    cloned: MutableMapping[_Handle, torch.Tensor]
    original: MutableMapping[_Handle, torch.Tensor]
    tid_to_weakhandle: MutableMapping[_TID, _Handle]
    sid_to_tid: dict[_SID, set[_TID]]
    def __init__(self) -> None: ...
    def clear(self) -> None: ...

@contextlib.contextmanager
def allow_mutation_on_saved_tensors() -> Generator[_AllowMutationOnSavedContext, None, None]:
    """Context manager under which mutating tensors saved for backward is allowed.

    Under this context manager, tensors saved for backward are cloned on mutation,
    so the original version can still be used during backward. Normally, mutating a tensor
    saved for backward will result in an error raised when it's used during backward.

    To ensure the correct behavior, both the forward and backward should be run under
    the same context manager.

    Returns:
        An _AllowMutationOnSavedContext object storing the state managed by this
        context manager. This object can be useful for debugging purposes. The state
        managed by the context manager is automatically cleared upon exiting.

    Example::

        >>> import torch
        >>> with torch.autograd.graph.allow_mutation_on_saved_tensors():
        ...     # forward
        ...     a = torch.ones(2, 3, requires_grad=True)
        ...     b = a.clone()
        ...     out = (b**2).sum()
        ...     b.sin_()
        ...     # backward
        ...     out.sum().backward()
        ...
        tensor([[0.8415, 0.8415, 0.8415],
                [0.8415, 0.8415, 0.8415]], grad_fn=<SinBackward0>)
    """
