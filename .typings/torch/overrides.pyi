import contextlib
import functools
import types
from _typeshed import Incomplete
from collections.abc import Generator, Iterable
from typing import Any, Callable, TypeVar
from typing_extensions import ParamSpec

__all__ = ['get_ignored_functions', 'get_overridable_functions', 'get_testing_overrides', 'handle_torch_function', 'has_torch_function', 'resolve_name', 'is_tensor_like', 'is_tensor_method_or_property', 'wrap_torch_function', 'enable_reentrant_dispatch']

_P = ParamSpec('_P')
_R = TypeVar('_R')

@functools.cache
@_disable_user_warnings
def get_ignored_functions() -> set[Callable]:
    """
    Return public functions that cannot be overridden by ``__torch_function__``.

    Returns
    -------
    set[Callable]
        A tuple of functions that are publicly available in the torch API but cannot
        be overridden with ``__torch_function__``. Mostly this is because none of the
        arguments of these functions are tensors or tensor-likes.

    Examples
    --------
    >>> torch.Tensor.as_subclass in torch.overrides.get_ignored_functions()
    True
    >>> torch.add in torch.overrides.get_ignored_functions()
    False
    """
@functools.cache
@_disable_user_warnings
def get_testing_overrides() -> dict[Callable, Callable]:
    """Return a dict containing dummy overrides for all overridable functions

    Returns
    -------
    Dict[Callable, Callable]
        A dictionary that maps overridable functions in the PyTorch API to
        lambda functions that have the same signature as the real function
        and unconditionally return -1. These lambda functions are useful
        for testing API coverage for a type that defines ``__torch_function__``.

    Examples
    --------
    >>> import inspect
    >>> my_add = torch.overrides.get_testing_overrides()[torch.add]
    >>> inspect.signature(my_add)
    <Signature (input, other, out=None)>
    """
def wrap_torch_function(dispatcher: Callable):
    """Wraps a given function with ``__torch_function__`` -related functionality.

    Parameters
    ----------
    dispatcher: Callable
        A callable that returns an iterable of Tensor-likes passed into the function.

    Note
    ----
    This decorator may reduce the performance of your code. Generally, it's enough to express
    your code as a series of functions that, themselves, support __torch_function__. If you
    find yourself in the rare situation where this is not the case, e.g. if you're wrapping a
    low-level library and you also need it to work for Tensor-likes, then this function is available.

    Examples
    --------
    >>> def dispatcher(a):  # Must have the same signature as func
    ...     return (a,)
    >>> @torch.overrides.wrap_torch_function(dispatcher)
    >>> def func(a):  # This will make func dispatchable by __torch_function__
    ...     return a + 0
    """
def handle_torch_function(public_api: Callable, relevant_args: Iterable[Any], *args, **kwargs) -> Any:
    """Implement a function with checks for ``__torch_function__`` overrides.

    See torch::autograd::handle_torch_function for the equivalent of this
    function in the C++ implementation.

    Arguments
    ---------
    public_api : function
        Function exposed by the public torch API originally called like
        ``public_api(*args, **kwargs)`` on which arguments are now being
        checked.
    relevant_args : iterable
        Iterable of arguments to check for __torch_function__ methods.
    args : tuple
        Arbitrary positional arguments originally passed into ``public_api``.
    kwargs : tuple
        Arbitrary keyword arguments originally passed into ``public_api``.

    Returns
    -------
    object
        Result from calling ``implementation`` or an ``__torch_function__``
        method, as appropriate.

    Raises
    ------
    TypeError : if no implementation is found.

    Example
    -------
    >>> def func(a):
    ...     if has_torch_function_unary(a):
    ...         return handle_torch_function(func, (a,), a)
    ...     return a + 0
    """

has_torch_function: Incomplete

@_disable_user_warnings
def get_overridable_functions() -> dict[Any, list[Callable]]:
    """List functions that are overridable via __torch_function__

    Returns
    -------
    Dict[Any, List[Callable]]
        A dictionary that maps namespaces that contain overridable functions
        to functions in that namespace that can be overridden.
    """
@_disable_user_warnings
def resolve_name(f):
    """Get a human readable string name for a function passed to
    __torch_function__

    Arguments
    ---------
    f : Callable
        Function to resolve the name of.

    Returns
    -------
    str
        Name of the function; if eval'ed it should give back the input
        function.
    """
@_disable_user_warnings
def is_tensor_method_or_property(func: Callable) -> bool:
    """
    Returns True if the function passed in is a handler for a
    method or property belonging to ``torch.Tensor``, as passed
    into ``__torch_function__``.

    .. note::
       For properties, their ``__get__`` method must be passed in.

    This may be needed, in particular, for the following reasons:

    1. Methods/properties sometimes don't contain a `__module__` slot.
    2. They require that the first passed-in argument is an instance
       of ``torch.Tensor``.

    Examples
    --------
    >>> is_tensor_method_or_property(torch.Tensor.add)
    True
    >>> is_tensor_method_or_property(torch.add)
    False
    """
def is_tensor_like(inp):
    """
    Returns ``True`` if the passed-in input is a Tensor-like.

    Currently, this occurs whenever there's a ``__torch_function__``
    attribute on the type of the input.

    Examples
    --------
    A subclass of tensor is generally a Tensor-like.

    >>> class SubTensor(torch.Tensor): ...
    >>> is_tensor_like(SubTensor([0]))
    True

    Built-in or user types aren't usually Tensor-like.

    >>> is_tensor_like(6)
    False
    >>> is_tensor_like(None)
    False
    >>> class NotATensor: ...
    >>> is_tensor_like(NotATensor())
    False

    But, they can be made Tensor-like by implementing __torch_function__.

    >>> class TensorLike:
    ...     @classmethod
    ...     def __torch_function__(cls, func, types, args, kwargs):
    ...         return -1
    >>> is_tensor_like(TensorLike())
    True
    """

class TorchFunctionMode:
    """
    A ``TorchFunctionMode`` allows you to override the meaning of all
    ``__torch_function__`` overrideable functions within a dynamic scope,
    without having to actually create a tensor subclass or manually
    monkey-patch functions in the PyTorch API.  Some common situations
    where you should use a mode:

        * You want to override the meaning of factory functions, or other
          functions that do not otherwise take a tensor as an argument
          (these cannot be overridden with tensor subclasses).

        * You want to override the behavior of all functions without needing
          to wrap your inputs in tensor subclasses; e.g., if you are just
          interested in logging intermediate computations.

        * You want to control the order of execution of various tensor
          subclasses explicitly, rather than implicitly via the return of
          ``NotImplemented``.

    Independent subclasses of :class:`TorchFunctionMode` are compositional:
    modes can be pushed onto a stack using ``with MyMode():``.
    When you call functions in the PyTorch API inside your
    ``__torch_function__`` implementation, by default, they will forward on to
    the next mode on the mode stack.  If you want recursively call back into
    your current ``__torch_function__`` implementation, either explicitly
    invoke ``self.__torch_function__(...)``, or use the context manager
    ``enable_torch_function_mode(self, replace=self.inner)`` to make PyTorch
    API self-referential (beware of infinite loops, in this case!)
    """
    inner: TorchFunctionMode
    def __init__(self) -> None: ...
    def __torch_function__(self, func, types, args=(), kwargs=None) -> None: ...
    def __enter__(self): ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: types.TracebackType | None) -> None: ...
    @classmethod
    def push(cls, *args, **kwargs): ...

class BaseTorchFunctionMode(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None): ...

@contextlib.contextmanager
def enable_reentrant_dispatch() -> Generator[None]: ...
