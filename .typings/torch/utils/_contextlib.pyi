from typing import Any, Callable, TypeVar

FuncType = Callable[..., Any]
F = TypeVar('F', bound=FuncType)

def _wrap_generator(ctx_factory, func):
    """
    Wrap each generator invocation with the context manager factory.

    The input should be a function that returns a context manager,
    not a context manager itself, to handle one-shot context managers.
    """
def context_decorator(ctx, func):
    """
    Like contextlib.ContextDecorator.

    But with the following differences:
    1. Is done by wrapping, rather than inheritance, so it works with context
       managers that are implemented from C and thus cannot easily inherit from
       Python classes
    2. Wraps generators in the intuitive way (c.f. https://bugs.python.org/issue37743)
    3. Errors out if you try to wrap a class, because it is ambiguous whether
       or not you intended to wrap only the constructor

    The input argument can either be a context manager (in which case it must
    be a multi-shot context manager that can be directly invoked multiple times)
    or a callable that produces a context manager.
    """

class _DecoratorContextManager:
    """Allow a context manager to be used as a decorator."""
    def __call__(self, orig_func: F) -> F: ...
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None: ...
    def clone(self): ...

class _NoParamDecoratorContextManager(_DecoratorContextManager):
    """Allow a context manager to be used as a decorator without parentheses."""
    def __new__(cls, orig_func=None): ...
