from _typeshed import Incomplete
from typing import Any, Callable, TypeVar, TypeVarTuple, Unpack
from typing_extensions import ParamSpec

_P = ParamSpec('_P')
_R = TypeVar('_R')
_Ts = TypeVarTuple('_Ts')
_IS_WINDOWS: Incomplete

def _call_function_and_return_exception(func: Callable[[Unpack[_Ts]], _R], args: tuple[Unpack[_Ts]], kwargs: dict[str, Any]) -> _R | Exception:
    """Call function and return a exception if there is one."""
def safe_call(func: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs) -> _R:
    """Call a function in a separate process.

    Args:
        func: The function to call.
        args: The positional arguments to pass to the function.
        kwargs: The keyword arguments to pass to the function.

    Returns:
        The return value of the function.

    Raises:
        Exception: If the function raised an exception.
    """
