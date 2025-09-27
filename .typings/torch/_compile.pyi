from typing import Callable, Literal, TypeVar, overload
from typing_extensions import ParamSpec

_T = TypeVar('_T')
_P = ParamSpec('_P')

@overload
def _disable_dynamo(fn: Callable[_P, _T], recursive: bool = True) -> Callable[_P, _T]: ...
@overload
def _disable_dynamo(fn: Literal[None] = None, recursive: bool = True) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...
