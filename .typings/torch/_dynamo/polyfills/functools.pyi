from collections.abc import Iterable
from typing import Callable, TypeVar

__all__ = ['reduce']

_T = TypeVar('_T')
_U = TypeVar('_U')

class _INITIAL_MISSING: ...

def reduce(function: Callable[[_U, _T], _U], iterable: Iterable[_T], initial: _U = ..., /) -> _U: ...
