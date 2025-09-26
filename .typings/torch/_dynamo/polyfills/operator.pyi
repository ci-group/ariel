from typing import Any, Callable, TypeVar, overload
from typing_extensions import TypeVarTuple, Unpack

__all__ = ['attrgetter', 'itemgetter', 'methodcaller']

_T = TypeVar('_T')
_T1 = TypeVar('_T1')
_T2 = TypeVar('_T2')
_Ts = TypeVarTuple('_Ts')
_U = TypeVar('_U')
_U1 = TypeVar('_U1')
_U2 = TypeVar('_U2')
_Us = TypeVarTuple('_Us')

@overload
def attrgetter(attr: str, /) -> Callable[[Any], _U]: ...
@overload
def attrgetter(attr1: str, attr2: str, /, *attrs: str) -> Callable[[Any], tuple[_U1, _U2, Unpack[_Us]]]: ...
@overload
def itemgetter(item: _T, /) -> Callable[[Any], _U]: ...
@overload
def itemgetter(item1: _T1, item2: _T2, /, *items: Unpack[_Ts]) -> Callable[[Any], tuple[_U1, _U2, Unpack[_Us]]]: ...
def methodcaller(name: str, /, *args: Any, **kwargs: Any) -> Callable[[Any], Any]: ...
