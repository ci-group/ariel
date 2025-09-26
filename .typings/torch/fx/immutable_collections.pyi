from typing import TypeVar
from typing_extensions import Self

__all__ = ['immutable_list', 'immutable_dict']

_T = TypeVar('_T')
_KT = TypeVar('_KT')
_VT = TypeVar('_VT')

class immutable_list(list[_T]):
    """An immutable version of :class:`list`."""
    __delitem__ = _no_mutation
    __iadd__ = _no_mutation
    __imul__ = _no_mutation
    __setitem__ = _no_mutation
    append = _no_mutation
    clear = _no_mutation
    extend = _no_mutation
    insert = _no_mutation
    pop = _no_mutation
    remove = _no_mutation
    reverse = _no_mutation
    sort = _no_mutation
    def __hash__(self) -> int: ...
    def __reduce__(self) -> tuple[type[Self], tuple[tuple[_T, ...]]]: ...

class immutable_dict(dict[_KT, _VT]):
    """An immutable version of :class:`dict`."""
    __delitem__ = _no_mutation
    __ior__ = _no_mutation
    __setitem__ = _no_mutation
    clear = _no_mutation
    pop = _no_mutation
    popitem = _no_mutation
    setdefault = _no_mutation
    update = _no_mutation
    def __hash__(self) -> int: ...
    def __reduce__(self) -> tuple[type[Self], tuple[tuple[tuple[_KT, _VT], ...]]]: ...
