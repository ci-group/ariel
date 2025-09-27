from _typeshed import Incomplete
from collections.abc import Iterator
from typing import Callable, TypeVar
from typing_extensions import Self

__all__ = ['CyclingIterator']

_T = TypeVar('_T')

class CyclingIterator(Iterator[_T]):
    '''
    An iterator decorator that cycles through the
    underlying iterator "n" times. Useful to "unroll"
    the dataset across multiple training epochs.

    The generator function is called as ``generator_fn(epoch)``
    to obtain the underlying iterator, where ``epoch`` is a
    number less than or equal to ``n`` representing the ``k``th cycle

    For example if ``generator_fn`` always returns ``[1,2,3]``
    then ``CyclingIterator(n=2, generator_fn)`` will iterate through
    ``[1,2,3,1,2,3]``
    '''
    _n: Incomplete
    _epoch: Incomplete
    _generator_fn: Incomplete
    _iter: Incomplete
    def __init__(self, n: int, generator_fn: Callable[[int], Iterator[_T]], start_epoch: int = 0) -> None: ...
    def __iter__(self) -> Self: ...
    def __next__(self) -> _T: ...
