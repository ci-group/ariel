from _typeshed import Incomplete
from collections.abc import Iterator
from torch.utils.data.datapipes.datapipe import IterDataPipe
from typing import Callable, TypeVar

__all__ = ['FilterIterDataPipe']

_T = TypeVar('_T')
_T_co = TypeVar('_T_co', covariant=True)

class FilterIterDataPipe(IterDataPipe[_T_co]):
    """
    Filters out elements from the source datapipe according to input ``filter_fn`` (functional name: ``filter``).

    Args:
        datapipe: Iterable DataPipe being filtered
        filter_fn: Customized function mapping an element to a boolean.
        input_col: Index or indices of data which ``filter_fn`` is applied, such as:

            - ``None`` as default to apply ``filter_fn`` to the data directly.
            - Integer(s) is used for list/tuple.
            - Key(s) is used for dict.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> def is_even(n):
        ...     return n % 2 == 0
        >>> dp = IterableWrapper(range(5))
        >>> filter_dp = dp.filter(filter_fn=is_even)
        >>> list(filter_dp)
        [0, 2, 4]
    """
    datapipe: IterDataPipe[_T_co]
    filter_fn: Callable
    input_col: Incomplete
    def __init__(self, datapipe: IterDataPipe[_T_co], filter_fn: Callable, input_col=None) -> None: ...
    def _apply_filter_fn(self, data) -> bool: ...
    def __iter__(self) -> Iterator[_T_co]: ...
    def _returnIfTrue(self, data: _T) -> tuple[bool, _T]: ...
