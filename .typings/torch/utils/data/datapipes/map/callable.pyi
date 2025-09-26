from torch.utils.data.datapipes.datapipe import MapDataPipe
from typing import Callable, TypeVar

__all__ = ['MapperMapDataPipe', 'default_fn']

_T_co = TypeVar('_T_co', covariant=True)

def default_fn(data): ...

class MapperMapDataPipe(MapDataPipe[_T_co]):
    """
    Apply the input function over each item from the source DataPipe (functional name: ``map``).

    The function can be any regular Python function or partial object. Lambda
    function is not recommended as it is not supported by pickle.

    Args:
        datapipe: Source MapDataPipe
        fn: Function being applied to each item

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper, Mapper
        >>> def add_one(x):
        ...     return x + 1
        >>> dp = SequenceWrapper(range(10))
        >>> map_dp_1 = dp.map(add_one)
        >>> list(map_dp_1)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> map_dp_2 = Mapper(dp, lambda x: x + 1)
        >>> list(map_dp_2)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    """
    datapipe: MapDataPipe
    fn: Callable
    def __init__(self, datapipe: MapDataPipe, fn: Callable = ...) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index) -> _T_co: ...
