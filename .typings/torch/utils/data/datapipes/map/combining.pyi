from torch.utils.data.datapipes.datapipe import MapDataPipe
from typing import TypeVar

__all__ = ['ConcaterMapDataPipe', 'ZipperMapDataPipe']

_T_co = TypeVar('_T_co', covariant=True)

class ConcaterMapDataPipe(MapDataPipe):
    """
    Concatenate multiple Map DataPipes (functional name: ``concat``).

    The new index of is the cumulative sum of source DataPipes.
    For example, if there are 2 source DataPipes both with length 5,
    index 0 to 4 of the resulting `ConcatMapDataPipe` would refer to
    elements of the first DataPipe, and 5 to 9 would refer to elements
    of the second DataPipe.

    Args:
        datapipes: Map DataPipes being concatenated

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> dp1 = SequenceWrapper(range(3))
        >>> dp2 = SequenceWrapper(range(3))
        >>> concat_dp = dp1.concat(dp2)
        >>> list(concat_dp)
        [0, 1, 2, 0, 1, 2]
    """
    datapipes: tuple[MapDataPipe]
    def __init__(self, *datapipes: MapDataPipe) -> None: ...
    def __getitem__(self, index) -> _T_co: ...
    def __len__(self) -> int: ...

class ZipperMapDataPipe(MapDataPipe[tuple[_T_co, ...]]):
    """
    Aggregates elements into a tuple from each of the input DataPipes (functional name: ``zip``).

    This MataPipe is out of bound as soon as the shortest input DataPipe is exhausted.

    Args:
        *datapipes: Map DataPipes being aggregated

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> dp1 = SequenceWrapper(range(3))
        >>> dp2 = SequenceWrapper(range(10, 13))
        >>> zip_dp = dp1.zip(dp2)
        >>> list(zip_dp)
        [(0, 10), (1, 11), (2, 12)]
    """
    datapipes: tuple[MapDataPipe[_T_co], ...]
    def __init__(self, *datapipes: MapDataPipe[_T_co]) -> None: ...
    def __getitem__(self, index) -> tuple[_T_co, ...]: ...
    def __len__(self) -> int: ...
