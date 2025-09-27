import collections
import contextlib
import typing as tp
from _typeshed import Incomplete
from nevergrad.common import sphere as sphere

def quasi_randomize(x: tp.Iterable[tp.Any], method: str = 'none') -> tp.Any: ...
def pytorch_import_fix() -> None:
    '''Hackfix needed before pytorch import ("dlopen: cannot load any more object with static TLS")
    See issue #305
    '''
def pairwise(iterable: tp.Iterable[tp.Any]) -> tp.Iterator[tuple[tp.Any, tp.Any]]:
    """Returns an iterator over sliding pairs of the input iterator
    s -> (s0,s1), (s1,s2), (s2, s3), ...

    Note
    ----
    Nothing will be returned if length of iterator is strictly less
    than 2.
    """
def grouper(iterable: tp.Iterable[tp.Any], n: int, fillvalue: tp.Any = None) -> tp.Iterator[list[tp.Any]]:
    '''Collect data into fixed-length chunks or blocks
    Copied from itertools recipe documentation
    Example: grouper(\'ABCDEFG\', 3, \'x\') --> ABC DEF Gxx"
    '''
def roundrobin(*iterables: tp.Iterable[tp.Any]) -> tp.Iterator[tp.Any]:
    """roundrobin('ABC', 'D', 'EF') --> A D E B F C"""

class Sleeper:
    """Simple object for managing the waiting time of a job

    Parameters
    ----------
    min_sleep: float
        minimum sleep time
    max_sleep: float
        maximum sleep time
    averaging_size: int
        size for averaging the registered durations
    """
    _min: Incomplete
    _max: Incomplete
    _start: float | None
    _queue: tp.Deque[float]
    _num_waits: int
    def __init__(self, min_sleep: float = 1e-07, max_sleep: float = 1.0, averaging_size: int = 10) -> None: ...
    def start_timer(self) -> None: ...
    def stop_timer(self) -> None: ...
    def _get_advised_sleep_duration(self) -> float: ...
    def sleep(self) -> None: ...
X = tp.TypeVar('X', bound=tp.Hashable)

class OrderedSet(tp.MutableSet[X]):
    """Set of elements retaining the insertion order
    All new elements are appended to the end of the set.
    """
    _data: collections.OrderedDict[X, int]
    _global_index: int
    def __init__(self, keys: tp.Iterable[X] | None = None) -> None: ...
    def add(self, key: X) -> None: ...
    def popright(self) -> X: ...
    def discard(self, key: X) -> None: ...
    def __contains__(self, key: tp.Any) -> bool: ...
    def __iter__(self) -> tp.Iterator[X]: ...
    def __len__(self) -> int: ...

def different_from_defaults(*, instance: tp.Any, instance_dict: dict[str, tp.Any] | None = None, check_mismatches: bool = False) -> dict[str, tp.Any]:
    """Checks which attributes are different from defaults arguments

    Parameters
    ----------
    instance: object
        the object to change
    instance_dict: dict
        the dict corresponding to the instance, if not provided it's self.__dict__
    check_mismatches: bool
        checks that the attributes match the parameters

    Note
    ----
    This is convenient for short repr of data structures
    """
@contextlib.contextmanager
def set_env(**environ: tp.Any) -> tp.Generator[None, None, None]:
    """Temporarily changes environment variables."""
def flatten(obj: tp.Any) -> tp.Any:
    '''Flatten a dict/list structure

    Example
    -------

    >>> flatten(["a", {"truc": [4, 5]}])
    >>> {"0": "a", "1.truc.0": 4, "1.truc.1": 5}
    '''
