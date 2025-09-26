import nevergrad.common.typing as tp
import numpy as np
from _typeshed import Incomplete
from nevergrad.common.tools import OrderedSet as OrderedSet
from nevergrad.parametrization import parameter as p

class MultiValue:
    """Estimation of a value based on one or multiple evaluations.
    This class provides easy access to:
    - count: how many times the point was evaluated
    - mean: the mean value.
    - square: the mean square value
    - variance: the variance
    - parameter: the corresponding Parameter


    It also provides access to optimistic and pessimistic bounds for the value.

    Parameter
    ---------
    parameter: Parameter
        the parameter for one of the evaluations
    y: float
        the first evaluation of the value
    """
    count: int
    mean: Incomplete
    _minimum: Incomplete
    square: Incomplete
    variance: float
    parameter: Incomplete
    _ref: Incomplete
    def __init__(self, parameter: p.Parameter, y: float, *, reference: p.Parameter) -> None: ...
    @property
    def x(self) -> np.ndarray: ...
    @property
    def optimistic_confidence_bound(self) -> float: ...
    @property
    def pessimistic_confidence_bound(self) -> float: ...
    def get_estimation(self, name: str) -> float: ...
    def add_evaluation(self, y: float) -> None:
        """Adds a new evaluation of the value

        Parameter
        ---------
        y: float
            the new evaluation
        """
    def as_array(self, reference: p.Parameter) -> np.ndarray: ...
    def __repr__(self) -> str: ...

def _get_nash(optimizer: tp.Any) -> tp.List[tp.Tuple[tp.Tuple[float, ...], int]]:
    """Returns an empirical distribution. limited using a threshold
    equal to max_num_trials^(1/4).
    """
def sample_nash(optimizer: tp.Any) -> tp.Tuple[float, ...]: ...

class DelayedJob:
    """Future-like object which delays computation"""
    func: Incomplete
    args: Incomplete
    kwargs: Incomplete
    _result: tp.Optional[tp.Any]
    _computed: bool
    def __init__(self, func: tp.Callable[..., tp.Any], *args: tp.Any, **kwargs: tp.Any) -> None: ...
    def done(self) -> bool: ...
    def result(self) -> tp.Any: ...

class SequentialExecutor:
    """Executor which run sequentially and locally
    (just calls the function and returns a FinishedJob)
    """
    def submit(self, fn: tp.Callable[..., tp.Any], *args: tp.Any, **kwargs: tp.Any) -> DelayedJob: ...

def _tobytes(x: tp.ArrayLike) -> bytes: ...

_ERROR_STR: str
Y = tp.TypeVar('Y')

class Archive(tp.Generic[Y]):
    """A dict-like object with numpy arrays as keys.
    The underlying `bytesdict` dict stores the arrays as bytes since arrays are not hashable.
    Keys can be converted back with np.frombuffer(key)
    """
    bytesdict: tp.Dict[bytes, Y]
    def __init__(self) -> None: ...
    def __setitem__(self, x: tp.ArrayLike, value: Y) -> None: ...
    def __getitem__(self, x: tp.ArrayLike) -> Y: ...
    def __contains__(self, x: tp.ArrayLike) -> bool: ...
    def get(self, x: tp.ArrayLike, default: tp.Optional[Y] = None) -> tp.Optional[Y]: ...
    def __len__(self) -> int: ...
    def values(self) -> tp.ValuesView[Y]: ...
    def keys(self) -> None: ...
    def items(self) -> None: ...
    def items_as_array(self) -> tp.Iterator[tp.Tuple[np.ndarray, Y]]: ...
    def items_as_arrays(self) -> tp.Iterator[tp.Tuple[np.ndarray, Y]]:
        """Functions that iterates on key-values but transforms keys
        to np.ndarray. This is to simplify interactions, but should not
        be used in an algorithm since the conversion can be inefficient.
        Prefer using self.bytesdict.items() directly, and convert the bytes
        to np.ndarray using np.frombuffer(b)
        """
    def keys_as_array(self) -> tp.Iterator[np.ndarray]: ...
    def keys_as_arrays(self) -> tp.Iterator[np.ndarray]:
        """Functions that iterates on keys but transforms them
        to np.ndarray. This is to simplify interactions, but should not
        be used in an algorithm since the conversion can be inefficient.
        Prefer using self.bytesdict.keys() directly, and convert the bytes
        to np.ndarray using np.frombuffer(b)
        """
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __iter__(self) -> None: ...

class Pruning:
    '''Callable for pruning archives in the optimizer class.
    See Optimizer.pruning attribute, called at each "tell".

    Parameters
    ----------
    min_len: int
        minimum length of the pruned archive.
    max_len: int
        length at which pruning is activated (maximum allowed length for the archive).

    Note
    ----
    For each of the 3 criteria (optimistic, pessimistic and average), the min_len best (lowest)
    points will be kept, which can lead to at most 3 * min_len points.
    '''
    min_len: Incomplete
    max_len: Incomplete
    _num_prunings: int
    def __init__(self, min_len: int, max_len: int) -> None: ...
    def __call__(self, archive: Archive[MultiValue]) -> Archive[MultiValue]: ...
    def _prune(self, archive: Archive[MultiValue]) -> Archive[MultiValue]: ...
    @classmethod
    def sensible_default(cls, num_workers: int, dimension: int) -> Pruning:
        """Very conservative pruning
        - keep at least 100 elements, or 7 times num_workers, whatever is biggest
        - keep at least 3 x min_len, or up to 10 x min_len if it does not exceed 1gb of data

        Parameters
        ----------
        num_workers: int
            number of evaluations which will be run in parallel at once
        dimension: int
            dimension of the optimization space
        """

class UidQueue:
    """Queue of uids to handle a population. This keeps track of:
    - told uids
    - asked uids
    When telling, it removes from the asked queue and adds to the told queue
    When asking, it takes from the told queue if not empty, else from the older
    asked, and then adds to the asked queue.
    """
    told: Incomplete
    asked: OrderedSet[str]
    def __init__(self) -> None: ...
    def clear(self) -> None:
        """Removes all uids from the queues"""
    def ask(self) -> str:
        """Takes a uid from the told queue if not empty, else from the older asked,
        then adds it to the asked queue.
        """
    def tell(self, uid: str) -> None:
        """Removes the uid from the asked queue and adds to the told queue"""
    def discard(self, uid: str) -> None: ...

class ConstraintManager:
    """Try max_constraints_trials random explorations for satisfying constraints.
    The finally chosen point, if it does not satisfy constraints, is penalized as shown in the penalty function,
    using coeffcieints mentioned here.


    Possibly unstable.
    """
    max_trials: int
    penalty_factor: float
    penalty_exponent: float
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...
    def update(self, max_trials: tp.Optional[int] = None, penalty_factor: tp.Optional[float] = None, penalty_exponent: tp.Optional[float] = None) -> None:
        """
        Parameters
        ----------
            max_trials: int
                number of random tries for satisfying constraints.
            penalty: float
                multiplicative factor on the constraint penalization.
            penalty_exponent: float
                exponent, usually close to 1 and slightly greater than 1.
        """
    def penalty(self, parameter: p.Parameter, num_ask: int, budget: tp.Optional[int]) -> float:
        """Computes the penalty associated with a Parameter, for constraint management"""
