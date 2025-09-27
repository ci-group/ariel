import nevergrad.common.typing as tp
import numpy as np
from . import _datalayers as _datalayers, container as container, core as core, discretization as discretization
from .data import Array as Array
from _typeshed import Incomplete

C = tp.TypeVar('C', bound='Choice')
T = tp.TypeVar('T', bound='TransitionChoice')

class BaseChoice(container.Container):
    _repetitions: Incomplete
    def __init__(self, *, choices: tp.Union[int, tp.Iterable[tp.Any]], repetitions: tp.Optional[int] = None, **kwargs: tp.Any) -> None: ...
    def __len__(self) -> int:
        """Number of choices"""
    def _get_parameters_str(self) -> str: ...
    @property
    def index(self) -> int:
        """Index of the chosen option"""
    @property
    def indices(self) -> Array:
        """Array of indices of the chosen option"""
    @property
    def choices(self) -> container.Tuple:
        """The different options, as a Tuple Parameter"""
    def _layered_get_value(self) -> tp.Any: ...
    def _layered_set_value(self, value: tp.List[tp.Any]) -> None:
        """Must be adapted to each class
        This handles a list of values, not just one
        """
    def get_value_hash(self) -> tp.Hashable: ...

class Choice(BaseChoice):
    '''Unordered categorical parameter, randomly choosing one of the provided choice options as a value.
    The choices can be Parameters, in which case there value will be returned instead.
    The chosen parameter is drawn randomly from the softmax of weights which are
    updated during the optimization.

    Parameters
    ----------
    choices: list or int
        a list of possible values or Parameters for the variable (or an integer as a shortcut for range(num))
    repetitions: None or int
        set to an integer :code:`n` if you want :code:`n` similar choices sampled independently (each with its own distribution)
        This is equivalent to :code:`Tuple(*[Choice(options) for _ in range(n)])` but can be
        30x faster for large :code:`n`.
    deterministic: bool
        whether to always draw the most likely choice (hence avoiding the stochastic behavior, but loosing
        continuity)

    Note
    ----
    - Since the chosen value is drawn randomly, the use of this variable makes deterministic
      functions become stochastic, hence "adding noise"
    - the "mutate" method only mutates the weights and the chosen Parameter (if it is not constant),
      leaving others untouched

    Examples
    --------

    >>> print(Choice(["a", "b", "c", "e"]).value)
    "c"

    >>> print(Choice(["a", "b", "c", "e"], repetitions=3).value)
    ("b", "b", "c")
    '''
    _indices: tp.Optional[np.ndarray]
    def __init__(self, choices: tp.Union[int, tp.Iterable[tp.Any]], repetitions: tp.Optional[int] = None, deterministic: bool = False) -> None: ...
    def mutate(self) -> None: ...

class TransitionChoice(BaseChoice):
    '''Categorical parameter, choosing one of the provided choice options as a value, with continuous transitions.
    By default, this is ordered, and most algorithms except discrete OnePlusOne algorithms will consider it as ordered.
    The choices can be Parameters, in which case there value will be returned instead.
    The chosen parameter is drawn using transitions between current choice and the next/previous ones.

    Parameters
    ----------
    choices: list or int
        a list of possible values or Parameters for the variable (or an integer as a shortcut for range(num))
    transitions: np.ndarray or Array
        the transition weights. During transition, the direction (forward or backward will be drawn with
        equal probabilities), then the transitions weights are normalized through softmax, the 1st value gives
        the probability to remain in the same state, the second to move one step (backward or forward) and so on.
    ordered: bool
        if False, changes the default behavior to be unordered and sampled uniformly when setting the data to a
        normalized and centered Gaussian (used in DiscreteOnePlusOne only)

    Note
    ----
    - the "mutate" method only mutates the weights and the chosen Parameter (if it is not constant),
      leaving others untouched
    - in order to support export to standardized space, the index is encoded as a scalar. A normal distribution N(O,1)
      on this scalar yields a uniform choice of index. This may come to evolve for simplicity\'s sake.
    - currently, transitions are computed through softmax, this may evolve since this is somehow impractical
    '''
    _ref: tp.Optional['TransitionChoice']
    def __init__(self, choices: tp.Union[int, tp.Iterable[tp.Any]], transitions: tp.Union[tp.ArrayLike, Array] = (1.0, 1.0), repetitions: tp.Optional[int] = None, ordered: bool = True) -> None: ...
    def _internal_set_standardized_data(self, data: np.ndarray, reference: T) -> None: ...
    def _internal_get_standardized_data(self, reference: T) -> np.ndarray: ...
    @property
    def transitions(self) -> Array: ...
    def mutate(self) -> None: ...
