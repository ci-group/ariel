import nevergrad.common.typing as tp
import numpy as np
from _typeshed import Incomplete

def threshold_discretization(x: tp.ArrayLike, arity: int = 2) -> tp.List[int]:
    """Discretize by casting values from 0 to arity -1, assuming that x values
    follow a normal distribution.

    Parameters
    ----------
    x: list/array
       values to discretize
    arity: int
       the number of possible integer values (arity n will lead to values from 0 to n - 1)

    Note
    ----
    - nans are processed as negative infs (yields 0)
    """
def inverse_threshold_discretization(indexes: tp.List[int], arity: int = 2) -> np.ndarray: ...
def noisy_inverse_threshold_discretization(indexes: tp.List[int], arity: int = 2, gen: tp.Any = None) -> np.ndarray: ...
def weight_for_reset(arity: int) -> float:
    """p is an arbitrary probability that the provided arg will be sampled with the returned point"""

class Encoder:
    """Handles softmax weights which need to be turned into probabilities and sampled
    This class is expected to evolve to be more usable and include more features (like
    conversion from probabilities to weights?)
    It will replace most of the code above if possible

    Parameters
    ----------
    weights: array
        the weights of size samples x options, that will be turned to probabilities
        using softmax.
    rng: RandomState
        random number generator for sampling following the probabilities

    Notes
    -----
    - if one or several inf values are present in a row, only those are considered
    - in case of tie, the deterministic value is the first one (lowest) of the tie
    - nans and -infs are ignored, except if all are (then uniform random choice)
    """
    weights: Incomplete
    _rng: Incomplete
    def __init__(self, weights: np.ndarray, rng: np.random.RandomState) -> None: ...
    def probabilities(self) -> np.ndarray:
        """Creates the probability matrix from the weights"""
    def encode(self, deterministic: bool = False) -> np.ndarray:
        """Sample an index from each row depending on the provided probabilities.

        Parameters
        ----------
        deterministic: bool
            set to True for sampling deterministically the more likely option
            (largest probability)
        """
