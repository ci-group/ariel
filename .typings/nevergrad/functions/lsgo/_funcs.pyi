import numpy as np
from . import _core as _core
from _typeshed import Incomplete
from nevergrad.common.decorators import Registry as Registry
from nevergrad.functions import ExperimentFunction as ExperimentFunction
from typing import Callable

registry: Incomplete

def read_data(name: str) -> dict[str, np.ndarray]: ...

class FunctionChunk:
    """Computes a sequence of transforms and applies a loss on it.
    This can be left multiplied by a scalar to add a weight.
    """
    loss: Incomplete
    transforms: Incomplete
    _scalar: float
    def __init__(self, transforms: list[Callable[[np.ndarray], np.ndarray]], loss: Callable[[np.ndarray], float]) -> None: ...
    def _apply_transforms(self, x: np.ndarray) -> np.ndarray: ...
    def __call__(self, x: np.ndarray) -> float: ...
    def __rmul__(self, scalar: float) -> FunctionChunk: ...
    def __repr__(self) -> str: ...

class LsgoFunction:
    '''Base function used in the LSGO testbed.
    It is in charge of computing the translation, and summing all function chunks.
    The "optimum" attribute holds the zero of the function. It is usually xopt, but is xopt + 1 in one case (F12),
    and unknown in one other (F14).
    '''
    bounds: Incomplete
    tags: Incomplete
    conditionning: float
    xopt: Incomplete
    optimum: np.ndarray | None
    functions: Incomplete
    def __init__(self, xopt: np.ndarray, functions: list[FunctionChunk]) -> None: ...
    def __call__(self, x: np.ndarray) -> float: ...
    @property
    def dimension(self) -> int:
        """Dimension of the function space"""
    def instrumented(self, transform: str = 'bouncing') -> ExperimentFunction:
        '''Returns an instrumented function, taking the bounds into account by composing it with
        a bounding transform. Instrumentated functions are necessary
        for nevergrad benchmarking

        Parameter
        ---------
        transform: str
            "bouncing", "arctan", "tanh" or "clipping"
        '''

class ShiftedElliptic(LsgoFunction):
    number: int
    bounds: Incomplete
    conditionning: float
    tags: Incomplete
    def __init__(self, xopt: np.ndarray) -> None: ...

class ShiftedRastrigin(LsgoFunction):
    number: int
    bounds: Incomplete
    conditionning: float
    tags: Incomplete
    def __init__(self, xopt: np.ndarray) -> None: ...

class ShiftedAckley(LsgoFunction):
    number: int
    bounds: Incomplete
    conditionning: float
    tags: Incomplete
    def __init__(self, xopt: np.ndarray) -> None: ...

class _MultiPartFunction(LsgoFunction):
    """Base class for most multi-part function, overlapping or not."""
    number: int
    bounds: Incomplete
    tags: list[str]
    conditionning: float
    overlap: int
    def _make_loss(self, dimension: int, side_loss: bool) -> Callable[[np.ndarray], float]: ...
    def _make_transforms(self, side_loss: bool) -> list[Callable[[np.ndarray], np.ndarray]]: ...
    def __init__(self, xopt: np.ndarray, p: np.ndarray, s: np.ndarray, w: np.ndarray, R25: np.ndarray, R50: np.ndarray, R100: np.ndarray) -> None: ...

class PartiallySeparableElliptic(_MultiPartFunction):
    number: int
    bounds: Incomplete
    tags: Incomplete
    conditionning: float
    def _make_loss(self, dimension: int, side_loss: bool) -> Callable[[np.ndarray], float]: ...
    def _make_transforms(self, side_loss: bool) -> list[Callable[[np.ndarray], np.ndarray]]: ...

class PartiallySeparableRastrigin(_MultiPartFunction):
    number: int
    bounds: Incomplete
    tags: Incomplete
    conditionning: float
    def _make_loss(self, dimension: int, side_loss: bool) -> Callable[[np.ndarray], float]: ...
    def _make_transforms(self, side_loss: bool) -> list[Callable[[np.ndarray], np.ndarray]]: ...

class PartiallySeparableAckley(_MultiPartFunction):
    number: int
    bounds: Incomplete
    tags: Incomplete
    conditionning: float
    def _make_loss(self, dimension: int, side_loss: bool) -> Callable[[np.ndarray], float]: ...
    def _make_transforms(self, side_loss: bool) -> list[Callable[[np.ndarray], np.ndarray]]: ...

class PartiallySeparableSchwefel(_MultiPartFunction):
    number: int
    bounds: Incomplete
    tags: Incomplete
    conditionning: float
    def _make_loss(self, dimension: int, side_loss: bool) -> Callable[[np.ndarray], float]: ...
    def _make_transforms(self, side_loss: bool) -> list[Callable[[np.ndarray], np.ndarray]]: ...

class PartiallySeparableElliptic2(PartiallySeparableElliptic):
    number: int
    bounds: Incomplete
    tags: Incomplete
    conditionning: float

class PartiallySeparableRastrigin2(PartiallySeparableRastrigin):
    number: int
    bounds: Incomplete
    tags: Incomplete
    conditionning: float

class PartiallySeparableAckley2(PartiallySeparableAckley):
    number: int
    bounds: Incomplete
    tags: Incomplete
    conditionning: float

class PartiallySeparableSchwefel2(PartiallySeparableSchwefel):
    number: int
    bounds: Incomplete
    tags: Incomplete
    conditionning: float

class ShiftedRosenbrock(LsgoFunction):
    number: int
    bounds: Incomplete
    conditionning: float
    tags: Incomplete
    optimum: Incomplete
    def __init__(self, xopt: np.ndarray) -> None: ...

class OverlappingSchwefel(PartiallySeparableSchwefel):
    number: int
    bounds: Incomplete
    tags: Incomplete
    overlap: int

class ConflictingSchwefel(LsgoFunction):
    number: int
    bounds: Incomplete
    tags: Incomplete
    conditionning: float
    optimum: Incomplete
    def __init__(self, xopt: np.ndarray, p: np.ndarray, s: np.ndarray, w: np.ndarray, R25: np.ndarray, R50: np.ndarray, R100: np.ndarray) -> None: ...

class ShiftedSchwefel(LsgoFunction):
    number: int
    bounds: Incomplete
    tags: Incomplete
    conditionning: float
    def __init__(self, xopt: np.ndarray) -> None: ...

def make_function(number: int) -> LsgoFunction:
    """Creates one of the LSGO functions.

    Parameters
    ----------
    number: int
        the number of the function, from 1 to 15 (included)

    Returns
    -------
    LsgoFunction
        A function which acts exactly as the CPP implementation of LSGO (which may deviate from matlab or the
        actual paper). It has an attribute dimension for the optimization space dimension, and bounds for the upper
        and lower bounds of this space.
    """
