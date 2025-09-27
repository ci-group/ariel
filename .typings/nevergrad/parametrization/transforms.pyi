import nevergrad.common.typing as tp
import numpy as np
from . import utils as utils
from _typeshed import Incomplete

def bound_to_array(x: tp.BoundValue) -> np.ndarray:
    """Updates type of bounds to use arrays"""

class Transform:
    """Base class for transforms implementing a forward and a backward (inverse)
    method.
    This provide a default representation, and a short representation should be implemented
    for each transform.
    """
    name: Incomplete
    def __init__(self) -> None: ...
    def forward(self, x: np.ndarray) -> np.ndarray: ...
    def backward(self, y: np.ndarray) -> np.ndarray: ...
    def reverted(self) -> Transform: ...
    def __repr__(self) -> str: ...

class Reverted(Transform):
    """Inverse of a transform.

    Parameters
    ----------
    transform: Transform
    """
    transform: Incomplete
    name: Incomplete
    def __init__(self, transform: Transform) -> None: ...
    def forward(self, x: np.ndarray) -> np.ndarray: ...
    def backward(self, y: np.ndarray) -> np.ndarray: ...

class Affine(Transform):
    """Affine transform a * x + b

    Parameters
    ----------
    a: float
    b: float
    """
    a: Incomplete
    b: Incomplete
    name: Incomplete
    def __init__(self, a: tp.BoundValue, b: tp.BoundValue) -> None: ...
    def forward(self, x: np.ndarray) -> np.ndarray: ...
    def backward(self, y: np.ndarray) -> np.ndarray: ...

class Exponentiate(Transform):
    """Exponentiation transform base ** (coeff * x)
    This can for instance be used for to get a logarithmicly distruted values 10**(-[1, 2, 3]).

    Parameters
    ----------
    base: float
    coeff: float
    """
    base: Incomplete
    coeff: Incomplete
    name: Incomplete
    def __init__(self, base: float = 10.0, coeff: float = 1.0) -> None: ...
    def forward(self, x: np.ndarray) -> np.ndarray: ...
    def backward(self, y: np.ndarray) -> np.ndarray: ...

BoundType: Incomplete

def _f(x: BoundType) -> BoundType:
    """Format for prints:
    array with one scalars are converted to floats
    """

class BoundTransform(Transform):
    a_min: tp.Optional[np.ndarray]
    a_max: tp.Optional[np.ndarray]
    shape: tp.Tuple[int, ...]
    def __init__(self, a_min: BoundType = None, a_max: BoundType = None) -> None: ...
    def _check_shape(self, x: np.ndarray) -> None: ...

class TanhBound(BoundTransform):
    """Bounds all real values into [a_min, a_max] using a tanh transform.
    Beware, tanh goes very fast to its limits.

    Parameters
    ----------
    a_min: float
    a_max: float
    """
    _b: Incomplete
    _a: Incomplete
    name: Incomplete
    def __init__(self, a_min: tp.Union[tp.ArrayLike, float], a_max: tp.Union[tp.ArrayLike, float]) -> None: ...
    def forward(self, x: np.ndarray) -> np.ndarray: ...
    def backward(self, y: np.ndarray) -> np.ndarray: ...

class Clipping(BoundTransform):
    """Bounds all real values into [a_min, a_max] using clipping (not bijective).

    Parameters
    ----------
    a_min: float or None
        lower bound
    a_max: float or None
        upper bound
    bounce: bool
        bounce (once) on borders instead of just clipping
    """
    _bounce: Incomplete
    name: Incomplete
    checker: Incomplete
    def __init__(self, a_min: BoundType = None, a_max: BoundType = None, bounce: bool = False) -> None: ...
    def forward(self, x: np.ndarray) -> np.ndarray: ...
    def backward(self, y: np.ndarray) -> np.ndarray: ...

class ArctanBound(BoundTransform):
    """Bounds all real values into [a_min, a_max] using an arctan transform.
    This is a much softer approach compared to tanh.

    Parameters
    ----------
    a_min: float
    a_max: float
    """
    _b: Incomplete
    _a: Incomplete
    name: Incomplete
    def __init__(self, a_min: tp.Union[tp.ArrayLike, float], a_max: tp.Union[tp.ArrayLike, float]) -> None: ...
    def forward(self, x: np.ndarray) -> np.ndarray: ...
    def backward(self, y: np.ndarray) -> np.ndarray: ...

class CumulativeDensity(BoundTransform):
    """Bounds all real values into [0, 1] using a gaussian cumulative density function (cdf)
    Beware, cdf goes very fast to its limits.

    Parameters
    ----------
    lower: float
        lower bound
    upper: float
        upper bound
    eps: float
        small values to avoid hitting the bounds
    scale: float
        scaling factor of the density
    density: str
        either gaussian, or cauchy distributions
    """
    _b: Incomplete
    _a: Incomplete
    _eps: Incomplete
    _scale: Incomplete
    name: Incomplete
    _forw: Incomplete
    _back: Incomplete
    def __init__(self, lower: float = 0.0, upper: float = 1.0, eps: float = 1e-09, scale: float = 1.0, density: str = 'gaussian') -> None: ...
    def forward(self, x: np.ndarray) -> np.ndarray: ...
    def backward(self, y: np.ndarray) -> np.ndarray: ...

class Fourrier(Transform):
    axes: tp.Tuple[int, ...]
    name: Incomplete
    def __init__(self, axes: tp.Union[int, tp.Sequence[int]] = 0) -> None: ...
    def forward(self, x: np.ndarray) -> np.ndarray: ...
    def backward(self, y: np.ndarray) -> np.ndarray: ...
