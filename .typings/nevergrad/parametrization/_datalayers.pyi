import nevergrad.common.typing as tp
import numpy as np
from . import _layering as _layering, data as _data, discretization as discretization, transforms as trans, utils as utils
from ._layering import Int as Int
from .core import Parameter as Parameter
from .data import Data as Data
from _typeshed import Incomplete
from nevergrad.common import errors as errors

D = tp.TypeVar('D', bound=Data)
Op = tp.TypeVar('Op', bound='Operation')
BL = tp.TypeVar('BL', bound='BoundLayer')

class Operation(_layering.Layered, _layering.Filterable):
    _LAYER_LEVEL: Incomplete
    _LEGACY: bool
    def __init__(self, *args: tp.Any, **kwargs: tp.Any) -> None: ...

class BoundLayer(Operation):
    _LAYER_LEVEL: Incomplete
    bounds: tp.Tuple[tp.Optional[np.ndarray], tp.Optional[np.ndarray]]
    uniform_sampling: bool
    def __init__(self, lower: tp.BoundValue = None, upper: tp.BoundValue = None, uniform_sampling: tp.Optional[bool] = None) -> None:
        '''Bounds all real values into [lower, upper]

        Parameters
        ----------
        lower: float or None
            minimum value
        upper: float or None
            maximum value
        method: str
            One of the following choices:
        uniform_sampling: Optional bool
            Changes the default behavior of the "sample" method (aka creating a child and mutating it from the current instance)
            or the sampling optimizers, to creating a child with a value sampled uniformly (or log-uniformly) within
            the while range of the bounds. The "sample" method is used by some algorithms to create an initial population.
            This is activated by default if both bounds are provided.
        '''
    def _normalizer(self) -> trans.Transform: ...
    def __call__(self, data: D, inplace: bool = False) -> D:
        """Creates a new Data instance with bounds"""
    def _layered_sample(self) -> Data: ...
    def set_normalized_value(self, value: np.ndarray) -> None:
        """Sets a value normalized between 0 and 1"""
    def get_normalized_value(self) -> np.ndarray:
        """Gets a value normalized between 0 and 1"""
    def _check(self, value: np.ndarray) -> None: ...

class Modulo(BoundLayer):
    """Applies a modulo operation on the array
    CAUTION: WIP
    """
    _module: Incomplete
    def __init__(self, module: tp.Any) -> None: ...
    def _layered_get_value(self) -> np.ndarray: ...
    def _layered_set_value(self, value: np.ndarray) -> None: ...

class ForwardableOperation(Operation):
    """Operation with simple forward and backward methods
    (simplifies chaining)
    """
    def _layered_get_value(self) -> np.ndarray: ...
    def _layered_set_value(self, value: np.ndarray) -> None: ...
    def forward(self, value: tp.Any) -> tp.Any: ...
    def backward(self, value: tp.Any) -> tp.Any: ...

class Exponent(ForwardableOperation):
    """Applies an array as exponent of a float"""
    _base: Incomplete
    _name: Incomplete
    def __init__(self, base: float) -> None: ...
    def forward(self, value: tp.Any) -> tp.Any: ...
    def backward(self, value: tp.Any) -> tp.Any: ...

class Power(ForwardableOperation):
    """Applies a float as exponent of a Data parameter"""
    _power: Incomplete
    def __init__(self, power: float) -> None: ...
    def forward(self, value: tp.Any) -> tp.Any: ...
    def backward(self, value: tp.Any) -> tp.Any: ...

class Add(ForwardableOperation):
    """Applies an array as exponent of a floar"""
    _offset: Incomplete
    def __init__(self, offset: tp.Any) -> None: ...
    def forward(self, value: tp.Any) -> tp.Any: ...
    def backward(self, value: tp.Any) -> tp.Any: ...

class Multiply(ForwardableOperation):
    """Applies an array as exponent of a floar"""
    _mult: Incomplete
    name: Incomplete
    def __init__(self, value: tp.Any) -> None: ...
    def forward(self, value: tp.Any) -> tp.Any: ...
    def backward(self, value: tp.Any) -> tp.Any: ...

class Bound(BoundLayer):
    _method: Incomplete
    _transform: Incomplete
    def __init__(self, lower: tp.BoundValue = None, upper: tp.BoundValue = None, method: str = 'bouncing', uniform_sampling: tp.Optional[bool] = None) -> None:
        """Bounds all real values into [lower, upper] using a provided method

        See Parameter.set_bounds
        """
    def _layered_get_value(self) -> np.ndarray: ...
    def _layered_set_value(self, value: np.ndarray) -> None: ...

class SoftmaxSampling(Int):
    arity: Incomplete
    ordered: bool
    def __init__(self, arity: int, deterministic: bool = False) -> None: ...
    def _get_name(self) -> str: ...
    _cache: Incomplete
    def _layered_get_value(self) -> tp.Any: ...
    def _layered_set_value(self, value: tp.Any) -> tp.Any: ...

class AngleOp(Operation):
    """Converts to and from angles from -pi to pi"""
    def _layered_get_value(self) -> tp.Any: ...
    def _layered_set_value(self, value: tp.Any) -> None: ...

def Angles(init: tp.Optional[tp.ArrayLike] = None, shape: tp.Optional[tp.Sequence[int]] = None, deg: bool = False, bound_method: tp.Optional[str] = None) -> _data.Array:
    """Creates an Array parameter representing an angle from -pi to pi (deg=False)
    or -180 to 180 (deg=True).
    Internally, this keeps track of coordinates which are transformed to an angle.

    Parameters
    ----------
    init: array-like or None
        initial values if provided (either shape or init is required)
    shape: sequence of int or None
        shape of the angle array, if provided (either shape or init is required)
    deg: bool
        whether to return the result in degrees instead of radians
    bound_method: optional str
        adds a bound in the standardized domain, to make sure the values do not
        diverge too much (experimental, the impact is not clear)

    Returns
    -------
    Array
        An Array Parameter instance which represents an angle between -pi and pi

    Notes
    ------
    This API is experimental and will probably evolve in the near future.
    """
