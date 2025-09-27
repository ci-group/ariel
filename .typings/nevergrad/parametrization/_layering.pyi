import nevergrad.common.typing as tp
import numpy as np
from _typeshed import Incomplete
from enum import Enum
from nevergrad.common import errors as errors

L = tp.TypeVar('L', bound='Layered')
F = tp.TypeVar('F', bound='Filterable')
In = tp.TypeVar('In')
Out = tp.TypeVar('Out')

class Level(Enum):
    """Lower level is deeper in the structure"""
    ROOT = 0
    OPERATION = 10
    INTEGER_CASTING = 800
    ARRAY_CASTING = 900
    SCALAR_CASTING = 950
    CONSTRAINT = 1000

class Layered:
    """Hidden API for overriding/modifying the behavior of a Parameter,
    which is itself a Layered object.

    Layers can be added and will be ordered depending on their level
    """
    _LAYER_LEVEL: Incomplete
    _layers: Incomplete
    _layer_index: int
    _name: tp.Optional[str]
    uid: Incomplete
    def __init__(self) -> None: ...
    def add_layer(self, other: Layered) -> L:
        """Adds a layer which will modify the object behavior"""
    def _on_layer_added(self) -> None:
        """Hook called when the layer is added to another one"""
    def _call_deeper(self, name: str, *args: tp.Any, **kwargs: tp.Any) -> tp.Any: ...
    def _layered_get_value(self) -> tp.Any: ...
    def _layered_set_value(self, value: tp.Any) -> tp.Any: ...
    def _layered_del_value(self) -> None: ...
    def _layered_sample(self) -> Layered: ...
    def _layered_mutate(self) -> None: ...
    def _layered_recombine(self, *args: Layered) -> None: ...
    @property
    def random_state(self) -> np.random.RandomState: ...
    def copy(self) -> L:
        """Creates a new unattached layer with the same behavior"""
    def _get_name(self) -> str:
        """Internal implementation of parameter name. This should be value independant, and should not account
        for internal/model parameters.
        """
    def __repr__(self) -> str: ...
    @property
    def name(self) -> str:
        """Name of the parameter
        This is used to keep track of how this Parameter is configured (included through internal/model parameters),
        mostly for reproducibility A default version is always provided, but can be overriden directly
        through the attribute, or through the set_name method (which allows chaining).
        """
    @name.setter
    def name(self, name: str) -> None: ...
    def set_name(self, name: str) -> L:
        """Sets a name and return the current instrumentation (for chaining)

        Parameters
        ----------
        name: str
            new name to use to represent the Parameter
        """

class ValueProperty(tp.Generic[In, Out]):
    """Typed property (descriptor) object so that the value attribute of
    Parameter objects fetches _layered_get_value and _layered_set_value methods
    """
    __doc__: str
    def __init__(self) -> None: ...
    def __get__(self, obj: Layered, objtype: tp.Optional[tp.Type[object]] = None) -> Out: ...
    def __set__(self, obj: Layered, value: In) -> None: ...
    def __delete__(self, obj: Layered) -> None: ...

class _ScalarCasting(Layered):
    """Cast Array as a scalar"""
    _LAYER_LEVEL: Incomplete
    def _layered_get_value(self) -> float: ...
    def _layered_set_value(self, value: tp.Any) -> None: ...

class ArrayCasting(Layered):
    """Cast inputs of type tuple/list etc to array"""
    _LAYER_LEVEL: Incomplete
    def _layered_set_value(self, value: tp.ArrayLike) -> None: ...

class Filterable:
    @classmethod
    def filter_from(cls, parameter: Layered) -> tp.List[F]: ...

class Int(Layered, Filterable):
    """Cast Data as integer (or integer array)

    Parameters
    ----------
    deterministic: bool
        if True, the data is rounded to the closest integer, if False, both surrounded
        integers can be sampled inversely proportionally to how close the actual value
        is from the integers.

    Example
    -------
    0.2 is cast to 0 in deterministic mode, and either 0 (80% chance) or 1 (20% chance) in
    non-deterministic mode

    Usage
    -----
    :code:`param = ng.ops.Int(deterministic=True)(ng.p.Array(shape=(3,)))`
    """
    _LAYER_LEVEL: Incomplete
    arity: tp.Optional[int]
    ordered: bool
    deterministic: Incomplete
    _cache: tp.Optional[np.ndarray]
    def __init__(self, deterministic: bool = True) -> None: ...
    def _get_name(self) -> str: ...
    def _layered_get_value(self) -> np.ndarray: ...
    def _layered_del_value(self) -> None: ...
    def __call__(self, layered: L) -> L:
        """Creates a new Data instance with int-casting"""

def _to_int(value: tp.Union[float, np.ndarray]) -> tp.Union[int, np.ndarray]:
    """Returns a int from a float, or a int typed array from a float array"""
