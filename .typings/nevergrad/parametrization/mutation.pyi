import numpy as np
import typing as tp
from . import _layering as _layering, core as core, transforms as transforms
from .choice import Choice as Choice
from .data import Data as Data, Scalar as Scalar
from _typeshed import Incomplete

D = tp.TypeVar('D', bound=Data)
P = tp.TypeVar('P', bound=core.Parameter)

class Mutation(_layering.Layered):
    """Custom mutation or recombination operation
    This is an experimental API

    Call on a Parameter to create a new Parameter with the
    provided mutation/recombination.
    """
    _TYPE = core.Parameter
    def root(self) -> core.Parameter: ...
    def _check_type(self, param: core.Layered) -> None: ...
    def __call__(self, parameter: P, inplace: bool = False) -> P: ...

class DataMutation(Mutation):
    _TYPE = Data
    _parameters: Incomplete
    def __init__(self, **parameters: tp.Any) -> None: ...
    def root(self) -> Data: ...
    def _on_layer_added(self) -> None: ...

class MutationChoice(DataMutation):
    """Selects one of the provided mutation based on a Choice subparameter
    Caution: there may be subparameter collisions
    """
    mutations: Incomplete
    with_default: Incomplete
    def __init__(self, mutations: tp.Sequence[Mutation], with_default: bool = True) -> None: ...
    def _on_layer_added(self) -> None: ...
    def _select(self) -> core.Layered: ...
    def _layered_recombine(self, *others: core.Layered) -> None: ...
    def _layered_mutate(self) -> None: ...

class Cauchy(Mutation):
    def _layered_mutate(self) -> None: ...

class Crossover(DataMutation):
    """Operator for merging part of an array into another one

    Parameters
    ----------
    axis: None or int or tuple of ints
        the axis (or axes) on which the merge will happen. This axis will be split into 3 parts: the first and last one will take
        value from the first array, the middle one from the second array.
    max_size: None or int
        maximum size of the part taken from the second array. By default, this is at most around half the number of total elements of the
        array to the power of 1/number of axis.


    Notes
    -----
    - this is experimental, the API may evolve
    - when using several axis, the size of the second array part is the same on each axis (aka a square in 2D, a cube in 3D, ...)

    Examples:
    ---------
    - 2-dimensional array, with crossover on dimension 1:
      0 1 0 0
      0 1 0 0
      0 1 0 0
    - 2-dimensional array, with crossover on dimensions 0 and 1:
      0 0 0 0
      0 1 1 0
      0 1 1 0
    """
    def __init__(self, axis: tp.Any = None, max_size: int | Scalar | None = None, fft: bool = False) -> None: ...
    @property
    def axis(self) -> tuple[int, ...] | None: ...
    def _layered_recombine(self, *arrays: Data) -> None: ...
    def _apply_array(self, arrays: tp.Sequence[np.ndarray]) -> np.ndarray: ...

class RavelCrossover(Crossover):
    """Operator for merging part of an array into another one, after raveling

    Parameters
    ----------
    max_size: None or int
        maximum size of the part taken from the second array. By default, this is at most around half the number of total elements of the
        array to the power of 1/number of axis.
    """
    def __init__(self, max_size: int | Scalar | None = None) -> None: ...
    def _apply_array(self, arrays: tp.Sequence[np.ndarray]) -> np.ndarray: ...

def _make_slices(shape: tuple[int, ...], axes: tuple[int, ...], size: int, rng: np.random.RandomState) -> list[slice]: ...

class Translation(DataMutation):
    def __init__(self, axis: int | tp.Iterable[int] | None = None) -> None: ...
    @property
    def axes(self) -> tuple[int, ...] | None: ...
    def _layered_mutate(self) -> None: ...
    def _apply_array(self, arrays: tp.Sequence[np.ndarray]) -> np.ndarray: ...

class AxisSlicedArray:
    array: Incomplete
    axis: Incomplete
    def __init__(self, array: np.ndarray, axis: int) -> None: ...
    def __getitem__(self, slice_: slice) -> np.ndarray: ...

class Jumping(DataMutation):
    """Move a chunk for a position to another in an array"""
    def __init__(self, axis: int, size: int) -> None: ...
    @property
    def axis(self) -> int: ...
    @property
    def size(self) -> int: ...
    def _layered_mutate(self) -> None: ...
    def _apply_array(self, arrays: tp.Sequence[np.ndarray]) -> np.ndarray: ...

class LocalGaussian(DataMutation):
    def __init__(self, size: int | core.Parameter, axes: int | tp.Iterable[int] | None = None) -> None: ...
    @property
    def axes(self) -> tuple[int, ...] | None: ...
    def _layered_mutate(self) -> None: ...

def rolling_mean(vector: np.ndarray, window: int) -> np.ndarray: ...
