import numpy as np
import typing as tp
from . import corefuncs as corefuncs
from .base import ExperimentFunction as ExperimentFunction
from _typeshed import Incomplete

class PBT(ExperimentFunction):
    """Population-Based Training, also known as Lamarckism or Meta-Optimization."""
    _funcs: Incomplete
    _optima: Incomplete
    _hyperparameter_dimension: Incomplete
    _dimensions: Incomplete
    _total_dimension: Incomplete
    _population_checkpoints: list[np.ndarray]
    _population_parameters: list[np.ndarray]
    _population_fitness: list[float]
    def __init__(self, names: tuple[str, ...] = ('sphere', 'cigar', 'ellipsoid'), dimensions: tuple[int, ...] = (7, 7, 7), num_workers: int = 10) -> None: ...
    def unflatten(self, x): ...
    def value(self, x): ...
    def evolve(self, x: np.ndarray, pp: np.ndarray): ...
    def _func(self, x: np.ndarray): ...
    @classmethod
    def itercases(cls) -> tp.Iterator['PBT']: ...
