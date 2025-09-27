import numpy as np
from ..base import ExperimentFunction as ExperimentFunction
from _typeshed import Incomplete
from nevergrad.parametrization import parameter as p

class OlympusSurface(ExperimentFunction):
    SURFACE_KINDS: Incomplete
    kind: Incomplete
    param_dim: Incomplete
    noise_kind: Incomplete
    noise_scale: Incomplete
    surface: Incomplete
    surface_without_noise: Incomplete
    shift: Incomplete
    def __init__(self, kind: str, dimension: int = 10, noise_kind: str = 'GaussianNoise', noise_scale: float = 1) -> None: ...
    def _simulate_surface(self, x: np.ndarray, noise: bool = True) -> float: ...
    def evaluation_function(self, *recommendations) -> float:
        """Averages multiple evaluations if necessary"""

class OlympusEmulator(ExperimentFunction):
    DATASETS: Incomplete
    dataset_kind: Incomplete
    model_kind: Incomplete
    def __init__(self, dataset_kind: str = 'alkox', model_kind: str = 'NeuralNet') -> None: ...
    def _get_parametrization(self) -> p.Parameter: ...
    def _simulate_emulator(self, x: np.ndarray) -> float: ...
