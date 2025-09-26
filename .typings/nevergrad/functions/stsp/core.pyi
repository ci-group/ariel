import numpy as np
from ..base import ExperimentFunction as ExperimentFunction
from _typeshed import Incomplete

class STSP(ExperimentFunction):
    order: Incomplete
    complex: Incomplete
    x: Incomplete
    y: Incomplete
    def __init__(self, dimension: int = 500, complex_tsp: bool = False) -> None: ...
    def _simulate_stsp(self, x: np.ndarray) -> float: ...
    def make_plots(self, filename: str = 'stsp.png') -> None: ...
