import numpy as np
from ..base import ExperimentFunction as ExperimentFunction
from _typeshed import Incomplete

class TO(ExperimentFunction):
    n: Incomplete
    idx: Incomplete
    def __init__(self, n: int = 50) -> None: ...
    def _simulate_to(self, x: np.ndarray) -> float: ...
