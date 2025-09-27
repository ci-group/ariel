import numpy as np
from .. import base as base

class OptimizeFish(base.ExperimentFunction):
    """
    Fishing simulator.

    Parameters
    ----------
    time: int
        number of days of the planning

    """
    def __init__(self, time: int = 365) -> None: ...

def _compute_total_fishing(list_number_fishermen: np.ndarray) -> float:
    """Lotka-Volterra equations.

    This computes the total fishing, given the fishing effort every day.
    The problem makes sense for abritrary number of days, so that this works for
    any length of the input. 365 means one year."""
