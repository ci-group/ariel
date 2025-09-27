from .. import base as base
from _typeshed import Incomplete

class OptimizeMix(base.ExperimentFunction):
    """
    MixSimulator is an application with an optimization model for calculating
    and simulating the least cost of an energy mix under certain constraints.

    For now, it uses a default dataset (more will be added soon).

    For more information, visit : https://github.com/Foloso/MixSimulator

    Parameters
    ----------
    time: int
        total time over which it evaluates the mix (must be in hour)

    """
    _mix: Incomplete
    _demand: Incomplete
    def __init__(self, time: int = 8760) -> None: ...
