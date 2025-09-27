import numpy as np
from .. import base as base
from _typeshed import Incomplete

class PBOFunction(base.ExperimentFunction):
    '''
    Pseudo-boolean functions taken from the IOHexperimenter project, adapted for minimization

    Parameters
    ----------
    fid: int
        function number (for a list of functions, please see the
        `documentation <https://www.sciencedirect.com/science/article/abs/pii/S1568494619308099>>`_
    iid: int
        the instance of the function, specifies the transformations in variable and objective space
    dim: int
        The dimensionality of the problem. For function 21 and 23 (N-queens), this needs to be a perfect square
    instrumentation: str
        How the paramerterization is handled. Either "Softmax", in which case the optimizers will work on probabilities
        (so the same parametrization may actually lead to different values), or "Ordered". With this one the optimizer
        will handle a value, which will lead to a 0 if it is negative, and 1 if it is positive.
        This way it is no more stochastic, but the problem becomes very non-continuous

    Notes
    -----
    The IOHexperimenter is part of the IOHprofiler project (IOHprofiler.github.io)

    References
    -----
    For a full description of the available problems and used transformation methods,
    please see "Benchmarking discrete optimization heuristics with IOHprofiler" by C. Doerr et al.
    '''
    f_internal: Incomplete
    def __init__(self, fid: int = 1, iid: int = 0, dim: int = 16, instrumentation: str = 'Softmax') -> None: ...
    def _evaluation_internal(self, x: np.ndarray) -> float: ...

class WModelFunction(base.ExperimentFunction):
    '''
    Implementation of the W-Model taken from the IOHexperimenter project, adapted for minimization
    Currently supports only OneMax and LeadingOnes as base-functions

    Parameters
    ----------
    base_function: str
        Which base function to use. Either "OneMax" or "LeadingOnes"
    iid: int
        the instance of the function, specifies the transformations in variable and objective space
    dim: int
        The dimensionality of the problem.
    dummy:
        Float between 0 and 1, fractoin of valid bits.
    epistasis:
        size of sub-string for epistasis
    neutrality:
        size of sub-string for neutrality
    ruggedness:
        gamma for ruggedness layper
    instrumentation: str
        How the paramerterization is handled. Either "Softmax", in which case the optimizers will work on probabilities
        (so the same parametrization may actually lead to different values), or "Ordered". With this one the optimizer
        will handle a value, which will lead to a 0 if it is negative, and 1 if it is positive.
        This way it is no more stochastic, but the problem becomes very non-continuous

    Notes
    -----
    The IOHexperimenter is part of the IOHprofiler project (IOHprofiler.github.io)

    References
    -----
    For a description of the W-model in the IOHexperimenter and of the used transformation methods,
    please see "Benchmarking discrete optimization heuristics with IOHprofiler" by C. Doerr et al.
    '''
    f_internal: Incomplete
    def __init__(self, base_function: str = 'OneMax', iid: int = 0, dim: int = 16, dummy: float = 0, epistasis: int = 0, neutrality: int = 0, ruggedness: int = 0, instrumentation: str = 'Softmax') -> None: ...
    def _evaluation_internal(self, x: np.ndarray) -> float: ...
