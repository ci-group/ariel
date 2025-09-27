import nevergrad.common.typing as tp
from . import base as base
from .base import IntOrParameter as IntOrParameter
from _typeshed import Incomplete
from nevergrad.parametrization import parameter as p, transforms as transforms

def _hp_parametrization_to_dict(x, **kwargs): ...
def _hp_dict_to_parametrization(x): ...
def _get_search_space(param_name, param): ...

class _HyperOpt(base.Optimizer):
    space: Incomplete
    _transform: Incomplete
    trials: Incomplete
    domain: Incomplete
    tpe_args: Incomplete
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1, *, prior_weight: float = 1.0, n_startup_jobs: int = 20, n_EI_candidates: int = 24, gamma: float = 0.25, verbose: bool = False) -> None: ...
    def _internal_ask_candidate(self) -> p.Parameter: ...
    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None: ...
    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: float) -> None: ...

class ParametrizedHyperOpt(base.ConfiguredOptimizer):
    '''Hyperopt: Distributed Asynchronous Hyper-parameter Optimization.
    This class is a wrapper over the `hyperopt <https://github.com/hyperopt/hyperopt>`_ package.

    Parameters
    ----------
    parametrization: int or Parameter
        Parametrization object
    budget: int
        Number of iterations
    num_workers: int
        Number of workers
    prior_weight: float (default 1.0)
        Smoothing factor to avoid having zero probabilities
    n_startup_jobs: int (default 20)
        Number of random uniform suggestions at initialization
    n_EI_candidates: int (default 24)
        Number of generated candidates during EI maximization
    gamma: float (default 0.25)
        Threshold to split between l(x) and g(x), see eq. 2 in

    verbose: bool (default False)
        Hyperopt algorithm verbosity

    Note
    ----
    HyperOpt is described in Bergstra, James S., et al.
    "Algorithms for hyper-parameter optimization."
    Advances in neural information processing systems. 2011
    '''
    no_parallelization: bool
    def __init__(self, *, prior_weight: float = 1.0, n_startup_jobs: int = 20, n_EI_candidates: int = 24, gamma: float = 0.25, verbose: bool = False) -> None: ...

HyperOpt: Incomplete
