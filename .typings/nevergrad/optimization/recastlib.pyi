import nevergrad.common.typing as tp
from . import base as base, recaster as recaster
from .base import IntOrParameter as IntOrParameter
from _typeshed import Incomplete
from nevergrad.common import errors as errors
from nevergrad.parametrization import parameter as p

class _NonObjectMinimizeBase(recaster.SequentialRecastOptimizer):
    multirun: int
    _normalizer: tp.Any
    initial_guess: tp.Optional[tp.ArrayLike]
    method: Incomplete
    random_restart: Incomplete
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1, *, method: str = 'Nelder-Mead', random_restart: bool = False) -> None: ...
    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: tp.Loss) -> None:
        '''Called whenever calling "tell" on a candidate that was not "asked".
        Defaults to the standard tell pipeline.
        '''
    def get_optimization_function(self) -> tp.Callable[[tp.Callable[[tp.ArrayLike], float]], tp.ArrayLike]: ...
    @staticmethod
    def _optimization_function(weakself: tp.Any, objective_function: tp.Callable[[tp.ArrayLike], float]) -> tp.ArrayLike: ...

class NonObjectOptimizer(base.ConfiguredOptimizer):
    """Wrapper over Scipy optimizer implementations, in standard ask and tell format.
    This is actually an import from scipy-optimize, including Sequential Quadratic Programming,

    Parameters
    ----------
    method: str
        Name of the method to use among:

        - Nelder-Mead
        - COBYLA
        - SQP (or SLSQP): very powerful e.g. in continuous noisy optimization. It is based on
          approximating the objective function by quadratic models.
        - Powell
        - NLOPT* (https://nlopt.readthedocs.io/en/latest/; by default, uses Sbplx, based on Subplex);
            can be NLOPT,
                NLOPT_LN_SBPLX,
                NLOPT_LN_PRAXIS,
                NLOPT_GN_DIRECT,
                NLOPT_GN_DIRECT_L,
                NLOPT_GN_CRS2_LM,
                NLOPT_GN_AGS,
                NLOPT_GN_ISRES,
                NLOPT_GN_ESCH,
                NLOPT_LN_COBYLA,
                NLOPT_LN_BOBYQA,
                NLOPT_LN_NEWUOA_BOUND,
                NLOPT_LN_NELDERMEAD.
    random_restart: bool
        whether to restart at a random point if the optimizer converged but the budget is not entirely
        spent yet (otherwise, restarts from best point)

    Note
    ----
    These optimizers do not support asking several candidates in a row
    """
    recast: bool
    no_parallelization: bool
    def __init__(self, *, method: str = 'Nelder-Mead', random_restart: bool = False) -> None: ...

AX: Incomplete
BOBYQA: Incomplete
NelderMead: Incomplete
CmaFmin2: Incomplete
Powell: Incomplete
RPowell: Incomplete
BFGS: Incomplete
RBFGS: Incomplete
LBFGSB: Incomplete
Cobyla: Incomplete
RCobyla: Incomplete
SQP: Incomplete
SLSQP = SQP
RSQP: Incomplete
RSLSQP = RSQP
NLOPT_LN_SBPLX: Incomplete
NLOPT_LN_PRAXIS: Incomplete
NLOPT_GN_DIRECT: Incomplete
NLOPT_GN_DIRECT_L: Incomplete
NLOPT_GN_CRS2_LM: Incomplete
NLOPT_GN_AGS: Incomplete
NLOPT_GN_ISRES: Incomplete
NLOPT_GN_ESCH: Incomplete
NLOPT_LN_COBYLA: Incomplete
NLOPT_LN_BOBYQA: Incomplete
NLOPT_LN_NEWUOA_BOUND: Incomplete
NLOPT_LN_NELDERMEAD: Incomplete
SMAC3: Incomplete

class _PymooMinimizeBase(recaster.SequentialRecastOptimizer):
    algorithm: Incomplete
    _no_hypervolume: bool
    _initial_seed: int
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1, *, algorithm: str) -> None: ...
    def get_optimization_function(self) -> tp.Callable[[tp.Callable[..., tp.Any]], tp.Optional[tp.ArrayLike]]: ...
    @staticmethod
    def _optimization_function(weakself: tp.Any, objective_function: tp.Callable[[tp.ArrayLike], float]) -> tp.Optional[tp.ArrayLike]: ...
    def _internal_ask_candidate(self) -> p.Parameter:
        """
        Special version to make sure that num_objectives has been set before
        the proper _internal_ask_candidate, in our parent class, is called.
        """
    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        """
        Special version to make sure that we the extra initial evaluation which
        we may have done in order to get num_objectives, is discarded.
        Note that this discarding means that the extra point will not make it into
        replay_archive_tell. Correspondingly, because num_objectives will make it into
        the pickle, __setstate__ will never need a dummy ask.
        """
    def _post_loss(self, candidate: p.Parameter, loss: float) -> tp.Loss:
        """
        Multi-Objective override for this function.
        """

class Pymoo(base.ConfiguredOptimizer):
    '''Wrapper over Pymoo optimizer implementations, in standard ask and tell format.
    This is actually an import from Pymoo Optimize.

    Parameters
    ----------
    algorithm: str

        Use "algorithm-name" with following names to access algorithm classes:
        Single-Objective
        -"de"
        -\'ga\'
        -"brkga"
        -"nelder-mead"
        -"pattern-search"
        -"cmaes"
        Multi-Objective
        -"nsga2"
        Multi-Objective requiring reference directions, points or lines
        -"rnsga2"
        -"nsga3"
        -"unsga3"
        -"rnsga3"
        -"moead"
        -"ctaea"

    Note
    ----
    These optimizers do not support asking several candidates in a row
    '''
    recast: bool
    no_parallelization: bool
    def __init__(self, *, algorithm: str) -> None: ...

class _PymooBatchMinimizeBase(recaster.BatchRecastOptimizer):
    algorithm: Incomplete
    _no_hypervolume: bool
    _initial_seed: int
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1, *, algorithm: str) -> None: ...
    def get_optimization_function(self) -> tp.Callable[[tp.Callable[..., tp.Any]], tp.Optional[tp.ArrayLike]]: ...
    @staticmethod
    def _optimization_function(weakself: tp.Any, objective_function: tp.Callable[[tp.ArrayLike], float]) -> tp.Optional[tp.ArrayLike]: ...
    def _internal_ask_candidate(self) -> p.Parameter:
        '''Reads messages from the thread in which the underlying optimization function is running
        New messages are sent as "ask".
        '''
    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        '''Returns value for a point which was "asked"
        (none asked point cannot be "tell")
        '''
    def _post_loss(self, candidate: p.Parameter, loss: float) -> tp.Loss:
        """
        Multi-Objective override for this function.
        """

class PymooBatch(base.ConfiguredOptimizer):
    '''Wrapper over Pymoo optimizer implementations, in standard ask and tell format.
    This is actually an import from Pymoo Optimize.

    Parameters
    ----------
    algorithm: str

        Use "algorithm-name" with following names to access algorithm classes:
        Single-Objective
        -"de"
        -\'ga\'
        -"brkga"
        -"nelder-mead"
        -"pattern-search"
        -"cmaes"
        Multi-Objective
        -"nsga2"
        Multi-Objective requiring reference directions, points or lines
        -"rnsga2"
        -"nsga3"
        -"unsga3"
        -"rnsga3"
        -"moead"
        -"ctaea"

    Note
    ----
    These optimizers do not support asking several candidates in a row
    '''
    recast: bool
    def __init__(self, *, algorithm: str) -> None: ...

def _create_pymoo_problem(optimizer: base.Optimizer, objective_function: tp.Callable[[tp.ArrayLike], float], elementwise: bool = True): ...

PymooCMAES: Incomplete
PymooBIPOP: Incomplete
PymooNSGA2: Incomplete
PymooBatchNSGA2: Incomplete
pysot: Incomplete
DSbase: Incomplete
DS3p: Incomplete
DSsubspace: Incomplete
DSproba: Incomplete
