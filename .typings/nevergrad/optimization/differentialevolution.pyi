import nevergrad.common.typing as tp
import numpy as np
from . import base as base, metamodel as metamodel, oneshot as oneshot
from _typeshed import Incomplete
from nevergrad.parametrization import parameter as p

class Crossover:
    CR: float
    crossover: Incomplete
    random_state: Incomplete
    shape: Incomplete
    def __init__(self, random_state: np.random.RandomState, crossover: tp.Union[str, float], parameter: tp.Optional[p.Parameter] = None) -> None: ...
    def apply(self, donor: np.ndarray, individual: np.ndarray) -> None: ...
    def variablewise(self, donor: np.ndarray, individual: np.ndarray) -> None: ...
    def onepoint(self, donor: np.ndarray, individual: np.ndarray) -> None: ...
    def twopoints(self, donor: np.ndarray, individual: np.ndarray) -> None: ...
    def rotated_twopoints(self, donor: np.ndarray, individual: np.ndarray) -> None: ...
    def voronoi(self, donor: np.ndarray, individual: np.ndarray) -> None: ...

class _DE(base.Optimizer):
    """Differential evolution.

    Default pop size equal to 30
    We return the mean of the individuals with fitness better than median, which might be stupid sometimes.
    CR =.5, F1=.8, F2=.8, curr-to-best.
    Initial population: pure random.
    """
    objective_weights: Incomplete
    _config: Incomplete
    scale: Incomplete
    llambda: Incomplete
    _MULTIOBJECTIVE_AUTO_BOUND: Incomplete
    _penalize_cheap_violations: bool
    _uid_queue: Incomplete
    population: tp.Dict[str, p.Parameter]
    sampler: tp.Optional[base.Optimizer]
    _no_hypervolume: Incomplete
    def __init__(self, parametrization: base.IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1, config: tp.Optional['DifferentialEvolution'] = None, weights: tp.Any = None) -> None: ...
    def set_objective_weights(self, weights: tp.Any) -> None: ...
    def recommend(self) -> p.Parameter: ...
    def _internal_ask_candidate(self) -> p.Parameter: ...
    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None: ...
    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None: ...

class DifferentialEvolution(base.ConfiguredOptimizer):
    '''Differential evolution is typically used for continuous optimization.
    It uses differences between points in the population for doing mutations in fruitful directions;
    it is therefore a kind of covariance adaptation without any explicit covariance,
    making it super fast in high dimension. This class implements several variants of differential
    evolution, some of them adapted to genetic mutations as in
    `Holland’s work <https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#Two-point_and_k-point_crossover>`_),
    (this combination is termed :code:`TwoPointsDE` in Nevergrad, corresponding to :code:`crossover="twopoints"`),
    or to the noisy setting (coined :code:`NoisyDE`, corresponding to :code:`recommendation="noisy"`).
    In that last case, the optimizer returns the mean of the individuals with fitness better than median,
    which might be stupid sometimes though.

    Default settings are CR =.5, F1=.8, F2=.8, curr-to-best, pop size is 30
    Initial population: pure random.

    Parameters
    ----------
    initialization: "parametrization", "LHS" or "QR" or "QO" or "SO"
        algorithm/distribution used for the initialization phase. If "parametrization", this uses the
        sample method of the parametrization.
    scale: float or str
        scale of random component of the updates
    recommendation: "pessimistic", "optimistic", "mean" or "noisy"
        choice of the criterion for the best point to recommend
    crossover: float or str
        crossover rate value, or strategy among:
        - "dimension": crossover rate of  1 / dimension,
        - "random": different random (uniform) crossover rate at each iteration
        - "onepoint": one point crossover
        - "twopoints": two points crossover
        - "rotated_twopoints": more genetic 2p cross-over
        - "parametrization": use the parametrization recombine method
    F1: float
        differential weight #1
    F2: float
        differential weight #2
    popsize: int, "standard", "dimension", "large"
        size of the population to use. "standard" is max(num_workers, 30), "dimension" max(num_workers, 30, dimension +1)
        and "large" max(num_workers, 30, 7 * dimension).
    multiobjective_adaptation: bool
        Automatically adapts to handle multiobjective case.  This is a very basic **experimental** version,
        activated by default because the non-multiobjective implementation is performing very badly.
    high_speed: bool
        Trying to make the optimization faster by a metamodel for the recommendation step.
    '''
    initialization: Incomplete
    scale: Incomplete
    high_speed: Incomplete
    recommendation: Incomplete
    propagate_heritage: Incomplete
    F1: Incomplete
    F2: Incomplete
    crossover: Incomplete
    popsize: Incomplete
    multiobjective_adaptation: Incomplete
    def __init__(self, *, initialization: str = 'parametrization', scale: tp.Union[str, float] = 1.0, recommendation: str = 'optimistic', crossover: tp.Union[str, float] = 0.5, F1: float = 0.8, F2: float = 0.8, popsize: tp.Union[str, int] = 'standard', propagate_heritage: bool = False, multiobjective_adaptation: bool = True, high_speed: bool = False) -> None: ...

DE: Incomplete
LPSDE: Incomplete
TwoPointsDE: Incomplete
VoronoiDE: Incomplete
RotatedTwoPointsDE: Incomplete
LhsDE: Incomplete
QrDE: Incomplete
QODE: Incomplete
SPQODE: Incomplete
QOTPDE: Incomplete
LQOTPDE: Incomplete
LQODE: Incomplete
SODE: Incomplete
NoisyDE: Incomplete
AlmostRotationInvariantDE: Incomplete
RotationInvariantDE: Incomplete
DiscreteDE: Incomplete
