from .differentialevolution import *
from .es import *
from .oneshot import *
from .recastlib import *
import nevergrad.common.typing as tp
import numpy as np
from . import base as base, experimentalvariants as experimentalvariants, mutations as mutations, oneshot as oneshot
from .base import IntOrParameter as IntOrParameter, addCompare as addCompare
from .externalbo import HyperOpt as HyperOpt
from .oneshot import RandomSearchMaker as RandomSearchMaker
from _typeshed import Incomplete
from bayes_opt import BayesianOptimization
from nevergrad.common import errors as errors
from nevergrad.parametrization import _datalayers as _datalayers, _layering as _layering, discretization as discretization, parameter as p, transforms as transforms

logger: Incomplete

def smooth_copy(array: p.Array, possible_radii: tp.Optional[tp.List[int]] = None) -> p.Array: ...

class _OnePlusOne(base.Optimizer):
    """Simple but sometimes powerful optimization algorithm.

    We use the one-fifth adaptation rule, going back to Schumer and Steiglitz (1968).
    It was independently rediscovered by Devroye (1972) and Rechenberg (1973).
    We use asynchronous updates, so that the 1+1 can actually be parallel and even
    performs quite well in such a context - this is naturally close to 1+lambda.

    Posssible mutations include gaussian and cauchy for the continuous case, and in the discrete case:
    discrete, fastga, rls, doublefastga, adaptive, portfolio, discreteBSO, doerr.
    - discrete is the most classical discrete mutation operator,
    - rls is the Randomized Local Search,
    - doubleFastGA is an adaptation of FastGA to arity > 2, Portfolio corresponds to random mutation rates,
    - discreteBSO corresponds to a decreasing schedule of mutation rate.
    - adaptive and doerr correspond to various self-adaptive mutation rates.
    - coordinatewise_adaptive is the anisotropic counterpart of the adaptive version.
    """
    antismooth: Incomplete
    crossover_type: Incomplete
    roulette_size: Incomplete
    _sigma: float
    _previous_best_loss: Incomplete
    _best_recent_mr: float
    inds: Incomplete
    imr: float
    use_pareto: Incomplete
    smoother: Incomplete
    super_radii: Incomplete
    annealing: Incomplete
    _annealing_base: tp.Optional[tp.ArrayLike]
    _max_loss: Incomplete
    sparse: Incomplete
    arity_for_discrete_mutation: Incomplete
    _adaptive_mr: float
    _global_mr: float
    _memory_index: int
    _memory_size: int
    _best_recent_loss: Incomplete
    _velocity: Incomplete
    _modified_variables: Incomplete
    noise_handling: Incomplete
    mutation: Incomplete
    crossover: Incomplete
    rotation: Incomplete
    _doerr_mutation_rates: Incomplete
    _doerr_mutation_rewards: Incomplete
    _doerr_counters: Incomplete
    _doerr_epsilon: float
    _doerr_gamma: Incomplete
    _doerr_current_best: Incomplete
    _doerr_index: int
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1, *, noise_handling: tp.Optional[tp.Union[str, tp.Tuple[str, float]]] = None, tabu_length: int = 0, mutation: str = 'gaussian', crossover: bool = False, rotation: bool = False, annealing: str = 'none', use_pareto: bool = False, sparse: tp.Union[bool, int] = False, smoother: bool = False, super_radii: bool = False, roulette_size: int = 2, antismooth: int = 55, crossover_type: str = 'none', forced_discretization: bool = False) -> None: ...
    def _internal_ask_candidate(self) -> p.Parameter: ...
    def _internal_tell(self, x: tp.ArrayLike, loss: tp.FloatLoss) -> None: ...
    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        '''Called whenever calling :code:`tell` on a candidate that was "asked".'''

class ParametrizedOnePlusOne(base.ConfiguredOptimizer):
    '''Simple but sometimes powerfull class of optimization algorithm.
    This use asynchronous updates, so that (1+1) can actually be parallel and even
    performs quite well in such a context - this is naturally close to (1+lambda).


    Parameters
    ----------
    noise_handling: str or Tuple[str, float]
        Method for handling the noise. The name can be:

        - `"random"`: a random point is reevaluated regularly, this uses the one-fifth adaptation rule,
          going back to Schumer and Steiglitz (1968). It was independently rediscovered by Devroye (1972) and Rechenberg (1973).
        - `"optimistic"`: the best optimistic point is reevaluated regularly, optimism in front of uncertainty
        - a coefficient can to tune the regularity of these reevaluations (default .05)
    mutation: str
        One of the available mutations from:

        - `"gaussian"`: standard mutation by adding a Gaussian random variable (with progressive
          widening) to the best pessimistic point
        - `"cauchy"`: same as Gaussian but with a Cauchy distribution.
        - `"discrete"`: when a variable is mutated (which happens with probability 1/d in dimension d), it\'s just
             randomly drawn. This means that on average, only one variable is mutated.
        - `"discreteBSO"`: as in brainstorm optimization, we slowly decrease the mutation rate from 1 to 1/d.
        - `"fastga"`: FastGA mutations from the current best
        - `"doublefastga"`: double-FastGA mutations from the current best (Doerr et al, Fast Genetic Algorithms, 2017)
        - `"rls"`: Randomized Local Search (randomly mutate one and only one variable).
        - `"portfolio"`: Random number of mutated bits (called niform mixing in
          Dang & Lehre "Self-adaptation of Mutation Rates in Non-elitist Population", 2016)
        - `"lengler"`: specific mutation rate chosen as a function of the dimension and iteration index.
        - `"lengler{2|3|half|fourth}"`: variant of Lengler
    crossover: bool
        whether to add a genetic crossover step every other iteration.
    use_pareto: bool
        whether to restart from a random pareto element in multiobjective mode, instead of the last one added
    sparse: bool
        whether we have random mutations setting variables to 0.
    smoother: bool
        whether we suggest smooth mutations.

    Notes
    -----
    After many papers advocated the mutation rate 1/d in the discrete (1+1) for the discrete case,
    `it was proposed <https://arxiv.org/abs/1606.05551>`_ to use a randomly
    drawn mutation rate. `Fast genetic algorithms <https://arxiv.org/abs/1703.03334>`_ are based on a similar idea
    These two simple methods perform quite well on a wide range of problems.

    '''
    def __init__(self, *, noise_handling: tp.Optional[tp.Union[str, tp.Tuple[str, float]]] = None, tabu_length: int = 0, mutation: str = 'gaussian', crossover: bool = False, rotation: bool = False, annealing: str = 'none', use_pareto: bool = False, sparse: bool = False, smoother: bool = False, super_radii: bool = False, roulette_size: int = 2, antismooth: int = 55, crossover_type: str = 'none') -> None: ...

OnePlusOne: Incomplete
OnePlusLambda: Incomplete
NoisyOnePlusOne: Incomplete
DiscreteOnePlusOne: Incomplete
SADiscreteLenglerOnePlusOneExp09: Incomplete
SADiscreteLenglerOnePlusOneExp099: Incomplete
SADiscreteLenglerOnePlusOneExp09Auto: Incomplete
SADiscreteLenglerOnePlusOneLinAuto: Incomplete
SADiscreteLenglerOnePlusOneLin1: Incomplete
SADiscreteLenglerOnePlusOneLin100: Incomplete
SADiscreteOnePlusOneExp099: Incomplete
SADiscreteOnePlusOneLin100: Incomplete
SADiscreteOnePlusOneExp09: Incomplete
DiscreteOnePlusOneT: Incomplete
PortfolioDiscreteOnePlusOne: Incomplete
PortfolioDiscreteOnePlusOneT: Incomplete
DiscreteLenglerOnePlusOne: Incomplete
DiscreteLengler2OnePlusOne: Incomplete
DiscreteLengler3OnePlusOne: Incomplete
DiscreteLenglerHalfOnePlusOne: Incomplete
DiscreteLenglerFourthOnePlusOne: Incomplete
DiscreteLenglerOnePlusOneT: Incomplete
AdaptiveDiscreteOnePlusOne: Incomplete
LognormalDiscreteOnePlusOne: Incomplete
XLognormalDiscreteOnePlusOne: Incomplete
XSmallLognormalDiscreteOnePlusOne: Incomplete
BigLognormalDiscreteOnePlusOne: Incomplete
SmallLognormalDiscreteOnePlusOne: Incomplete
TinyLognormalDiscreteOnePlusOne: Incomplete
HugeLognormalDiscreteOnePlusOne: Incomplete
AnisotropicAdaptiveDiscreteOnePlusOne: Incomplete
DiscreteBSOOnePlusOne: Incomplete
DiscreteDoerrOnePlusOne: Incomplete
CauchyOnePlusOne: Incomplete
OptimisticNoisyOnePlusOne: Incomplete
OptimisticDiscreteOnePlusOne: Incomplete
OLNDiscreteOnePlusOne: Incomplete
NoisyDiscreteOnePlusOne: Incomplete
DoubleFastGADiscreteOnePlusOne: Incomplete
RLSOnePlusOne: Incomplete
SparseDoubleFastGADiscreteOnePlusOne: Incomplete
RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne: Incomplete
RecombiningPortfolioDiscreteOnePlusOne: Incomplete

class _CMA(base.Optimizer):
    _CACHE_KEY: str
    algorithm: Incomplete
    _config: Incomplete
    _popsize: Incomplete
    _to_be_asked: tp.Deque[np.ndarray]
    _to_be_told: tp.List[p.Parameter]
    _num_spawners: Incomplete
    _parents: Incomplete
    _es: tp.Any
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1, config: tp.Optional['ParametrizedCMA'] = None, algorithm: str = 'quad') -> None: ...
    @property
    def es(self) -> tp.Any: ...
    def _internal_ask_candidate(self) -> p.Parameter: ...
    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None: ...
    def _internal_provide_recommendation(self) -> np.ndarray: ...

class ParametrizedCMA(base.ConfiguredOptimizer):
    """CMA-ES optimizer,
    This evolution strategy uses Gaussian sampling, iteratively modified
    for searching in the best directions.
    This optimizer wraps an external implementation: https://github.com/CMA-ES/pycma

    Parameters
    ----------
    scale: float
        scale of the search
    elitist: bool
        whether we switch to elitist mode, i.e. mode + instead of comma,
        i.e. mode in which we always keep the best point in the population.
    popsize: Optional[int] = None
        population size, should be n * self.num_workers for int n >= 1.
        default is max(self.num_workers, 4 + int(3 * np.log(self.dimension)))
    popsize_factor: float = 3.
        factor in the formula for computing the population size
    diagonal: bool
        use the diagonal version of CMA (advised in big dimension)
    high_speed: bool
        use metamodel for recommendation
    fcmaes: bool
        use fast implementation, doesn't support diagonal=True.
        produces equivalent results, preferable for high dimensions or
        if objective function evaluation is fast.
    random_init: bool
        Use a randomized initialization
    inopts: optional dict
        use this to averride any inopts parameter of the wrapped CMA optimizer
        (see https://github.com/CMA-ES/pycma)
    """
    scale: Incomplete
    elitist: Incomplete
    zero: Incomplete
    popsize: Incomplete
    popsize_factor: Incomplete
    diagonal: Incomplete
    fcmaes: Incomplete
    high_speed: Incomplete
    random_init: Incomplete
    inopts: Incomplete
    def __init__(self, *, scale: float = 1.0, elitist: bool = False, popsize: tp.Optional[int] = None, popsize_factor: float = 3.0, diagonal: bool = False, zero: bool = False, high_speed: bool = False, fcmaes: bool = False, random_init: bool = False, inopts: tp.Optional[tp.Dict[str, tp.Any]] = None, algorithm: str = 'quad') -> None: ...

class ChoiceBase(base.Optimizer):
    """Nevergrad optimizer by competence map."""
    has_noise: Incomplete
    noise_from_instrumentation: Incomplete
    fully_continuous: Incomplete
    has_discrete_not_softmax: Incomplete
    _has_discrete: Incomplete
    _arity: int
    _optim: tp.Optional[base.Optimizer]
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...
    @property
    def optim(self) -> base.Optimizer: ...
    def _select_optimizer_cls(self) -> base.OptCls: ...
    def _internal_ask_candidate(self) -> p.Parameter: ...
    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None: ...
    def recommend(self) -> p.Parameter: ...
    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None: ...
    def _info(self) -> tp.Dict[str, tp.Any]: ...
    def enable_pickling(self) -> None: ...

OldCMA: Incomplete
LargeCMA: Incomplete
LargeDiagCMA: Incomplete
TinyCMA: Incomplete
CMAbounded: Incomplete
CMAsmall: Incomplete
CMAstd: Incomplete
CMApara: Incomplete
CMAtuning: Incomplete

class MetaCMA(ChoiceBase):
    """Nevergrad CMA optimizer by competence map. You might modify this one for designing your own competence map."""
    def _select_optimizer_cls(self) -> base.OptCls: ...

DiagonalCMA: Incomplete
EDCMA: Incomplete
SDiagonalCMA: Incomplete
FCMA: Incomplete

class CMA(MetaCMA): ...

class _PopulationSizeController:
    """Population control scheme for TBPSA and EDA"""
    llambda: Incomplete
    min_mu: Incomplete
    mu: Incomplete
    dimension: Incomplete
    num_workers: Incomplete
    _loss_record: tp.List[float]
    def __init__(self, llambda: int, mu: int, dimension: int, num_workers: int = 1) -> None: ...
    def add_value(self, loss: tp.FloatLoss) -> None: ...

class EDA(base.Optimizer):
    """Estimation of distribution algorithm.

    Population-size equal to lambda = 4 x dimension by default.
    """
    _POPSIZE_ADAPTATION: bool
    _COVARIANCE_MEMORY: bool
    sigma: int
    covariance: Incomplete
    popsize: Incomplete
    current_center: np.ndarray
    children: tp.List[p.Parameter]
    parents: tp.List[p.Parameter]
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...
    def _internal_provide_recommendation(self) -> tp.ArrayLike: ...
    def _internal_ask_candidate(self) -> p.Parameter: ...
    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None: ...
    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None: ...

class AXP(base.Optimizer):
    """AX-platform.

    Usually computationally slow and not better than the rest
    in terms of performance per iteration.
    Maybe prefer HyperOpt or Cobyla for low budget optimization.
    """
    ax_parametrization: Incomplete
    ax_client: Incomplete
    _trials: Incomplete
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...
    def _internal_ask_candidate(self) -> p.Parameter: ...
    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None: ...
    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None: ...

class PCEDA(EDA):
    _POPSIZE_ADAPTATION: bool
    _COVARIANCE_MEMORY: bool

class MPCEDA(EDA):
    _POPSIZE_ADAPTATION: bool
    _COVARIANCE_MEMORY: bool

class MEDA(EDA):
    _POPSIZE_ADAPTATION: bool
    _COVARIANCE_MEMORY: bool

class _TBPSA(base.Optimizer):
    """Test-based population-size adaptation.

    Population-size equal to lambda = 4 x dimension.
    Test by comparing the first fifth and the last fifth of the 5lambda evaluations.
    """
    sigma: int
    naive: Incomplete
    popsize: Incomplete
    current_center: np.ndarray
    parents: tp.List[p.Parameter]
    children: tp.List[p.Parameter]
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1, naive: bool = True, initial_popsize: tp.Optional[int] = None) -> None: ...
    def recommend(self) -> p.Parameter: ...
    def _internal_ask_candidate(self) -> p.Parameter: ...
    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None: ...
    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None: ...

class ParametrizedTBPSA(base.ConfiguredOptimizer):
    """`Test-based population-size adaptation <https://homepages.fhv.at/hgb/New-Papers/PPSN16_HB16.pdf>`_
    This method, based on adapting the population size, performs the best in
    many noisy optimization problems, even in large dimension

    Parameters
    ----------
    naive: bool
        set to False for noisy problem, so that the best points will be an
        average of the final population.
    initial_popsize: Optional[int]
        initial (and minimal) population size (default: 4 x dimension)

    Note
    ----
    Derived from:
    Hellwig, Michael & Beyer, Hans-Georg. (2016).
    Evolution under Strong Noise: A Self-Adaptive Evolution Strategy
    Reaches the Lower Performance Bound -- the pcCMSA-ES.
    https://homepages.fhv.at/hgb/New-Papers/PPSN16_HB16.pdf
    """
    def __init__(self, *, naive: bool = True, initial_popsize: tp.Optional[int] = None) -> None: ...

TBPSA: Incomplete
NaiveTBPSA: Incomplete

class NoisyBandit(base.Optimizer):
    """UCB.
    This is upper confidence bound (adapted to minimization),
    with very poor parametrization; in particular, the logarithmic term is set to zero.
    Infinite arms: we add one arm when `20 * #ask >= #arms ** 3`.
    """
    def _internal_ask(self) -> tp.ArrayLike: ...

class _PSO(base.Optimizer):
    _config: Incomplete
    llambda: Incomplete
    _uid_queue: Incomplete
    population: tp.Dict[str, p.Parameter]
    _best: Incomplete
    previous_candidate: tp.Optional[tp.Any]
    previous_speed: tp.Optional[tp.Any]
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1, config: tp.Optional['ConfPSO'] = None) -> None: ...
    def _internal_ask_candidate(self) -> p.Parameter: ...
    def _get_boxed_data(self, particle: p.Parameter) -> np.ndarray: ...
    def _spawn_mutated_particle(self, particle: p.Parameter) -> p.Parameter: ...
    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None: ...
    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None: ...

class ConfPSO(base.ConfiguredOptimizer):
    '''`Particle Swarm Optimization <https://en.wikipedia.org/wiki/Particle_swarm_optimization>`_
    is based on a set of particles with their inertia.
    Wikipedia provides a beautiful illustration ;) (see link)


    Parameters
    ----------
    transform: str
        name of the transform to use to map from PSO optimization space to R-space.
    popsize: int
        population size of the particle swarm. Defaults to max(40, num_workers)
    omega: float
        particle swarm optimization parameter
    phip: float
        particle swarm optimization parameter
    phig: float
        particle swarm optimization parameter
    qo: bool
        whether we use quasi-opposite initialization
    sqo: bool
        whether we use quasi-opposite initialization for speed
    so: bool
        whether we use the special quasi-opposite initialization for speed

    Note
    ----
    - Using non-default "transform" and "wide" parameters can lead to extreme values
    - Implementation partially following SPSO2011. However, no randomization of the population order.
    - Reference:
      M. Zambrano-Bigiarini, M. Clerc and R. Rojas,
      Standard Particle Swarm Optimisation 2011 at CEC-2013: A baseline for future PSO improvements,
      2013 IEEE Congress on Evolutionary Computation, Cancun, 2013, pp. 2337-2344.
      https://ieeexplore.ieee.org/document/6557848
    '''
    transform: Incomplete
    popsize: Incomplete
    omega: Incomplete
    phip: Incomplete
    phig: Incomplete
    qo: Incomplete
    sqo: Incomplete
    so: Incomplete
    def __init__(self, transform: str = 'identity', popsize: tp.Optional[int] = None, omega: float = ..., phip: float = ..., phig: float = ..., qo: bool = False, sqo: bool = False, so: bool = False) -> None: ...
ConfiguredPSO = ConfPSO
RealSpacePSO: Incomplete
PSO: Incomplete
QOPSO: Incomplete
QORealSpacePSO: Incomplete
SQOPSO: Incomplete
SOPSO: Incomplete
SQORealSpacePSO: Incomplete

class SPSA(base.Optimizer):
    '''The First order SPSA algorithm as shown in [1,2,3], with implementation details
    from [4,5].

    1) https://en.wikipedia.org/wiki/Simultaneous_perturbation_stochastic_approximation
    2) https://www.chessprogramming.org/SPSA
    3) Spall, James C. "Multivariate stochastic approximation using a simultaneous perturbation gradient approximation."
       IEEE transactions on automatic control 37.3 (1992): 332-341.
    4) Section 7.5.2 in "Introduction to Stochastic Search and Optimization: Estimation, Simulation and Control" by James C. Spall.
    5) Pushpendre Rastogi, Jingyi Zhu, James C. Spall CISS (2016).
       Efficient implementation of Enhanced Adaptive Simultaneous Perturbation Algorithms.
    '''
    no_parallelization: bool
    init: bool
    idx: int
    delta: tp.Any
    ym: tp.Optional[np.ndarray]
    yp: tp.Optional[np.ndarray]
    t: np.ndarray
    avg: np.ndarray
    A: Incomplete
    c: float
    a: float
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...
    def _ck(self, k: int) -> float:
        """c_k determines the pertubation."""
    def _ak(self, k: int) -> float:
        """a_k is the learning rate."""
    def _internal_ask(self) -> tp.ArrayLike: ...
    def _internal_tell(self, x: tp.ArrayLike, loss: tp.FloatLoss) -> None: ...
    def _internal_provide_recommendation(self) -> tp.ArrayLike: ...

class _Rescaled(base.Optimizer):
    """Proposes a version of a base optimizer which works at a different scale."""
    _optimizer: Incomplete
    no_parallelization: Incomplete
    _subcandidates: tp.Dict[str, p.Parameter]
    scale: Incomplete
    shift: Incomplete
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1, base_optimizer: base.OptCls = ..., scale: tp.Optional[float] = None, shift: tp.Optional[float] = None) -> None: ...
    def rescale_candidate(self, candidate: p.Parameter, inverse: bool = False) -> p.Parameter: ...
    def _internal_ask_candidate(self) -> p.Parameter: ...
    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None: ...
    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None: ...
    def enable_pickling(self) -> None: ...

class Rescaled(base.ConfiguredOptimizer):
    """Configured optimizer for creating rescaled optimization algorithms.

    By default, scales to sqrt(log(budget)/n_dimensions).

    Parameters
    ----------
    base_optimizer: base.OptCls
        optimization algorithm to be rescaled.
    scale: how much do we rescale. E.g. 0.001 if we want to focus on the center
        with std 0.001 (assuming the std of the domain is set to 1).
    """
    def __init__(self, *, base_optimizer: base.OptCls = ..., scale: tp.Optional[float] = None, shift: tp.Optional[float] = None) -> None: ...

RescaledCMA: Incomplete
TinyLhsDE: Incomplete
LocalBFGS: Incomplete
TinyQODE: Incomplete
TinySQP: Incomplete
MicroSQP: Incomplete
TinySPSA: Incomplete
MicroSPSA: Incomplete
VastLengler: Incomplete
VastDE: Incomplete
LSDE: Incomplete

class SplitOptimizer(base.Optimizer):
    """Combines optimizers, each of them working on their own variables. (use ConfSplitOptimizer)"""
    _config: Incomplete
    _subcandidates: tp.Dict[str, tp.List[p.Parameter]]
    optims: tp.List[base.Optimizer]
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1, config: tp.Optional['ConfSplitOptimizer'] = None) -> None: ...
    def _internal_ask_candidate(self) -> p.Parameter: ...
    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None: ...
    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None: ...
    def _info(self) -> tp.Dict[str, tp.Any]: ...

class ConfSplitOptimizer(base.ConfiguredOptimizer):
    '''Combines optimizers, each of them working on their own variables.

    Parameters
    ----------
    num_optims: int (or float("inf"))
        number of optimizers to create (if not provided through :code:`num_vars: or
        :code:`max_num_vars`)
    num_vars: int or None
        number of variable per optimizer (should not be used if :code:`max_num_vars` or
        :code:`num_optims` is set)
    max_num_vars: int or None
        maximum number of variables per optimizer. Should not be defined if :code:`num_vars` or
        :code:`num_optims` is defined since they will be chosen automatically.
    progressive: optional bool
        whether we progressively add optimizers.
    non_deterministic_descriptor: bool
        subparts parametrization descriptor is set to noisy function.
        This can have an impact for optimizer selection for competence maps.

    Example
    -------
    for 5 optimizers, each of them working on 2 variables, one can use:

    opt = ConfSplitOptimizer(num_vars=[2, 2, 2, 2, 2])(parametrization=10, num_workers=3)
    or equivalently:
    opt = SplitOptimizer(parametrization=10, num_workers=3, num_vars=[2, 2, 2, 2, 2])
    Given that all optimizers have the same number of variables, one can also run:
    opt = SplitOptimizer(parametrization=10, num_workers=3, num_optims=5)

    Note
    ----
    By default, it uses CMA for multivariate groups and RandomSearch for monovariate groups.

    Caution
    -------
    The variables refer to the deep representation used by optimizers.
    For example, a categorical variable with 5 possible values becomes 5 continuous variables.
    '''
    num_optims: Incomplete
    num_vars: Incomplete
    max_num_vars: Incomplete
    multivariate_optimizer: Incomplete
    monovariate_optimizer: Incomplete
    progressive: Incomplete
    non_deterministic_descriptor: Incomplete
    def __init__(self, *, num_optims: tp.Optional[float] = None, num_vars: tp.Optional[tp.List[int]] = None, max_num_vars: tp.Optional[int] = None, multivariate_optimizer: base.OptCls = ..., monovariate_optimizer: base.OptCls = ..., progressive: bool = False, non_deterministic_descriptor: bool = True) -> None: ...

class NoisySplit(base.ConfiguredOptimizer):
    '''Non-progressive noisy split of variables based on 1+1

    Parameters
    ----------
    num_optims: optional int
        number of optimizers (one per variable if float("inf"))
    discrete: bool
        uses OptimisticDiscreteOnePlusOne if True, else NoisyOnePlusOne
    '''
    def __init__(self, *, num_optims: tp.Optional[float] = None, discrete: bool = False) -> None: ...

class ConfPortfolio(base.ConfiguredOptimizer):
    """Alternates :code:`ask()` on several optimizers

    Parameters
    ----------
    optimizers: list of Optimizer, optimizer name, Optimizer class or ConfiguredOptimizer
        the list of optimizers to use.
    warmup_ratio: optional float
        ratio of the budget used before choosing to focus on one optimizer

    Notes
    -----
    - if providing an initialized  optimizer, the parametrization of the optimizer
      must be the exact same instance as the one of the Portfolio.
    - this API is temporary and will be renamed very soon
    """
    optimizers: Incomplete
    warmup_ratio: Incomplete
    no_crossing: Incomplete
    def __init__(self, *, optimizers: tp.Sequence[tp.Union[base.Optimizer, base.OptCls, str]] = (), warmup_ratio: tp.Optional[float] = None, no_crossing: bool = False) -> None: ...

class Portfolio(base.Optimizer):
    """Passive portfolio of CMA, 2-pt DE and Scr-Hammersley."""
    no_crossing: bool
    _config: Incomplete
    optims: tp.List[base.Optimizer]
    str_info: str
    turns: Incomplete
    _current: int
    _warmup_budget: tp.Optional[int]
    num_times: Incomplete
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1, config: tp.Optional['ConfPortfolio'] = None) -> None: ...
    def _internal_ask_candidate(self) -> p.Parameter: ...
    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None: ...
    def enable_pickling(self) -> None: ...

ParaPortfolio: Incomplete
ASCMADEthird: Incomplete
MultiCMA: Incomplete
MultiDS: Incomplete
TripleCMA: Incomplete
PolyCMA: Incomplete
MultiScaleCMA: Incomplete
LPCMA: Incomplete
VLPCMA: Incomplete

class _MetaModel(base.Optimizer):
    frequency_ratio: Incomplete
    algorithm: Incomplete
    degree: Incomplete
    _optim: Incomplete
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1, *, multivariate_optimizer: tp.Optional[base.OptCls] = None, frequency_ratio: float = 0.9, algorithm: str, degree: int = 2) -> None: ...
    def _internal_ask_candidate(self) -> p.Parameter: ...
    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None: ...
    def _internal_provide_recommendation(self) -> tp.Optional[tp.ArrayLike]: ...
    def enable_pickling(self) -> None: ...

class ParametrizedMetaModel(base.ConfiguredOptimizer):
    """
    Adds a metamodel to an optimizer.
    The optimizer is alway OnePlusOne if dimension is 1.

    Parameters
    ----------
    multivariate_optimizer: base.OptCls or None
        Optimizer to which the metamodel is added
    frequency_ratio: float
        used for deciding the frequency at which we use the metamodel
    """
    def __init__(self, *, multivariate_optimizer: tp.Optional[base.OptCls] = None, frequency_ratio: float = 0.9, algorithm: str = 'quad', degree: int = 2) -> None: ...

MetaModel: Incomplete
NeuralMetaModel: Incomplete
ImageMetaModel: Incomplete
ImageMetaModelD: Incomplete
ImageMetaModelE: Incomplete
SVMMetaModel: Incomplete
RFMetaModel: Incomplete
Quad1MetaModel: Incomplete
Neural1MetaModel: Incomplete
SVM1MetaModel: Incomplete
RF1MetaModel: Incomplete
Quad1MetaModelE: Incomplete
Neural1MetaModelE: Incomplete
SVM1MetaModelE: Incomplete
RF1MetaModelE: Incomplete
Quad1MetaModelD: Incomplete
Neural1MetaModelD: Incomplete
SVM1MetaModelD: Incomplete
RF1MetaModelD: Incomplete
Quad1MetaModelOnePlusOne: Incomplete
Neural1MetaModelOnePlusOne: Incomplete
SVM1MetaModelOnePlusOne: Incomplete
RF1MetaModelOnePlusOne: Incomplete
MetaModelOnePlusOne: Incomplete
ImageMetaModelOnePlusOne: Incomplete
VoxelizeMetaModelOnePlusOne: Incomplete
ImageMetaModelDiagonalCMA: Incomplete
MetaModelDSproba: Incomplete
RFMetaModelOnePlusOne: Incomplete
ImageMetaModelLengler: Incomplete
ImageMetaModelLogNormal: Incomplete
RF1MetaModelLogNormal: Incomplete
SVM1MetaModelLogNormal: Incomplete
Neural1MetaModelLogNormal: Incomplete
RFMetaModelLogNormal: Incomplete
SVMMetaModelLogNormal: Incomplete
MetaModelLogNormal: Incomplete
NeuralMetaModelLogNormal: Incomplete
MetaModelPSO: Incomplete
RFMetaModelPSO: Incomplete
SVMMetaModelPSO: Incomplete
MetaModelDE: Incomplete
MetaModelQODE: Incomplete
NeuralMetaModelDE: Incomplete
SVMMetaModelDE: Incomplete
RFMetaModelDE: Incomplete
MetaModelTwoPointsDE: Incomplete
NeuralMetaModelTwoPointsDE: Incomplete
SVMMetaModelTwoPointsDE: Incomplete
RFMetaModelTwoPointsDE: Incomplete

def rescaled(n: int, o: tp.Any): ...

class MultiBFGSPlus(Portfolio):
    """Passive portfolio of several BFGS."""
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...

class LogMultiBFGSPlus(Portfolio):
    """Passive portfolio of several BFGS (at least logarithmic in the budget)."""
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...

class SqrtMultiBFGSPlus(Portfolio):
    """Passive portfolio of several BFGS (at least sqrt of budget)."""
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...

class MultiCobylaPlus(Portfolio):
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...

class MultiSQPPlus(Portfolio):
    """Passive portfolio of several SQP."""
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...

class BFGSCMAPlus(Portfolio):
    """Passive portfolio of CMA and several BFGS; at least log(budget)."""
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...

class LogBFGSCMAPlus(Portfolio):
    """Passive portfolio of CMA and several BFGS; at least log(budget)."""
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...

class SqrtBFGSCMAPlus(Portfolio):
    """Passive portfolio of CMA and several BFGS; at least sqrt(budget)."""
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...

class SQPCMAPlus(Portfolio):
    """Passive portfolio of CMA and several SQP."""
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...

class LogSQPCMAPlus(Portfolio):
    """Passive portfolio of CMA and several SQP."""
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...

class SqrtSQPCMAPlus(Portfolio):
    """Passive portfolio of MetaCMA and many SQP."""
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...

class MultiBFGS(Portfolio):
    """Passive portfolio of many BFGS."""
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...

class LogMultiBFGS(Portfolio):
    """Passive portfolio of many BFGS."""
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...

class SqrtMultiBFGS(Portfolio):
    """Passive portfolio of many BFGS."""
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...

class MultiCobyla(Portfolio):
    """Passive portfolio of several Cobyla."""
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...

class ForceMultiCobyla(Portfolio):
    """Passive portfolio of several Cobyla."""
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...

class MultiSQP(Portfolio):
    """Passive portfolio of several SQP."""
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...

class BFGSCMA(Portfolio):
    """Passive portfolio of MetaCMA and many BFGS."""
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...

class LogBFGSCMA(Portfolio):
    """Passive portfolio of MetaCMA and many BFGS."""
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...

class SqrtBFGSCMA(Portfolio):
    """Passive portfolio of MetaCMA and many BFGS."""
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...

class SQPCMA(Portfolio):
    """Passive portfolio of MetaCMA and many SQP."""
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...

class LogSQPCMA(Portfolio):
    """Passive portfolio of MetaCMA and many SQP."""
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...

class SqrtSQPCMA(Portfolio):
    """Passive portfolio of MetaCMA and many SQP."""
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...

class FSQPCMA(Portfolio):
    """Passive portfolio of MetaCMA and many SQP."""
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...

class F2SQPCMA(Portfolio):
    """Passive portfolio of MetaCMA and many SQP."""
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...

class F3SQPCMA(Portfolio):
    """Passive portfolio of MetaCMA and many SQP."""
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...

class MultiDiscrete(Portfolio):
    """Combining 3 Discrete(1+1) optimizers. Active selection at 1/4th of the budget."""
    optims: Incomplete
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...

class CMandAS2(Portfolio):
    """Competence map, with algorithm selection in one of the cases (3 CMAs)."""
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...

class CMandAS3(Portfolio):
    """Competence map, with algorithm selection in one of the cases (3 CMAs)."""
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...

class CM(Portfolio):
    """Competence map, simplest."""
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...

class _FakeFunction:
    """Simple function that returns the loss which was registered just before.
    This is a hack for BO.
    """
    num_digits: Incomplete
    _registered: tp.List[tp.Tuple[np.ndarray, float]]
    def __init__(self, num_digits: int) -> None: ...
    def key(self, num: int) -> str:
        """Key corresponding to the array sample
        (uses zero-filling to keep order)
        """
    def register(self, x: np.ndarray, loss: tp.FloatLoss) -> None: ...
    def __call__(self, **kwargs: float) -> float: ...

class _BO(base.Optimizer):
    _normalizer: Incomplete
    _bo: tp.Optional[BayesianOptimization]
    _fake_function: Incomplete
    _init_budget: Incomplete
    _middle_point: Incomplete
    _InitOpt: tp.Optional[base.ConfiguredOptimizer]
    utility_kind: Incomplete
    utility_kappa: Incomplete
    utility_xi: Incomplete
    gp_parameters: Incomplete
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1, *, initialization: tp.Optional[str] = None, init_budget: tp.Optional[int] = None, middle_point: bool = False, utility_kind: str = 'ucb', utility_kappa: float = 2.576, utility_xi: float = 0.0, gp_parameters: tp.Optional[tp.Dict[str, tp.Any]] = None) -> None: ...
    @property
    def bo(self) -> BayesianOptimization: ...
    def _internal_ask_candidate(self) -> p.Parameter: ...
    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None: ...
    def _internal_provide_recommendation(self) -> tp.Optional[tp.ArrayLike]: ...

class ParametrizedBO(base.ConfiguredOptimizer):
    '''Bayesian optimization.
        Hyperparameter tuning method, based on statistical modeling of the objective function.
        This class is a wrapper over the `bayes_opt <https://github.com/fmfn/BayesianOptimization>`_ package.

        Parameters
        ----------
        initialization: str
            Initialization algorithms (None, "Hammersley", "random" or "LHS")
        init_budget: int or None
            Number of initialization algorithm steps
        middle_point: bool
            whether to sample the 0 point first
        utility_kind: str
            Type of utility function to use among "ucb", "ei" and "poi"
        utility_kappa: float
            Kappa parameter for the utility function
        utility_xi: float
            Xi parameter for the utility function
        gp_parameters: dict
            dictionnary of parameters for the gaussian process
        '''
    no_parallelization: bool
    def __init__(self, *, initialization: tp.Optional[str] = None, init_budget: tp.Optional[int] = None, middle_point: bool = False, utility_kind: str = 'ucb', utility_kappa: float = 2.576, utility_xi: float = 0.0, gp_parameters: tp.Optional[tp.Dict[str, tp.Any]] = None) -> None: ...

BO: Incomplete
BOSplit: Incomplete

class _BayesOptim(base.Optimizer):
    _config: Incomplete
    _normalizer: Incomplete
    _buffer: tp.List[float]
    _newX: tp.List[float]
    _losses: tp.List[float]
    _alg: Incomplete
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1, *, config: tp.Optional['BayesOptim'] = None) -> None: ...
    def _internal_ask_candidate(self) -> p.Parameter: ...
    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None: ...
    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None: ...

class BayesOptim(base.ConfiguredOptimizer):
    '''
    Algorithms from bayes-optim package.

    We use:
    - BO
    - PCA-BO: Principle Component Analysis (PCA) Bayesian Optimization for dimensionality reduction in BO

    References

    [RaponiWB+20]
        Raponi, Elena, Hao Wang, Mariusz Bujny, Simonetta Boria, and Carola Doerr.
        "High dimensional bayesian optimization assisted by principal component analysis."
        In International Conference on Parallel Problem Solving from Nature, pp. 169-183.
        Springer, Cham, 2020.


    Parameters
    ----------
    init_budget: int or None
        Number of initialization algorithm steps
    pca: bool
        whether to use the PCA transformation defining PCA-BO rather than BO
    n_components: float or 0.95
        Principal axes in feature space, representing the directions of maximum variance in the data.
        It represents the percentage of explained variance
    prop_doe_factor: float or None
        Percentage of the initial budget used for DoE and eventually overwriting init_budget
    '''
    no_parallelization: bool
    recast: bool
    init_budget: Incomplete
    pca: Incomplete
    n_components: Incomplete
    prop_doe_factor: Incomplete
    def __init__(self, *, init_budget: tp.Optional[int] = None, pca: tp.Optional[bool] = False, n_components: tp.Optional[float] = 0.95, prop_doe_factor: tp.Optional[float] = None) -> None: ...

PCABO: Incomplete
BayesOptimBO: Incomplete

class _Chain(base.Optimizer):
    no_crossing: Incomplete
    optimizers: tp.List[base.Optimizer]
    budgets: Incomplete
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1, *, optimizers: tp.Optional[tp.Sequence[tp.Union[base.ConfiguredOptimizer, tp.Type[base.Optimizer]]]] = None, budgets: tp.Sequence[tp.Union[str, int]] = (10,), no_crossing: tp.Optional[bool] = False) -> None: ...
    def _internal_ask_candidate(self) -> p.Parameter: ...
    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None: ...
    def enable_pickling(self) -> None: ...

class Chaining(base.ConfiguredOptimizer):
    """
    A chaining consists in running algorithm 1 during T1, then algorithm 2 during T2, then algorithm 3 during T3, etc.
    Each algorithm is fed with what happened before it.

    Parameters
    ----------
    optimizers: list of Optimizer classes
        the sequence of optimizers to use
    budgets: list of int
        the corresponding budgets for each optimizer but the last one

    """
    def __init__(self, optimizers: tp.Sequence[tp.Union[base.ConfiguredOptimizer, tp.Type[base.Optimizer]]], budgets: tp.Sequence[tp.Union[str, int]], no_crossing: tp.Optional[bool] = False) -> None: ...

CMAL: Incomplete
GeneticDE: Incomplete
MemeticDE: Incomplete
QNDE: Incomplete
ChainDE: Incomplete
OpoDE: Incomplete
OpoTinyDE: Incomplete
Carola1: Incomplete
Carola2: Incomplete
DS2: Incomplete
Carola4: Incomplete
DS4: Incomplete
Carola5: Incomplete
DS5: Incomplete
Carola6: Incomplete
DS6: Incomplete
PCarola6: Incomplete
pCarola6: Incomplete
Carola7: Incomplete
Carola8: Incomplete
DS8: Incomplete
Carola9: Incomplete
DS9: Incomplete
Carola10: Incomplete

class Carola3(Portfolio):
    """Passive portfolio of MetaCMA and many SQP."""
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...

BAR: Incomplete
BAR2: Incomplete
BAR3: Incomplete
discretememetic: Incomplete
ChainCMAPowell: Incomplete
ChainDSPowell: Incomplete
ChainMetaModelSQP: Incomplete
ChainMetaModelDSSQP: Incomplete
ChainMetaModelPowell: Incomplete
ChainDiagonalCMAPowell: Incomplete
ChainNaiveTBPSAPowell: Incomplete
ChainNaiveTBPSACMAPowell: Incomplete
BAR4: Incomplete

class cGA(base.Optimizer):
    """`Compact Genetic Algorithm <https://ieeexplore.ieee.org/document/797971>`_.
    A discrete optimization algorithm, introduced in and often used as a first baseline.
    """
    _arity: Incomplete
    _penalize_cheap_violations: bool
    p: np.ndarray
    llambda: Incomplete
    _previous_value_candidate: tp.Optional[tp.Tuple[float, np.ndarray]]
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1, arity: tp.Optional[int] = None) -> None: ...
    def _internal_ask_candidate(self) -> p.Parameter: ...
    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None: ...

class _EMNA(base.Optimizer):
    """Simple Estimation of Multivariate Normal Algorithm (EMNA)."""
    isotropic: bool
    naive: bool
    population_size_adaptation: Incomplete
    min_coef_parallel_context: int
    sigma: tp.Union[float, np.ndarray]
    popsize: Incomplete
    current_center: np.ndarray
    parents: tp.List[p.Parameter]
    children: tp.List[p.Parameter]
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1, isotropic: bool = True, naive: bool = True, population_size_adaptation: bool = False, initial_popsize: tp.Optional[int] = None) -> None: ...
    def recommend(self) -> p.Parameter: ...
    def _internal_ask_candidate(self) -> p.Parameter: ...
    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None: ...
    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None: ...

class EMNA(base.ConfiguredOptimizer):
    """Estimation of Multivariate Normal Algorithm
    This algorithm is quite efficient in a parallel context, i.e. when
    the population size is large.

    Parameters
    ----------
    isotropic: bool
        isotropic version on EMNA if True, i.e. we have an
        identity matrix for the Gaussian, else  we here consider the separable
        version, meaning we have a diagonal matrix for the Gaussian (anisotropic)
    naive: bool
        set to False for noisy problem, so that the best points will be an
        average of the final population.
    population_size_adaptation: bool
        population size automatically adapts to the landscape
    initial_popsize: Optional[int]
        initial (and minimal) population size (default: 4 x dimension)
    """
    def __init__(self, *, isotropic: bool = True, naive: bool = True, population_size_adaptation: bool = False, initial_popsize: tp.Optional[int] = None) -> None: ...

NaiveIsoEMNA: Incomplete

class NGOptBase(base.Optimizer):
    """Nevergrad optimizer by competence map."""
    has_noise: Incomplete
    has_real_noise: Incomplete
    noise_from_instrumentation: Incomplete
    fully_continuous: Incomplete
    has_discrete_not_softmax: Incomplete
    _has_discrete: Incomplete
    _arity: int
    _optim: tp.Optional[base.Optimizer]
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...
    @property
    def optim(self) -> base.Optimizer: ...
    def _select_optimizer_cls(self) -> base.OptCls: ...
    def _internal_ask_candidate(self) -> p.Parameter: ...
    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None: ...
    def recommend(self) -> p.Parameter: ...
    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None: ...
    def _info(self) -> tp.Dict[str, tp.Any]: ...
    def enable_pickling(self) -> None: ...

class NGOptDSBase(NGOptBase):
    """Nevergrad optimizer by competence map."""
    def _select_optimizer_cls(self) -> base.OptCls: ...

class Shiwa(NGOptBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""
    def _select_optimizer_cls(self) -> base.OptCls: ...

class NGO(NGOptBase): ...

class NGOpt4(NGOptBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""
    fully_continuous: Incomplete
    def _select_optimizer_cls(self) -> base.OptCls: ...

class NGOpt8(NGOpt4):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""
    def _select_optimizer_cls(self) -> base.OptCls: ...
    _optim: Incomplete
    def _num_objectives_set_callback(self) -> None: ...

class NGOpt10(NGOpt8):
    def _select_optimizer_cls(self) -> base.OptCls: ...
    def recommend(self) -> p.Parameter: ...

class NGOpt12(NGOpt10):
    def _select_optimizer_cls(self) -> base.OptCls: ...

class NGOpt13(NGOpt12):
    def _select_optimizer_cls(self) -> base.OptCls: ...

class NGOpt14(NGOpt12):
    def _select_optimizer_cls(self) -> base.OptCls: ...

class NGOpt15(NGOpt12):
    def _select_optimizer_cls(self) -> base.OptCls: ...

class NGOpt16(NGOpt15):
    def _select_optimizer_cls(self) -> base.OptCls: ...

class NGOpt21(NGOpt16):
    def _select_optimizer_cls(self) -> base.OptCls: ...

class NGOpt36(NGOpt16):
    def _select_optimizer_cls(self) -> base.OptCls: ...

class NGOpt38(NGOpt16):
    def _select_optimizer_cls(self) -> base.OptCls: ...

class NGOpt39(NGOpt16):
    def _select_optimizer_cls(self) -> base.OptCls: ...

class NGOptRW(NGOpt39):
    def _select_optimizer_cls(self) -> base.OptCls: ...

class NGOptF(NGOpt39):
    def _select_optimizer_cls(self) -> base.OptCls: ...

class NGOptF2(NGOpt39):
    def _select_optimizer_cls(self) -> base.OptCls: ...

class NGOptF3(NGOpt39):
    def _select_optimizer_cls(self) -> base.OptCls: ...

class NGOptF5(NGOpt39):
    def _select_optimizer_cls(self) -> base.OptCls: ...

class NGOpt(NGOpt39): ...

class Wiz(NGOpt16):
    def _select_optimizer_cls(self) -> base.OptCls: ...

class NgIoh(NGOpt16):
    def _select_optimizer_cls(self) -> base.OptCls: ...

class NgIoh2(NGOpt16):
    def _select_optimizer_cls(self) -> base.OptCls: ...

class NgIoh3(NGOpt16):
    def _select_optimizer_cls(self) -> base.OptCls: ...

class NgIoh4(NGOptBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""
    def _select_optimizer_cls(self) -> base.OptCls: ...

class NgIohRW2(NgIoh4):
    def _select_optimizer_cls(self) -> base.OptCls: ...

class NgIoh5(NGOptBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""
    def _select_optimizer_cls(self) -> base.OptCls: ...

class NgIoh6(NGOptBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""
    def _select_optimizer_cls(self) -> base.OptCls: ...

class _MSR(Portfolio):
    """This code applies multiple copies of NGOpt with random weights for the different objective functions.

    Variants dedicated to multiobjective optimization by multiple singleobjective optimization.
    """
    coeffs: tp.List[np.ndarray]
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1, num_single_runs: int = 9, base_optimizer: base.OptCls = ...) -> None: ...
    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None: ...

class MultipleSingleRuns(base.ConfiguredOptimizer):
    """Multiple single-objective runs, in particular for multi-objective optimization.
    Parameters
    ----------
    num_single_runs: int
        number of single runs.
    """
    def __init__(self, *, num_single_runs: int = 9, base_optimizer: base.OptCls = ...) -> None: ...

SmoothDiscreteOnePlusOne: Incomplete
SmoothPortfolioDiscreteOnePlusOne: Incomplete
SmoothDiscreteLenglerOnePlusOne: Incomplete
SmoothDiscreteLognormalOnePlusOne: Incomplete
SuperSmoothDiscreteLenglerOnePlusOne: Incomplete
SuperSmoothTinyLognormalDiscreteOnePlusOne: Incomplete
UltraSmoothDiscreteLenglerOnePlusOne: Incomplete
SmootherDiscreteLenglerOnePlusOne: Incomplete
YoSmoothDiscreteLenglerOnePlusOne: Incomplete
CMALS: Incomplete
UltraSmoothDiscreteLognormalOnePlusOne: Incomplete
CMALYS: Incomplete
CLengler: Incomplete
CMALL: Incomplete
CMAILL: Incomplete
CMASL: Incomplete
CMASL2: Incomplete
CMASL3: Incomplete
CMAL2: Incomplete
CMAL3: Incomplete
SmoothLognormalDiscreteOnePlusOne: Incomplete
SmoothAdaptiveDiscreteOnePlusOne: Incomplete
SmoothRecombiningPortfolioDiscreteOnePlusOne: Incomplete
SmoothRecombiningDiscreteLenglerOnePlusOne: Incomplete
UltraSmoothRecombiningDiscreteLenglerOnePlusOne: Incomplete
UltraSmoothElitistRecombiningDiscreteLognormalOnePlusOne: Incomplete
UltraSmoothElitistRecombiningDiscreteLenglerOnePlusOne: Incomplete
SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne: Incomplete
SuperSmoothRecombiningDiscreteLenglerOnePlusOne: Incomplete
SuperSmoothRecombiningDiscreteLognormalOnePlusOne: Incomplete
SmoothElitistRecombiningDiscreteLenglerOnePlusOne: Incomplete
SmoothElitistRandRecombiningDiscreteLenglerOnePlusOne: Incomplete
SmoothElitistRandRecombiningDiscreteLognormalOnePlusOne: Incomplete
RecombiningDiscreteLenglerOnePlusOne: Incomplete
RecombiningDiscreteLognormalOnePlusOne: Incomplete
MaxRecombiningDiscreteLenglerOnePlusOne: Incomplete
MinRecombiningDiscreteLenglerOnePlusOne: Incomplete
OnePtRecombiningDiscreteLenglerOnePlusOne: Incomplete
TwoPtRecombiningDiscreteLenglerOnePlusOne: Incomplete
RandRecombiningDiscreteLenglerOnePlusOne: Incomplete
RandRecombiningDiscreteLognormalOnePlusOne: Incomplete

class NgIoh7(NGOptBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""
    def _select_optimizer_cls(self) -> base.OptCls: ...

class NgDS11(NGOptDSBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""
    budget: Incomplete
    def _select_optimizer_cls(self, budget: tp.Optional[int] = None) -> base.OptCls: ...

class NgIoh11(NGOptBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""
    budget: Incomplete
    def _select_optimizer_cls(self, budget: tp.Optional[int] = None) -> base.OptCls: ...

class NgIoh14(NgIoh11):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""
    def _select_optimizer_cls(self, budget: tp.Optional[int] = None) -> base.OptCls: ...

class NgIoh13(NgIoh11):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""
    def _select_optimizer_cls(self, budget: tp.Optional[int] = None) -> base.OptCls: ...

class NgIoh15(NgIoh11):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""
    def _select_optimizer_cls(self, budget: tp.Optional[int] = None) -> base.OptCls: ...

class NgIoh12(NgIoh11):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""
    def _select_optimizer_cls(self, budget: tp.Optional[int] = None) -> base.OptCls: ...

class NgIoh16(NgIoh11):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""
    def _select_optimizer_cls(self, budget: tp.Optional[int] = None) -> base.OptCls: ...

class NgIoh17(NgIoh11):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""
    budget: Incomplete
    def _select_optimizer_cls(self, budget: tp.Optional[int] = None) -> base.OptCls: ...

class NgDS(NgDS11):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""
    def _select_optimizer_cls(self, budget: tp.Optional[int] = None) -> base.OptCls: ...

class NgIoh21(NgIoh11):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""
    def _select_optimizer_cls(self, budget: tp.Optional[int] = None) -> base.OptCls: ...

class NgDS2(NgDS11):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""
    def _select_optimizer_cls(self, budget: tp.Optional[int] = None) -> base.OptCls: ...

class NGDSRW(NGOpt39):
    def _select_optimizer_cls(self) -> base.OptCls: ...

class NgIoh20(NgIoh11):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""
    def _select_optimizer_cls(self, budget: tp.Optional[int] = None) -> base.OptCls: ...

class NgIoh19(NgIoh11):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""
    def _select_optimizer_cls(self, budget: tp.Optional[int] = None) -> base.OptCls: ...

class NgIoh18(NgIoh11):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""
    budget: Incomplete
    def _select_optimizer_cls(self, budget: tp.Optional[int] = None) -> base.OptCls: ...

class NgIoh10(NGOptBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""
    def _select_optimizer_cls(self) -> base.OptCls: ...

class NgIoh9(NGOptBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""
    def _select_optimizer_cls(self) -> base.OptCls: ...

class NgIoh8(NGOptBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""
    def _select_optimizer_cls(self) -> base.OptCls: ...

MixDeterministicRL: Incomplete
SpecialRL: Incomplete
NoisyRL1: Incomplete
NoisyRL2: Incomplete
NoisyRL3: Incomplete
FCarola6: Incomplete
Carola11: Incomplete
Carola14: Incomplete
DS14: Incomplete
Carola13: Incomplete
Carola15: Incomplete

class NgIoh12b(NgIoh12):
    no_crossing: bool
    def __init__(self, *args, **kwargs) -> None: ...

class NgIoh13b(NgIoh13):
    no_crossing: bool
    def __init__(self, *args, **kwargs) -> None: ...

class NgIoh14b(NgIoh14):
    no_crossing: bool
    def __init__(self, *args, **kwargs) -> None: ...

class NgIoh15b(NgIoh15):
    no_crossing: bool
    def __init__(self, *args, **kwargs) -> None: ...

NgDS3: Incomplete
NgLn: Incomplete
NgLglr: Incomplete
NgRS: Incomplete

class CSEC(NGOpt39):
    def _select_optimizer_cls(self, budget: tp.Optional[int] = None) -> base.OptCls: ...

class CSEC10(NGOptBase):
    def _select_optimizer_cls(self, budget: tp.Optional[int] = None) -> base.OptCls: ...

class CSEC11(NGOptBase):
    def _select_optimizer_cls(self, budget: tp.Optional[int] = None) -> base.OptCls: ...

class NgIohTuned(CSEC11): ...

SplitCSEC11: Incomplete
SplitSQOPSO: Incomplete
SplitPSO: Incomplete
SplitCMA: Incomplete
SplitQODE: Incomplete
SplitTwoPointsDE: Incomplete
SplitDE: Incomplete
SQOPSODCMA: Incomplete
SQOPSODCMA20: Incomplete
SQOPSODCMA20bar: Incomplete
SparseOrNot: Incomplete
TripleOnePlusOne: Incomplete
TripleDiagonalCMA: Incomplete
NgIohLn: Incomplete
CMALn: Incomplete
CMARS: Incomplete
NgIohRS: Incomplete
PolyLN: Incomplete
MultiLN: Incomplete
ManyLN: Incomplete
NgIohMLn: Incomplete
Zero: Incomplete
StupidRandom: Incomplete
