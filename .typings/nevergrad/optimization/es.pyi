import nevergrad.common.typing as tp
from . import base as base
from _typeshed import Incomplete
from nevergrad.optimization.utils import UidQueue as UidQueue
from nevergrad.parametrization import parameter as p

class _EvolutionStrategy(base.Optimizer):
    """Experimental evolution-strategy-like algorithm
    The behavior is going to evolve
    """
    _population: tp.Dict[str, p.Parameter]
    _uid_queue: Incomplete
    _waiting: tp.List[p.Parameter]
    _config: Incomplete
    _rank_method: tp.Any
    _no_hypervolume: Incomplete
    def __init__(self, parametrization: base.IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1, *, config: tp.Optional['EvolutionStrategy'] = None) -> None: ...
    def _internal_ask_candidate(self) -> p.Parameter: ...
    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None: ...
    def _select(self) -> None: ...

class EvolutionStrategy(base.ConfiguredOptimizer):
    """Experimental evolution-strategy-like algorithm
    The API is going to evolve

    Parameters
    ----------
    recombination_ratio: float
        probability of using a recombination (after the mutation) for generating new offsprings
    popsize: int
        population size of the parents (lambda)
    offsprings: int
        number of generated offsprings (mu)
    only_offsprings: bool
        use only offsprings for the new generation if True (True: lambda,mu, False: lambda+mu)
    ranker: str
        ranker for the multiobjective case (defaults to NSGA2)
    """
    recombination_ratio: Incomplete
    popsize: Incomplete
    offsprings: Incomplete
    only_offsprings: Incomplete
    ranker: Incomplete
    def __init__(self, *, recombination_ratio: float = 0, popsize: int = 40, offsprings: tp.Optional[int] = None, only_offsprings: bool = False, ranker: str = 'nsga2') -> None: ...

RecES: Incomplete
RecMixES: Incomplete
RecMutDE: Incomplete
ES: Incomplete
MixES: Incomplete
MutDE: Incomplete
NonNSGAIIES: Incomplete
