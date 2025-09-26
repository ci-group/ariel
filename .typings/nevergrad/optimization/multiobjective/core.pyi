import nevergrad.common.typing as tp
import numpy as np
from .hypervolume import HypervolumeIndicator as HypervolumeIndicator
from _typeshed import Incomplete
from nevergrad.parametrization import parameter as p

AUTO_BOUND: int

class HypervolumePareto:
    '''Given several functions, and threshold on their values (above which solutions are pointless),
    this object can be used as a single-objective function, the minimization of which
    yields a solution to the original multiobjective problem.

    Parameters
    -----------
    upper_bounds: Tuple of float or np.ndarray
        upper_bounds[i] is a threshold above which x is pointless if function(x)[i] > upper_bounds[i].
    auto_bound: int
        if no upper bounds are provided, number of initial points used to estimate the upper bounds. Their
        loss will be 0 (except if they are uniformly worse than the previous points).
    seed: optional int or RandomState
        seed to use for selecting random subsamples of the pareto
    no_hypervolume: bool
        Most optimizers are designed for single objective and use a float loss.
        To use these in a multi-objective optimization, we provide the negative of
        the hypervolume of the pareto front as the loss.
        If not needed, an optimizer can set this to True.

    Notes
    -----
    When no_hypervolume is false:
    - This function is not stationary!
    - The minimum value obtained for this objective function is -h,
      where h is the hypervolume of the Pareto front obtained, given upper_bounds as a reference point.
    - The callable keeps track of the pareto_front (see attribute paretor_front) and is therefor stateful.
      For this reason it cannot be distributed. A user can however call the multiobjective_function
      remotely, and aggregate locally. This is what happens in the "minimize" method of optimizers.
    when no_hypervolume is true:
    - Hypervolume isn\'t used at all!
    - We simply add every point to the pareto front and state that the pareto front needs to be filtered.
    - The Pareto front is lazily kept up to date because every time you call pareto_front()
      an algorithm is performed that filters the pareto front into what it should be.
    '''
    _auto_bound: int
    _upper_bounds: Incomplete
    _best_volume: Incomplete
    _hypervolume: tp.Optional[HypervolumeIndicator]
    _pareto_needs_filtering: bool
    _no_hypervolume: Incomplete
    _pf: Incomplete
    def __init__(self, *, upper_bounds: tp.Optional[tp.ArrayLike] = None, auto_bound: int = ..., seed: tp.Optional[tp.Union[int, np.random.RandomState]] = None, no_hypervolume: bool = False) -> None: ...
    @property
    def num_objectives(self) -> int: ...
    @property
    def best_volume(self) -> float: ...
    def extend(self, parameters: tp.Sequence[p.Parameter]) -> float: ...
    def add(self, parameter: p.Parameter) -> float:
        """
        when _no_hypervolume = False
            Given parameters and the multiobjective loss, this computes the hypervolume
            and update the state of the function with new points if it belongs to the pareto front.
        when _no_hypervolume = True
            Add every point to pareto front. Don't compute hypervolume. Return 0.0 since loss
            not looked at in this context.
        """
    def _calc_hypervolume(self, parameter: p.Parameter, losses: np.ndarray) -> float: ...
    def pareto_front(self, size: tp.Optional[int] = None, subset: str = 'random', subset_tentatives: int = 12) -> tp.List[p.Parameter]:
        '''Pareto front, as a list of Parameter. The losses can be accessed through
        parameter.losses

        Parameters
        ------------
        size:  int (optional)
            if provided, selects a subset of the full pareto front with the given maximum size
        subset: str
            method for selecting the subset ("random, "loss-covering", "EPS", "domain-covering", "hypervolume")
            EPS is the epsilon indicator described e.g.
                here: https://hal.archives-ouvertes.fr/hal-01159961v2/document
        subset_tentatives: int
            number of random tentatives for finding a better subset

        Returns
        --------
        list
            the list of Parameter of the pareto front
        '''
    def get_min_losses(self) -> tp.List[float]: ...

class ParetoFront:
    _pareto: tp.List[p.Parameter]
    _pareto_needs_filtering: bool
    _no_hypervolume: Incomplete
    _rng: Incomplete
    _hypervolume: tp.Optional[HypervolumeIndicator]
    def __init__(self, *, seed: tp.Optional[tp.Union[int, np.random.RandomState]] = None, no_hypervolume: bool = False) -> None: ...
    def add_to_pareto(self, parameter: p.Parameter) -> None: ...
    def _filter_pareto_front(self):
        """Filters the Pareto front by removing dominated points.
        Implementation from: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
        """
    def get_raw(self) -> tp.List[p.Parameter]:
        """Retrieve current values, which may not be a Pareto front, as they have not been filtered."""
    def get_front(self, size: tp.Optional[int] = None, subset: str = 'random', subset_tentatives: int = 12) -> tp.List[p.Parameter]:
        '''Pareto front, as a list of Parameter. The losses can be accessed through
        parameter.losses

        Parameters
        ------------
        size:  int (optional)
            if provided, selects a subset of the full pareto front with the given maximum size
        subset: str
            method for selecting the subset ("random, "loss-covering", "EPS", "domain-covering", "hypervolume")
            EPS is the epsilon indicator described e.g.
                here: https://hal.archives-ouvertes.fr/hal-01159961v2/document
        subset_tentatives: int
            number of random tentatives for finding a better subset

        Returns
        --------
        list
            the list of Parameter of the pareto front
        '''
