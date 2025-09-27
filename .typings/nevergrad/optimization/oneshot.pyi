import nevergrad.common.typing as tp
import numpy as np
from . import base as base, sequences as sequences, utils as utils
from .base import IntOrParameter as IntOrParameter
from _typeshed import Incomplete
from nevergrad.parametrization import parameter as p

def convex_limit(struct_points: np.ndarray) -> int:
    """Given points in order from best to worst,
    Returns the length of the maximum initial segment of points such that quasiconvexity is verified."""
def hull_center(points: np.ndarray, k: int) -> np.ndarray:
    """Center of the cuboid enclosing the hull."""
def avg_of_k_best(archive: utils.Archive[utils.MultiValue], method: str = 'dimfourth') -> np.ndarray:
    """Operators inspired by the work of Yann Chevaleyre, Laurent Meunier, Clement Royer, Olivier Teytaud, Fabien Teytaud.

    Parameters
    ----------
    archive: utils.Archive[utils.Value]
        Provides a random recommendation instead of the best point so far (for baseline)
    method: str
        If dimfourth, we use the Fteytaud heuristic, i.e. k = min(len(archive) // 4, dimension)
        If exp, we use the Lmeunier method, i.e. k=max(1, len(archiv) // (2**dimension))
        If hull, we use the maximum k <= dimfourth-value, such that the function looks quasiconvex on the k best points.
    """

class OneShotOptimizer(base.Optimizer):
    one_shot: bool
    def _internal_ask_candidate(self) -> p.Parameter: ...

class _RandomSearch(OneShotOptimizer):
    middle_point: Incomplete
    opposition_mode: Incomplete
    stupid: Incomplete
    recommendation_rule: Incomplete
    scale: Incomplete
    sampler: Incomplete
    _opposable_data: tp.Optional[np.ndarray]
    _no_hypervolume: bool
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1, middle_point: bool = False, stupid: bool = False, opposition_mode: tp.Optional[str] = None, sampler: str = 'parametrization', scale: tp.Union[float, str] = 1.0, recommendation_rule: str = 'pessimistic') -> None: ...
    def _internal_ask(self) -> tp.ArrayLike: ...
    def _internal_provide_recommendation(self) -> tp.Optional[tp.ArrayLike]: ...

class RandomSearchMaker(base.ConfiguredOptimizer):
    '''Provides random suggestions.

    Parameters
    ----------
    stupid: bool
        Provides a random recommendation instead of the best point so far (for baseline)
    middle_point: bool
        enforces that the first suggested point (ask) is zero.
    opposition_mode: str or None
        symmetrizes exploration wrt the center: (e.g. https://ieeexplore.ieee.org/document/4424748)
             - full symmetry if "opposite"
             - random * symmetric if "quasi"
    sampler: str
        - parametrization: uses the default sample() method of the parametrization, which samples uniformly
          between bounds and a Gaussian otherwise
        - gaussian: uses a Gaussian distribution
        - cauchy: uses a Cauchy distribution
        use a Cauchy distribution instead of Gaussian distribution
    scale: float or "random"
        scalar for multiplying the suggested point values, or string:
         - "random": uses a randomized pattern for the scale.
         - "auto": scales in function of dimension and budget (version 1: sigma = (1+log(budget)) / (4log(dimension)) )
         - "autotune": scales in function of dimension and budget (version 2: sigma = sqrt(log(budget) / dimension) )
    recommendation_rule: str
        "average_of_best" or "pessimistic" or "average_of_exp_best"; "pessimistic" is
        the default and implies selecting the pessimistic best.
    '''
    one_shot: bool
    def __init__(self, *, middle_point: bool = False, stupid: bool = False, opposition_mode: tp.Optional[str] = None, sampler: str = 'parametrization', scale: tp.Union[float, str] = 1.0, recommendation_rule: str = 'pessimistic') -> None: ...

RandomSearch: Incomplete
QORandomSearch: Incomplete
ORandomSearch: Incomplete
RandomSearchPlusMiddlePoint: Incomplete

class _SamplingSearch(OneShotOptimizer):
    _sampler_instance: tp.Optional[sequences.Sampler]
    _rescaler: tp.Optional[sequences.Rescaler]
    _opposable_data: tp.Optional[np.ndarray]
    _sampler: Incomplete
    opposition_mode: Incomplete
    middle_point: Incomplete
    scrambled: Incomplete
    cauchy: Incomplete
    autorescale: Incomplete
    scale: Incomplete
    rescaled: Incomplete
    recommendation_rule: Incomplete
    _no_hypervolume: bool
    _normalizer: Incomplete
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1, sampler: str = 'Halton', scrambled: bool = False, middle_point: bool = False, opposition_mode: tp.Optional[str] = None, cauchy: bool = False, autorescale: tp.Union[bool, str] = False, scale: float = 1.0, rescaled: bool = False, recommendation_rule: str = 'pessimistic') -> None: ...
    @property
    def sampler(self) -> sequences.Sampler: ...
    def _internal_ask(self) -> tp.ArrayLike: ...
    def _internal_provide_recommendation(self) -> tp.Optional[tp.ArrayLike]: ...

class SamplingSearch(base.ConfiguredOptimizer):
    '''This is a one-shot optimization method, hopefully better than random search
    by ensuring more uniformity.

    Parameters
    ----------
    sampler: str
        Choice of the sampler among "Halton", "Hammersley" and "LHS".
    scrambled: bool
        Adds scrambling to the search; much better in high dimension and rarely worse
        than the original search.
    middle_point: bool
        enforces that the first suggested point (ask) is zero.
    cauchy: bool
        use Cauchy inverse distribution instead of Gaussian when fitting points to real space
        (instead of box).
    scale: float or "random"
        scalar for multiplying the suggested point values.
    rescaled: bool or str
        rescales the sampling pattern to reach the boundaries and/or applies automatic rescaling.
    recommendation_rule: str
        "average_of_best" or "pessimistic"; "pessimistic" is the default and implies selecting the pessimistic best.

    Notes
    -----
    - Halton is a low quality sampling method when the dimension is high; it is usually better
      to use Halton with scrambling.
    - When the budget is known in advance, it is also better to replace Halton by Hammersley.
      Basically the key difference with Halton is adding one coordinate evenly spaced
      (the discrepancy is better).
      budget, low discrepancy sequences (e.g. scrambled Hammersley) have a better discrepancy.
    - Reference: Halton 1964: Algorithm 247: Radical-inverse quasi-random point sequence, ACM, p. 701.
      adds scrambling to the Halton search; much better in high dimension and rarely worse
      than the original Halton search.
    - About Latin Hypercube Sampling (LHS):
      Though partially incremental versions exist, this implementation needs the budget in advance.
      This can be great in terms of discrepancy when the budget is not very high.
    '''
    one_shot: bool
    def __init__(self, *, sampler: str = 'Halton', scrambled: bool = False, middle_point: bool = False, opposition_mode: tp.Optional[str] = None, cauchy: bool = False, autorescale: tp.Union[bool, str] = False, scale: float = 1.0, rescaled: bool = False, recommendation_rule: str = 'pessimistic') -> None: ...

MetaRecentering: Incomplete
MetaTuneRecentering: Incomplete
HullAvgMetaTuneRecentering: Incomplete
HullAvgMetaRecentering: Incomplete
AvgMetaRecenteringNoHull: Incomplete
HaltonSearch: Incomplete
HaltonSearchPlusMiddlePoint: Incomplete
LargeHaltonSearch: Incomplete
ScrHaltonSearch: Incomplete
ScrHaltonSearchPlusMiddlePoint: Incomplete
HammersleySearch: Incomplete
HammersleySearchPlusMiddlePoint: Incomplete
ScrHammersleySearchPlusMiddlePoint: Incomplete
ScrHammersleySearch: Incomplete
QOScrHammersleySearch: Incomplete
OScrHammersleySearch: Incomplete
CauchyScrHammersleySearch: Incomplete
LHSSearch: Incomplete
CauchyLHSSearch: Incomplete
