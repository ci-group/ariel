import nevergrad.common.typing as tp
import numpy as np
from _typeshed import Incomplete
from nevergrad.common.decorators import Registry as Registry
from numpy.random import RandomState

samplers: Registry[tp.Type['Sampler']]

def _get_first_primes(num: int) -> np.ndarray:
    """Computes the first num primes"""

class Sampler:
    random_state: Incomplete
    dimension: Incomplete
    budget: Incomplete
    index: int
    def __init__(self, dimension: int, budget: tp.Optional[int] = None, random_state: tp.Optional[RandomState] = None) -> None: ...
    def _internal_sampler(self) -> tp.ArrayLike: ...
    def __call__(self) -> tp.ArrayLike: ...
    def __iter__(self) -> tp.Iterator[tp.ArrayLike]: ...
    def reinitialize(self) -> None: ...
    def draw(self) -> None:
        """Simple ASCII drawing of the sampling pattern (for testing/visualization purpose only)"""

class LHSSampler(Sampler):
    permutations: Incomplete
    seed: Incomplete
    randg: Incomplete
    def __init__(self, dimension: int, budget: int, scrambling: bool = False, random_state: tp.Optional[RandomState] = None) -> None: ...
    def reinitialize(self) -> None: ...
    def _internal_sampler(self) -> tp.ArrayLike: ...

class RandomSampler(Sampler):
    def __init__(self, dimension: int, budget: int, scrambling: bool = False, random_state: tp.Optional[RandomState] = None) -> None: ...
    def _internal_sampler(self) -> tp.ArrayLike: ...

class HaltonPermutationGenerator:
    """Provides a light-memory access to a possibly huge list of permutations
    (at the cost of being slightly slower)
    """
    dimension: Incomplete
    scrambling: Incomplete
    primes: Incomplete
    seed: Incomplete
    fulllist: Incomplete
    def __init__(self, dimension: int, scrambling: bool = False, random_state: tp.Optional[RandomState] = None) -> None: ...
    def get_permutations_generator(self) -> tp.Iterator[tp.ArrayLike]: ...

class HaltonSampler(Sampler):
    permgen: Incomplete
    def __init__(self, dimension: int, budget: tp.Optional[int] = None, scrambling: bool = False, random_state: tp.Optional[RandomState] = None) -> None: ...
    def vdc(self, n: int, permut: tp.List[int]) -> float: ...
    def _internal_sampler(self) -> tp.ArrayLike: ...

class HammersleySampler(HaltonSampler):
    def __init__(self, dimension: int, budget: tp.Optional[int] = None, scrambling: bool = False, random_state: tp.Optional[RandomState] = None) -> None: ...
    def _internal_sampler(self) -> tp.ArrayLike: ...

class Rescaler:
    sample_mins: Incomplete
    sample_maxs: Incomplete
    epsilon: Incomplete
    def __init__(self, points: tp.Iterable[tp.ArrayLike]) -> None: ...
    def apply(self, point: tp.ArrayLike) -> np.ndarray: ...
