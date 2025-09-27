import nevergrad as ng
import nevergrad.common.typing as tp
import numpy as np
from . import corefuncs as corefuncs, utils as utils
from .base import ExperimentFunction as ExperimentFunction
from _typeshed import Incomplete
from nevergrad.common import tools as tools

class ArtificialVariable:
    _dimension: Incomplete
    _transforms: tp.List[utils.Transform]
    rotation: Incomplete
    translation_factor: Incomplete
    num_blocks: Incomplete
    block_dimension: Incomplete
    only_index_transform: Incomplete
    hashing: Incomplete
    dimension: Incomplete
    random_state: Incomplete
    expo: Incomplete
    def __init__(self, dimension: int, num_blocks: int, block_dimension: int, translation_factor: float, rotation: bool, hashing: bool, only_index_transform: bool, random_state: np.random.RandomState, expo: float) -> None: ...
    def _initialize(self) -> None:
        """Delayed initialization of the transforms to avoid slowing down the instance creation
        (makes unit testing much faster).
        This functions creates the random transform used upon each block (translation + optional rotation).
        """
    def process(self, data: tp.ArrayLike, deterministic: bool = True) -> np.ndarray: ...
    def _short_repr(self) -> str: ...

class ArtificialFunction(ExperimentFunction):
    '''Artificial function object. This allows the creation of functions with different
    dimension and structure to be used for benchmarking in many different settings.

    Parameters
    ----------
    name: str
        name of the underlying function to use (like "sphere" for instance). If a wrong
        name is provided, an error is raised with all existing names.
    block_dimension: int
        the dimension on which the underlying function will be applied.
    num_blocks: int
        the number of blocks of size "block_dimension" on which the underlying function
        will be applied. The number of useful dimension is therefore num_blocks * core_dimension
    useless_variables: int
        the number of additional variables which have no impact on the core function.
        The full dimension of the function is therefore useless_variables + num_blocks * core_dimension
    noise_level: float
        noise level for the additive noise: noise_level * N(0, 1, size=1) * [f(x + N(0, 1, size=dim)) - f(x)]
    noise_dissymmetry: bool
        True if we dissymmetrize the model of noise
    rotation: bool
        whether the block space should be rotated (random rotation)
    hashing: bool
        whether the input data should be hashed. In this case, the function expects an array of size 1 with
        string as element.
    aggregator: str
        how to aggregate the multiple block outputs
    bounded: bool
        bound the search domain to [-5,5]

    Example
    -------
    >>> func = ArtificialFunction("sphere", 5, noise_level=.1)
    >>> x = [1, 2, 1, 0, .5]
    >>> func(x)  # returns a float
    >>> func(x)  # returns a different float since the function is noisy
    >>> func.oracle_call(x)   # returns a float
    >>> func.oracle_call(x)   # returns the same float (no noise for oracles + sphere function is deterministic)
    >>> func2 = ArtificialFunction("sphere", 5, noise_level=.1)
    >>> func2.oracle_call(x)   # returns a different float than before, because a random translation is applied

    Note
    ----
    - The full dimension of the function is available through the dimension attribute.
      Its value is useless_variables + num_blocks * block_dimension
    - The blocks are chosen with random sorted indices (blocks do not overlap)
    - A random translation is always applied to the function at initialization, so that
      instantiating twice the functions will give 2 different functions (unless you use
      seeding)
    - the noise formula is: noise_level * N(0, 1) * (f(x + N(0, 1)) - f(x))
    '''
    name: Incomplete
    expo: Incomplete
    translation_factor: Incomplete
    zero_pen: Incomplete
    constraint_violation: tp.ArrayLike
    _parameters: Incomplete
    _dimension: Incomplete
    _func: Incomplete
    transform_var: Incomplete
    _aggregator: tp.Callable[[tp.ArrayLike], float]
    def __init__(self, name: str, block_dimension: int, num_blocks: int = 1, useless_variables: int = 0, noise_level: float = 0, noise_dissymmetry: bool = False, rotation: bool = False, translation_factor: float = 1.0, hashing: bool = False, aggregator: str = 'max', split: bool = False, bounded: bool = False, expo: float = 1.0, zero_pen: bool = False) -> None: ...
    @property
    def dimension(self) -> int: ...
    @staticmethod
    def list_sorted_function_names() -> tp.List[str]:
        """Returns a sorted list of function names that can be used for the blocks"""
    def _transform(self, x: tp.ArrayLike) -> np.ndarray: ...
    def function_from_transform(self, x: np.ndarray) -> float:
        """Implements the call of the function.
        Under the hood, __call__ delegates to oracle_call + add some noise if noise_level > 0.
        """
    def evaluation_function(self, *recommendations: ng.p.Parameter) -> float:
        """Implements the call of the function.
        Under the hood, __call__ delegates to oracle_call + add some noise if noise_level > 0.
        """
    def noisy_function(self, *x: tp.ArrayLike) -> float: ...
    def compute_pseudotime(self, input_parameter: tp.ArgsKwargs, loss: tp.Loss) -> float:
        """Delay before returning results in steady state mode benchmarks (fake execution time)"""

def _noisy_call(x: np.ndarray, transf: tp.Callable[[np.ndarray], np.ndarray], func: tp.Callable[[np.ndarray], float], noise_level: float, noise_dissymmetry: bool, random_state: np.random.RandomState) -> float: ...

class FarOptimumFunction(ExperimentFunction):
    """Very simple 2D norm-1 function with optimal value at (x_optimum, 100)"""
    _optimum: Incomplete
    multiobjective_upper_bounds: Incomplete
    def __init__(self, independent_sigma: bool = True, mutable_sigma: bool = True, multiobjective: bool = False, recombination: str = 'crossover', optimum: tp.Tuple[int, int] = (80, 100)) -> None: ...
    def _multifunc(self, x: np.ndarray) -> np.ndarray: ...
    def _monofunc(self, x: np.ndarray) -> float: ...
    def evaluation_function(self, *recommendations: ng.p.Parameter) -> float: ...
    @classmethod
    def itercases(cls) -> tp.Iterator['FarOptimumFunction']: ...
