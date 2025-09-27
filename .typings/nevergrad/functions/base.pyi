import nevergrad.common.typing as tp
import numpy as np
from _typeshed import Incomplete
from nevergrad.common import errors as errors
from nevergrad.parametrization import parameter as p

EF = tp.TypeVar('EF', bound='ExperimentFunction')
ME = tp.TypeVar('ME', bound='MultiExperiment')

def _reset_copy(obj: p.Parameter) -> p.Parameter:
    """Copy a parameter and resets its random state to obtain variability"""

class ExperimentFunction:
    '''Combines a function and its parametrization for running experiments (see benchmark subpackage)

    Parameters
    ----------
    function: callable
        the callable to convert
    parametrization: Parameter
        the parametrization of the function
    Notes
    -----
    - you can redefine custom "evaluation_function" and "compute_pseudotime" for custom behaviors in experiments
    - the bool/int/str/float init arguments are added as descriptors for the experiment which will serve in
      definining test cases. You can add more through "add_descriptors".
    - Makes sure you the "copy()" methods works (provides a new copy of the function *and* its parametrization)
      if you subclass ExperimentFunction since it is intensively used in benchmarks.
      By default, this will create a new instance using the same init arguments as your current instance
      (they were recorded through "__new__"\'s magic) and apply the additional descriptors you may have added,
      as well as propagate the new parametrization *if it has a different name as the current one*.
    '''
    def __new__(cls, *args: tp.Any, **kwargs: tp.Any) -> EF:
        """Identifies initialization parameters during initialization and store them"""
    _auto_init: tp.Dict[str, tp.Any]
    _descriptors: tp.Dict[str, tp.Any]
    _parametrization: p.Parameter
    multiobjective_upper_bounds: tp.Optional[np.ndarray]
    __function: Incomplete
    def __init__(self, function: tp.Callable[..., tp.Loss], parametrization: p.Parameter) -> None: ...
    @property
    def dimension(self) -> int: ...
    @property
    def parametrization(self) -> p.Parameter: ...
    @parametrization.setter
    def parametrization(self, parametrization: p.Parameter) -> None: ...
    @property
    def function(self) -> tp.Callable[..., tp.Loss]: ...
    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Loss:
        """Call the function directly (equivalent to parametrized_function.function(*args, **kwargs))"""
    @property
    def descriptors(self) -> tp.Dict[str, tp.Any]:
        """Description of the function parameterization, as a dict. This base class implementation provides function_class,
        noise_level, transform and dimension
        """
    def add_descriptors(self, **kwargs: tp.Optional[tp.Hashable]) -> None: ...
    def __repr__(self) -> str:
        """Shows the function name and its summary"""
    def equivalent_to(self, other: tp.Any) -> bool:
        """Check that two instances where initialized with same settings.
        This is not meant to be used to check if functions are exactly equal
        (initialization may hold some randomness)
        This is only useful for unit testing.
        (may need to be overloaded to make faster if tests are getting slow)
        """
    def _internal_copy(self) -> EF:
        '''This is "black magic" which creates a new instance using the same init parameters
        that you provided and which were recorded through the __new__ method of ExperimentFunction
        '''
    def copy(self) -> EF:
        """Provides a new equivalent instance of the class, possibly with
        different random initialization, to provide different equivalent test cases
        when using different seeds.
        This also checks that parametrization and descriptors are correct.
        You should preferably override _internal_copy
        """
    def compute_pseudotime(self, input_parameter: tp.ArgsKwargs, loss: tp.Loss) -> float:
        """Computes a pseudotime used during benchmarks for mocking parallelization in a reproducible way.
        By default, each call takes 1 unit of pseudotime, but this can be modified by overriding this
        function and the pseudo time can be a function of the function inputs and output.

        Note: This replaces get_postponing_delay which has been aggressively deprecated

        Parameters
        ----------
        input_parameter: Any
            the input that was provided to the actual function
        value: float
            the output of the actual function

        Returns
        -------
        float
            the pseudo computation time of the call to the actual function
        """
    def evaluation_function(self, *recommendations: p.Parameter) -> float:
        """Provides the evaluation crieterion for the experiment.
        In case of mono-objective, it defers to evaluation_function
        Otherwise, it uses the hypervolume.
        This function can be overriden to provide custom behaviors.

        Parameters
        ----------
        *pareto: Parameter
            pareto front provided by the optimizer
        """

def update_leaderboard(identifier: str, loss: float, array: np.ndarray, verbose: bool = True) -> None:
    """Handy function for storing best results for challenging functions (eg.: Photonics)
    The best results are kept in a file that is automatically updated with new data.
    This may require installing nevergrad in dev mode.

    Parameters
    ----------
    identifier: str
        the identifier of the problem
    loss: float
        the new loss, if better than the one in the file, the file will be updated
    array: np.ndarray
        the array corresponding to the loss
    verbose: bool
        whether to also print a message if the leaderboard was updated
    """

class ArrayExperimentFunction(ExperimentFunction):
    """Combines a function and its parametrization for running experiments (see benchmark subpackage).
    Extends ExperimentFunction, in the special case of an array, by allowing the creation of symmetries
    of a single function. We can create ArrayExperimentFunction(callable, symmetry=i) for i in range(0, 2**d)
    when the callable works on R^d.
    Works only if there are no constraints.

    Parameters
    ----------
    function: callable
        the callable to convert
    parametrization: Parameter
        the parametrization of the function
    symmetry: int
        number parametrizing how we symmetrize the function.
    """
    _inner_function: Incomplete
    _function: Incomplete
    threshold_coefficients: Incomplete
    slope_coefficients: Incomplete
    def __init__(self, function: tp.Callable[..., tp.Loss], parametrization: p.Parameter, symmetry: int = 0) -> None:
        '''Adds a "symmetry" parameter, which allows the creation of many symmetries of a given function.

        symmetry: an int, 0 by default.
        if not zero, a symmetrization is applied to the input; each of the 2^d possible values
        for symmetry % 2^d gives one different function.
        Makes sense if and only if (1) the input is a single ndarray (2) the domains are symmetric.'''
    def symmetrized_function(self, x: np.ndarray) -> tp.Loss: ...

class MultiExperiment(ExperimentFunction):
    """Pack several mono-objective experiments into a multiobjective experiment


    Parameters
    ----------
    experiments: iterable of ExperimentFunction

    Notes
    -----
    - packing of multiobjective experiments is not supported.
    - parametrization must match between all functions (only their name is checked as initialization)
    - there is no descriptor for the packed functions, except the name (concatenetion of packed function names).
    """
    multiobjective_upper_bounds: Incomplete
    _experiments: Incomplete
    def __init__(self, experiments: tp.Iterable[ExperimentFunction], upper_bounds: tp.ArrayLike) -> None: ...
    def _multi_func(self, *args: tp.Any, **kwargs: tp.Any) -> np.ndarray: ...
    def _internal_copy(self) -> MultiExperiment: ...
