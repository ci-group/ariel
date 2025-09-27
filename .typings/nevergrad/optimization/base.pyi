import nevergrad.common.typing as tp
import numpy as np
from . import multiobjective as mobj, utils as utils
from _typeshed import Incomplete
from nevergrad.common.decorators import Registry as Registry
from nevergrad.parametrization import parameter as p
from pathlib import Path

OptCls: Incomplete
registry: Registry[OptCls]
_OptimCallBack: Incomplete
X = tp.TypeVar('X', bound='Optimizer')
Y = tp.TypeVar('Y')
IntOrParameter: Incomplete
_PruningCallable: Incomplete

def _loss(param: p.Parameter) -> float:
    """Returns the loss if available, or inf otherwise.
    Used to simplify handling of losses
    """
def load(cls, filepath: tp.PathLike) -> X:
    """Loads a pickle file and checks that it contains an optimizer.
    The optimizer class is not always fully reliable though (e.g.: optimizer families) so the user is responsible for it.
    """

class Optimizer:
    """Algorithm framework with 3 main functions:

    - :code:`ask()` which provides a candidate on which to evaluate the function to optimize.
    - :code:`tell(candidate, loss)` which lets you provide the loss associated to points.
    - :code:`provide_recommendation()` which provides the best final candidate.

    Typically, one would call :code:`ask()` num_workers times, evaluate the
    function on these num_workers points in parallel, update with the fitness value when the
    evaluations is finished, and iterate until the budget is over. At the very end,
    one would call provide_recommendation for the estimated optimum.

    This class is abstract, it provides internal equivalents for the 3 main functions,
    among which at least :code:`_internal_ask_candidate` has to be overridden.

    Each optimizer instance should be used only once, with the initial provided budget

    Parameters
    ----------
    parametrization: int or Parameter
        either the dimension of the optimization space, or its parametrization
    budget: int/None
        number of allowed evaluations
    num_workers: int
        number of evaluations which will be run in parallel at once
    """
    recast: bool
    one_shot: bool
    no_parallelization: bool
    num_workers: Incomplete
    budget: Incomplete
    optim_curve: tp.List[tp.Any]
    skip_constraints: bool
    _constraints_manager: Incomplete
    _penalize_cheap_violations: bool
    parametrization: Incomplete
    name: Incomplete
    archive: utils.Archive[utils.MultiValue]
    current_bests: Incomplete
    pruning: tp.Optional[_PruningCallable]
    _MULTIOBJECTIVE_AUTO_BOUND: Incomplete
    _hypervolume_pareto: tp.Optional[mobj.HypervolumePareto]
    _asked: tp.Set[str]
    _num_objectives: int
    _suggestions: tp.Deque[p.Parameter]
    _num_ask: int
    _num_tell: int
    _num_tell_not_asked: int
    _callbacks: tp.Dict[str, tp.List[tp.Any]]
    _running_jobs: tp.List[tp.Tuple[p.Parameter, tp.JobLike[tp.Loss]]]
    _finished_jobs: tp.Deque[tp.Tuple[p.Parameter, tp.JobLike[tp.Loss]]]
    _sent_warnings: tp.Set[tp.Any]
    _no_hypervolume: bool
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...
    def _warn(self, msg: str, e: tp.Any) -> None:
        """Warns only once per warning type"""
    @property
    def _rng(self) -> np.random.RandomState:
        """np.random.RandomState: parametrization random state the optimizer must pull from.
        It can be seeded or updated directly on the parametrization instance (`optimizer.parametrization.random_state`)
        """
    @property
    def dimension(self) -> int:
        """int: Dimension of the optimization space."""
    @property
    def num_objectives(self) -> int:
        """Provides 0 if the number is not known yet, else the number of objectives
        to optimize upon.
        """
    @num_objectives.setter
    def num_objectives(self, num: int) -> None: ...
    def _num_objectives_set_callback(self) -> None:
        """Callback for when num objectives is first known"""
    @property
    def num_ask(self) -> int:
        """int: Number of time the `ask` method was called."""
    @property
    def num_tell(self) -> int:
        """int: Number of time the `tell` method was called."""
    @property
    def num_tell_not_asked(self) -> int:
        """int: Number of time the :code:`tell` method was called on candidates that were not asked for by the optimizer
        (or were suggested).
        """
    def pareto_front(self, size: tp.Optional[int] = None, subset: str = 'random', subset_tentatives: int = 12) -> tp.List[p.Parameter]:
        '''Pareto front, as a list of Parameter. The losses can be accessed through
        parameter.losses

        Parameters
        ------------
        size:  int (optional)
            if provided, selects a subset of the full pareto front with the given maximum size
        subset: str
            method for selecting the subset ("random, "loss-covering", "domain-covering", "hypervolume")
        subset_tentatives: int
            number of random tentatives for finding a better subset

        Returns
        --------
        list
            the list of Parameter of the pareto front

        Note
        ----
        During non-multiobjective optimization, this returns the current pessimistic best
        '''
    def dump(self, filepath: tp.Union[str, Path]) -> None:
        """Pickles the optimizer into a file."""
    @classmethod
    def load(cls, filepath: tp.Union[str, Path]) -> X:
        """Loads a pickle and checks that the class is correct."""
    def __repr__(self) -> str: ...
    def register_callback(self, name: str, callback: _OptimCallBack) -> None:
        """Add a callback method called either when `tell` or `ask` are called, with the same
        arguments (including the optimizer / self). This can be useful for custom logging.

        Parameters
        ----------
        name: str
            name of the method to register the callback for (either :code:`ask` or :code:`tell`)
        callback: callable
            a callable taking the same parameters as the method it is registered upon (including self)
        """
    def remove_all_callbacks(self) -> None:
        """Removes all registered callables"""
    def suggest(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        """Suggests a new point to ask.
        It will be asked at the next call (last in first out).

        Parameters
        ----------
        *args: Any
            positional arguments matching the parametrization pattern.
        *kwargs: Any
            keyword arguments matching the parametrization pattern.

        Note
        ----
        - This relies on optmizers implementing a way to deal with unasked candidate.
          Some optimizers may not support it and will raise a :code:`TellNotAskedNotSupportedError`
          at :code:`tell` time.
        - LIFO is used so as to be able to suggest and ask straightaway, as an alternative to
          creating a new candidate with :code:`optimizer.parametrization.spawn_child(new_value)`
        """
    def tell(self, candidate: p.Parameter, loss: tp.Loss, constraint_violation: tp.Optional[tp.Loss] = None, penalty_style: tp.Optional[tp.ArrayLike] = None) -> None:
        """Provides the optimizer with the evaluation of a fitness value for a candidate.

        Parameters
        ----------
        x: np.ndarray
            point where the function was evaluated
        loss: float/list/np.ndarray
            loss of the function (or multi-objective function
        constraint_violation: float/list/np.ndarray/None
            constraint violation (> 0 means that this is not correct)
        penalty_style: ArrayLike/None
            to be read as [a,b,c,d,e,f]
            with cv the constraint violation vector (above):
            penalty = (a + sum(|loss|)) * (f+num_tell)**e * (b * sum(cv**c)) ** d
            default: [1e5, 1., .5, 1., .5, 1.]

        Note
        ----
        The candidate should generally be one provided by :code:`ask()`, but can be also
        a non-asked candidate. To create a p.Parameter instance from args and kwargs,
        you can use :code:`candidate = optimizer.parametrization.spawn_child(new_value=your_value)`:

        - for an :code:`Array(shape(2,))`: :code:`optimizer.parametrization.spawn_child(new_value=[12, 12])`

        - for an :code:`Instrumentation`: :code:`optimizer.parametrization.spawn_child(new_value=(args, kwargs))`

        Alternatively, you can provide a suggestion with :code:`optimizer.suggest(*args, **kwargs)`, the next :code:`ask`
        will use this suggestion.
        """
    def _preprocess_multiobjective(self, candidate: p.Parameter) -> tp.FloatLoss: ...
    def _update_archive_and_bests(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None: ...
    def ask(self) -> p.Parameter:
        """Provides a point to explore.
        This function can be called multiple times to explore several points in parallel

        Returns
        -------
        p.Parameter:
            The candidate to try on the objective function. :code:`p.Parameter` have field :code:`args` and :code:`kwargs`
            which can be directly used on the function (:code:`objective_function(*candidate.args, **candidate.kwargs)`).
        """
    def provide_recommendation(self) -> p.Parameter:
        """Provides the best point to use as a minimum, given the budget that was used

        Returns
        -------
        p.Parameter
            The candidate with minimal value. p.Parameters have field :code:`args` and :code:`kwargs` which can be directly used
            on the function (:code:`objective_function(*candidate.args, **candidate.kwargs)`).
        """
    def recommend(self) -> p.Parameter:
        """Provides the best candidate to use as a minimum, given the budget that was used.

        Returns
        -------
        p.Parameter
            The candidate with minimal loss. :code:`p.Parameters` have field :code:`args` and :code:`kwargs` which can be directly used
            on the function (:code:`objective_function(*candidate.args, **candidate.kwargs)`).
        """
    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        '''Called whenever calling :code:`tell` on a candidate that was not "asked".
        Defaults to the standard tell pipeline.
        '''
    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        '''Called whenever calling :code:`tell` on a candidate that was "asked".'''
    def _internal_ask_candidate(self) -> p.Parameter: ...
    def _internal_tell(self, x: tp.ArrayLike, loss: tp.FloatLoss) -> None: ...
    def _internal_ask(self) -> tp.ArrayLike: ...
    def _internal_provide_recommendation(self) -> tp.Optional[tp.ArrayLike]:
        """Override to provide a recommendation in standardized space"""
    def enable_pickling(self) -> None:
        """
        Some optimizers are only optionally picklable, because picklability
        requires saving the whole history which would be a waste of memory
        in general. To tell an optimizer to be picklable, call this function
        before any asks.

        In this base class, the function is a no-op, but it is overridden
        in some optimizers.
        """
    def minimize(self, objective_function: tp.Callable[..., tp.Loss], executor: tp.Optional[tp.ExecutorLike] = None, batch_mode: bool = False, verbosity: int = 0, constraint_violation: tp.Any = None, max_time: tp.Optional[float] = None) -> p.Parameter:
        """Optimization (minimization) procedure

        Parameters
        ----------
        objective_function: callable
            A callable to optimize (minimize)
        executor: Executor
            An executor object, with method :code:`submit(callable, *args, **kwargs)` and returning a Future-like object
            with methods :code:`done() -> bool` and :code:`result() -> float`. The executor role is to dispatch the execution of
            the jobs locally/on a cluster/with multithreading depending on the implementation.
            Eg: :code:`concurrent.futures.ProcessPoolExecutor`
        batch_mode: bool
            when :code:`num_workers = n > 1`, whether jobs are executed by batch (:code:`n` function evaluations are launched,
            we wait for all results and relaunch n evals) or not (whenever an evaluation is finished, we launch
            another one)
        verbosity: int
            print information about the optimization (0: None, 1: fitness values, 2: fitness values and recommendation)
        constraint_violation: list of functions or None
            each function in the list returns >0 for a violated constraint.

        Returns
        -------
        ng.p.Parameter
            The candidate with minimal value. :code:`ng.p.Parameters` have field :code:`args` and :code:`kwargs` which can
            be directly used on the function (:code:`objective_function(*candidate.args, **candidate.kwargs)`).

        Note
        ----
        for evaluation purpose and with the current implementation, it is better to use batch_mode=True
        """
    def _info(self) -> tp.Dict[str, tp.Any]:
        """Easy access to debug/benchmark info"""

def addCompare(optimizer: Optimizer) -> None: ...

class ConfiguredOptimizer:
    """Creates optimizer-like instances with configuration.

    Parameters
    ----------
    OptimizerClass: type
        class of the optimizer to configure, or another ConfiguredOptimizer (config will then be ignored
        except for the optimizer name/representation)
    config: dict
        dictionnary of all the configurations
    as_config: bool
        whether to provide all config as kwargs to the optimizer instantiation (default, see ConfiguredCMA for an example),
        or through a config kwarg referencing self. (if True, see EvolutionStrategy for an example)

    Note
    ----
    This provides a default repr which can be bypassed through set_name
    """
    recast: bool
    one_shot: bool
    no_parallelization: bool
    _OptimizerClass: Incomplete
    _as_config: Incomplete
    _config: Incomplete
    name: Incomplete
    def __init__(self, OptimizerClass: OptCls, config: tp.Dict[str, tp.Any], as_config: bool = False) -> None: ...
    def config(self) -> tp.Dict[str, tp.Any]: ...
    def __call__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> Optimizer:
        """Creates an optimizer from the parametrization

        Parameters
        ----------
        instrumentation: int or Instrumentation
            either the dimension of the optimization space, or its instrumentation
        budget: int/None
            number of allowed evaluations
        num_workers: int
            number of evaluations which will be run in parallel at once
        """
    def __repr__(self) -> str: ...
    def set_name(self, name: str, register: bool = False) -> ConfiguredOptimizer:
        """Set a new representation for the instance"""
    def load(self, filepath: tp.Union[str, Path]) -> Optimizer:
        """Loads a pickle and checks that it is an Optimizer."""
    def __eq__(self, other: tp.Any) -> tp.Any: ...

def _constraint_solver(parameter: p.Parameter, budget: int) -> p.Parameter:
    """Runs a suboptimization to solve the parameter constraints"""
