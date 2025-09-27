import logging
import nevergrad.common.typing as tp
import numpy as np
from . import base as base
from _typeshed import Incomplete
from nevergrad.common import errors as errors
from nevergrad.parametrization import helpers as helpers, parameter as p
from pathlib import Path

global_logger: Incomplete

class OptimizationPrinter:
    """Printer to register as callback in an optimizer, for printing
    best point regularly.

    Parameters
    ----------
    print_interval_tells: int
        max number of evaluation before performing another print
    print_interval_seconds: float
        max number of seconds before performing another print
    """
    _print_interval_tells: Incomplete
    _print_interval_seconds: Incomplete
    _next_tell: Incomplete
    _next_time: Incomplete
    def __init__(self, print_interval_tells: int = 1, print_interval_seconds: float = 60.0) -> None: ...
    def __call__(self, optimizer: base.Optimizer, *args: tp.Any, **kwargs: tp.Any) -> None: ...

class OptimizationLogger:
    """Logger to register as callback in an optimizer, for Logging
    best point regularly.

    Parameters
    ----------
    logger:
        given logger that callback will use to log
    log_level:
        log level that logger will write to
    log_interval_tells: int
        max number of evaluation before performing another log
    log_interval_seconds:
        max number of seconds before performing another log
    """
    _logger: Incomplete
    _log_level: Incomplete
    _log_interval_tells: Incomplete
    _log_interval_seconds: Incomplete
    _next_tell: Incomplete
    _next_time: Incomplete
    def __init__(self, *, logger: logging.Logger = ..., log_level: int = ..., log_interval_tells: int = 1, log_interval_seconds: float = 60.0) -> None: ...
    def __call__(self, optimizer: base.Optimizer, *args: tp.Any, **kwargs: tp.Any) -> None: ...

class ParametersLogger:
    '''Logs parameter and run information throughout into a file during
    optimization.

    Parameters
    ----------
    filepath: str or pathlib.Path
        the path to dump data to
    append: bool
        whether to append the file (otherwise it replaces it)
    order: int
        order of the internal/model parameters to extract

    Example
    -------

    .. code-block:: python

        logger = ParametersLogger(filepath)
        optimizer.register_callback("tell",  logger)
        optimizer.minimize()
        list_of_dict_of_data = logger.load()

    Note
    ----
    Arrays are converted to lists
    '''
    _session: Incomplete
    _filepath: Incomplete
    _order: Incomplete
    def __init__(self, filepath: tp.Union[str, Path], append: bool = True, order: int = 1) -> None: ...
    def __call__(self, optimizer: base.Optimizer, candidate: p.Parameter, loss: tp.FloatLoss) -> None: ...
    def load(self) -> tp.List[tp.Dict[str, tp.Any]]:
        """Loads data from the log file"""
    def load_flattened(self, max_list_elements: int = 24) -> tp.List[tp.Dict[str, tp.Any]]:
        """Loads data from the log file, and splits lists (arrays) into multiple arguments

        Parameters
        ----------
        max_list_elements: int
            Maximum number of elements displayed from the array, each element is given a
            unique id of type list_name#i0_i1_...
        """
    def to_hiplot_experiment(self, max_list_elements: int = 24) -> tp.Any:
        """Converts the logs into an hiplot experiment for display.

        Parameters
        ----------
        max_list_elements: int
            maximum number of elements of list/arrays to export (only the first elements are extracted)

        Example
        -------
        .. code-block:: python

            exp = logs.to_hiplot_experiment()
            exp.display(force_full_width=True)

        Note
        ----
        - You can easily change the axes of the XY plot:
          :code:`exp.display_data(hip.Displays.XY).update({'axis_x': '0#0', 'axis_y': '0#1'})`
        - For more context about hiplot, check:

          - blogpost: https://ai.facebook.com/blog/hiplot-high-dimensional-interactive-plots-made-easy/
          - github repo: https://github.com/facebookresearch/hiplot
          - documentation: https://facebookresearch.github.io/hiplot/
        """

class OptimizerDump:
    """Dumps the optimizer to a pickle file at every call.

    Parameters
    ----------
    filepath: str or Path
        path to the pickle file
    """
    _filepath: Incomplete
    def __init__(self, filepath: tp.Union[str, Path]) -> None: ...
    def __call__(self, opt: base.Optimizer, *args: tp.Any, **kwargs: tp.Any) -> None: ...

class ProgressBar:
    """Progress bar to register as callback in an optimizer"""
    _progress_bar: tp.Any
    _current: int
    def __init__(self) -> None: ...
    def __call__(self, optimizer: base.Optimizer, *args: tp.Any, **kwargs: tp.Any) -> None: ...
    def __getstate__(self) -> tp.Dict[str, tp.Any]:
        """Used for pickling (tqdm is not picklable)"""

class EarlyStopping:
    '''Callback for stopping the :code:`minimize` method before the budget is
    fully used.

    Parameters
    ----------
    stopping_criterion: func(optimizer) -> bool
        function that takes the current optimizer as input and returns True
        if the minimization must be stopped

    Note
    ----
    This callback must be register on the "ask" method only.

    Example
    -------
    In the following code, the :code:`minimize` method will be stopped at the 4th "ask"

    >>> early_stopping = ng.callbacks.EarlyStopping(lambda opt: opt.num_ask > 3)
    >>> optimizer.register_callback("ask", early_stopping)
    >>> optimizer.minimize(_func, verbosity=2)

    A couple other options (equivalent in case of non-noisy optimization) for stopping
    if the loss is below 12:

    >>> early_stopping = ng.callbacks.EarlyStopping(lambda opt: opt.recommend().loss < 12)
    >>> early_stopping = ng.callbacks.EarlyStopping(lambda opt: opt.current_bests["minimum"].mean < 12)
    '''
    stopping_criterion: Incomplete
    def __init__(self, stopping_criterion: tp.Callable[[base.Optimizer], bool]) -> None: ...
    def __call__(self, optimizer: base.Optimizer, *args: tp.Any, **kwargs: tp.Any) -> None: ...
    @classmethod
    def timer(cls, max_duration: float) -> EarlyStopping:
        """Early stop when max_duration seconds has been reached (from the first ask)"""
    @classmethod
    def no_improvement_stopper(cls, tolerance_window: int) -> EarlyStopping:
        """Early stop when loss didn't reduce during tolerance_window asks"""

class _DurationCriterion:
    _start: Incomplete
    _max_duration: Incomplete
    def __init__(self, max_duration: float) -> None: ...
    def __call__(self, optimizer: base.Optimizer) -> bool: ...

class _LossImprovementToleranceCriterion:
    _tolerance_window: int
    _best_value: tp.Optional[np.ndarray]
    _tolerance_count: int
    def __init__(self, tolerance_window: int) -> None: ...
    def __call__(self, optimizer: base.Optimizer) -> bool: ...
