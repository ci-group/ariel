import nevergrad.common.typing as tp
import threading
from . import base as base
from .base import IntOrParameter as IntOrParameter
from _typeshed import Incomplete
from nevergrad.common.errors import NevergradError as NevergradError
from nevergrad.parametrization import parameter as p

class StopOptimizerThread(Exception): ...
class TooManyAskError(NevergradError): ...

class _MessagingThread(threading.Thread):
    """Thread that runs a function taking another function as input. Each call of the inner function
    adds the point given by the algorithm into the ask queue and then blocks until the main thread sends
    the result back into the tell queue.

    Note
    ----
    This thread must be overlaid into another MessagingThread  because:
    - the threading part should hold no reference from outside (otherwise the destructors may wait each other)
    - the destructor cannot be implemented, hence there is no way to stop the thread automatically
    """
    messages_ask: tp.Any
    messages_tell: tp.Any
    call_count: int
    error: tp.Optional[Exception]
    _caller: Incomplete
    _args: Incomplete
    _kwargs: Incomplete
    output: tp.Optional[tp.Any]
    def __init__(self, caller: tp.Callable[..., tp.Any], *args: tp.Any, **kwargs: tp.Any) -> None: ...
    def run(self) -> None:
        '''Starts the thread and run the "caller" function argument on
        the fake callable, which posts messages and awaits for their answers.
        '''
    def _fake_callable(self, *args: tp.Any) -> tp.Any:
        """
        Puts a new point into the ask queue to be evaluated on the
        main thread and blocks on get from tell queue until point
        is evaluated on main thread and placed into tell queue when
        it is then returned to the caller.
        """
    def stop(self) -> None:
        """Notifies the thread that it must stop"""

class MessagingThread:
    """Encapsulate the inner thread, so that kill order is automatically called at deletion."""
    _thread: Incomplete
    def __init__(self, caller: tp.Callable[..., tp.Any], *args: tp.Any, **kwargs: tp.Any) -> None: ...
    def is_alive(self) -> bool: ...
    @property
    def output(self) -> tp.Any: ...
    @property
    def error(self) -> tp.Optional[Exception]: ...
    @property
    def messages_tell(self) -> tp.Any: ...
    @property
    def messages_ask(self) -> tp.Any: ...
    def stop(self) -> None: ...
    def __del__(self) -> None: ...

class RecastOptimizer(base.Optimizer):
    '''Base class for ask and tell optimizer derived from implementations with no ask and tell interface.
    The underlying optimizer implementation is a function which is supposed to call directly the function
    to optimize. It is tricked into optimizing a "fake" function in a thread:
    - calls to the fake functions are returned by the "ask()" interface
    - return values of the fake functions are provided to the thread when calling "tell(x, value)"

    Note
    ----
    These implementations are not necessarily robust. More specifically, one cannot "tell" any
    point which was not "asked" before.

    An optimization is performed by a third-party library in a background thread. This communicates
    with the main thread using two queue objects. Specifically:

        messages_ask is filled by the background thread with a candidate (or batch of candidates)
        it wants evaluated for, or None if the background thread is over, or an Exception
        which needs to be raised to the user.

        messages_tell supplies the background thread with a value to return from the fake function.
        A value of None means the background thread is no longer relevant and should exit.
    '''
    recast: bool
    _messaging_thread: tp.Optional[MessagingThread]
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...
    def get_optimization_function(self) -> tp.Callable[[tp.Callable[..., tp.Any]], tp.Optional[tp.ArrayLike]]:
        """Return an optimization procedure function (taking a function to optimize as input)

        Note
        ----
        This optimization procedure must be a function or an object which is completely
        independent from self, otherwise deletion of the optimizer may hang indefinitely.
        """
    def _check_error(self) -> None: ...
    def _post_loss(self, candidate: p.Parameter, loss: float) -> tp.Loss:
        """
        Posts the value, and the thread will deal with it.
        """
    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: float) -> None: ...
    def _internal_provide_recommendation(self) -> tp.Optional[tp.ArrayLike]:
        """Returns the underlying optimizer output if provided (ie if the optimizer did finish)
        else the best pessimistic point.
        """
    def __del__(self) -> None: ...

class SequentialRecastOptimizer(RecastOptimizer):
    """Recast Optimizer which cannot deal with parallelization

    There can only be one worker. Each ask must be followed by
    a tell.

    A simple usage is that you have a library which can minimize
    a function which returns a scalar.
    Just make an optimizer inheriting from this class, and inplement
    get_optimization_function to return a callable which runs the
    optimization, taking the objective as its only parameter. The
    callable must not have any references to the optimizer itself.
    (This avoids a reference cycle between the background thread and
    the optimizer, aiding cleanup.) It can have a weakref though.

    If you want your optimizer instance to be picklable, we have to
    store every candidate during optimization, which may use a lot
    of memory. This lets us replay the optimization when
    unpickling. We only do this if you ask for it. To enable:
        - The optimization must be reproducible, asking for the same
          candidates every time. If you need a seed from nevergrad's
          generator, you can't necessarily generate this again after
          unpickling. One solution is to store it in self, if it is
          not there yet, in the body of get_optimization_function.

          As in general in nevergrad, do not set the seed from the
          RNG in your own __init__ because it will cause surprises
          to anyone re-seeding your parametrization after init.
        - The user must call enable_pickling() after initializing
          the optimizer instance.
    """
    no_parallelization: bool
    _enable_pickling: bool
    replay_archive_tell: tp.List[p.Parameter]
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int], num_workers: int = 1) -> None: ...
    def enable_pickling(self) -> None:
        """Make the optimizer store its history of tells, so
        that it can be serialized.
        """
    _messaging_thread: Incomplete
    def _internal_ask_candidate(self) -> p.Parameter:
        '''Reads messages from the thread in which the underlying optimization function is running
        New messages are sent as "ask".
        '''
    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        '''Returns value for a point which was "asked"
        (none asked point cannot be "tell")
        '''
    def __getstate__(self): ...
    def __setstate__(self, state) -> None: ...

class BatchRecastOptimizer(RecastOptimizer):
    """Recast optimizer where points to evaluate are provided in batches
    and stored by the optimizer to be asked and told on. The fake_callable
    is only brought into action every 'batch size' number of asks and tells
    instead of every ask and tell. This opens up the optimizer to
    parallelism.

    Note
    ----
    You have to complete a batch before you start a new one so parallelism
    is only possible within batches i.e. if a batch size is 100 and you have
    done 100 asks, you must do 100 tells before you ask again but you could do
    those 100 asks and tells in parallel. To find out if you can perform an ask
    at any given time call self.can_ask.
    """
    _current_batch: tp.List[p.Parameter]
    _batch_losses: tp.List[tp.Loss]
    _tell_counter: int
    batch_size: int
    indices: tp.Dict[str, int]
    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None: ...
    _messaging_thread: Incomplete
    def _internal_ask_candidate(self) -> p.Parameter:
        '''Reads messages from the thread in which the underlying optimization function is running
        New messages are sent as "ask".
        '''
    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        '''Returns value for a point which was "asked"
        (none asked point cannot be "tell")
        '''
    def minimize(self, objective_function: tp.Callable[..., tp.Loss], executor: tp.Optional[tp.ExecutorLike] = None, batch_mode: bool = False, verbosity: int = 0, constraint_violation: tp.Any = None, max_time: tp.Optional[float] = None) -> p.Parameter: ...
    def can_ask(self) -> bool:
        """Returns whether the optimizer is able to perform another ask,
        either because there are points left in the current batch to ask
        or you are ready for a new batch (You have asked and told on every
        point in the last batch.)
        """
