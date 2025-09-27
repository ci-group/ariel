from _typeshed import Incomplete
from queue import Queue
from torch._lazy.device_context import get_device_context as get_device_context

class ClosureHandler:
    def __init__(self) -> None: ...
    def run(self, closure) -> None:
        """Run closure function

        Args:
        closure: callable function to run
        """
    def __call__(self, closures) -> None: ...

class AsyncClosureHandler(ClosureHandler):
    """Handler for Asynchronous Step Closures
    Args:
        max_queue_size: The maximum length of the closure queue after which
        the training loop will block until closures are evaluated.
        By default, a reasonable limit of a maximum of 100 on the queue.
        This value can be set using the `XLA_MAX_ASYNC_QUEUE` environment
        variable.
    """
    _closure_queue: Queue
    _closure_exception: Queue
    _closure_lock: Incomplete
    _closure_event_loop_finished: Incomplete
    _closure_event_loop: Incomplete
    def __init__(self, max_queue_size: int = 100) -> None: ...
    def start_event_loop(self) -> None:
        """Start closure event loop if not started"""
    def run(self, closure) -> None: ...

def add_step_closure(closure, args=(), run_async: bool = False):
    """Adds a closure to the list of the ones to be run at the end of the step.
    Many times during model training there is the need to print/report (print to
    console, post to tensorboard, etc...) information which require the content of
    intermediary tensors to be inspected.
    Inspecting different tensors content in different points of the model code
    requires many executions and typically causes performance issues.
    Adding a step closure will ensure that it will be run after the barrier, when
    all the live tensors will be already materialized to device data.
    Live tensors which will include the ones captured by the closure arguments.
    So using `add_step_closure()` will ensure a single execution will be
    performed, even when multiple closures are queued, requiring multiple tensors
    to be inspected.
    Step closures will be run sequentially in the order they have been queued.
    Note that even though using this API the execution will be optimized, it is
    advised to throttle the printing/reporting events once every N steps.
    Args:
      closure (callable): The function to be called.
      args (tuple): The arguments to be passed to the closure.
      run_async: If True, run the closure asynchronously.
    """
def run_step_closures(): ...
