import multiprocessing.pool
from .queue import SimpleQueue as SimpleQueue
from _typeshed import Incomplete

def clean_worker(*args, **kwargs) -> None: ...

class Pool(multiprocessing.pool.Pool):
    """Pool implementation which uses our version of SimpleQueue.

    This lets us pass tensors in shared memory across processes instead of
    serializing the underlying data.
    """
    _inqueue: Incomplete
    _outqueue: Incomplete
    _quick_put: Incomplete
    _quick_get: Incomplete
    def _setup_queues(self) -> None: ...
    def _repopulate_pool(self) -> None:
        """Increase the number of pool processes to the specified number.

        Bring the number of pool processes up to the specified number, for use after
        reaping workers which have exited.
        """
