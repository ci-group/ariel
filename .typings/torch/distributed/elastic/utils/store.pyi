import torch
from contextlib import contextmanager
from typing import Callable

__all__ = ['store_timeout', 'get_all', 'synchronize', 'barrier']

DistStoreError = torch._C._DistStoreError

@contextmanager
def store_timeout(store, timeout: float):
    """
    This sets the timeout and then restores the old timeout when the context
    manager exits.

    Args:
        store: the store to set the timeout on
        timeout: the timeout to set
    """
def get_all(store, rank: int, prefix: str, world_size: int):
    '''
    Given a store and a prefix, the method goes through the array of keys
    of the following format: ``{prefix}{idx}``, where idx is in a range
    from 0 to size, and tries to retrieve the data.

    The Rank0 process waits at the end to make sure all other processes
    finished the procedure before exiting.

    Usage

    ::

     values = get_all(store, "torchelastic/data", 3)
     value1 = values[0]  # retrieves the data for key torchelastic/data0
     value2 = values[1]  # retrieves the data for key torchelastic/data1
     value3 = values[2]  # retrieves the data for key torchelastic/data2

    '''
def synchronize(store, data: bytes, rank: int, world_size: int, key_prefix: str, timeout: float = 300) -> list[bytes]:
    """
    Synchronizes ``world_size`` agents between each other using the underlying c10d store.
    The ``data`` will be available on each of the agents.

    Note: The data on the path is not deleted, as a result there can be stale data if
        you use the same key_prefix twice.

    Time complexity: O(N) per worker, O(N^2) globally.
    """
def barrier(store, world_size: int, key_prefix: str, barrier_timeout: float = 300, rank: int | None = None, rank_tracing_decoder: Callable[[int], str] | None = None, trace_timeout: float = 10) -> None:
    """
    A global lock between agents. This will pause all workers until at least
    ``world_size`` workers respond.

    This uses a fast incrementing index to assign waiting ranks and a success
    flag set by the last worker.

    Time complexity: O(1) per worker, O(N) globally.

    Optionally, passing rank will enable tracing of missing ranks on timeouts.
    `rank_tracing_decoder` lambda arg can be used to convert rank data
    into a more meaningful information at an app level (e.g. hostname).

    Note: Since the data is not removed from the store, the barrier can be used
        once per unique ``key_prefix``.
    """
