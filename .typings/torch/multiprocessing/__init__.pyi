from multiprocessing import *

__all__ = ['set_sharing_strategy', 'get_sharing_strategy', 'get_all_sharing_strategies', 'Array', 'AuthenticationError', 'Barrier', 'BoundedSemaphore', 'BufferTooShort', 'Condition', 'Event', 'JoinableQueue', 'Lock', 'Manager', 'Pipe', 'Pool', 'Process', 'ProcessError', 'Queue', 'RLock', 'RawArray', 'RawValue', 'Semaphore', 'SimpleQueue', 'TimeoutError', 'Value', 'active_children', 'allow_connection_pickling', 'cpu_count', 'current_process', 'freeze_support', 'get_all_start_methods', 'get_context', 'get_logger', 'get_start_method', 'log_to_stderr', 'parent_process', 'reducer', 'set_executable', 'set_forkserver_preload', 'set_start_method']

def set_sharing_strategy(new_strategy) -> None:
    """Set the strategy for sharing CPU tensors.

    Args:
        new_strategy (str): Name of the selected strategy. Should be one of
            the values returned by :func:`get_all_sharing_strategies()`.
    """
def get_sharing_strategy():
    """Return the current strategy for sharing CPU tensors."""
def get_all_sharing_strategies():
    """Return a set of sharing strategies supported on a current system."""

# Names in __all__ with no definition:
#   Array
#   AuthenticationError
#   Barrier
#   BoundedSemaphore
#   BufferTooShort
#   Condition
#   Event
#   JoinableQueue
#   Lock
#   Manager
#   Pipe
#   Pool
#   Process
#   ProcessError
#   Queue
#   RLock
#   RawArray
#   RawValue
#   Semaphore
#   SimpleQueue
#   TimeoutError
#   Value
#   active_children
#   allow_connection_pickling
#   cpu_count
#   current_process
#   freeze_support
#   get_all_start_methods
#   get_context
#   get_logger
#   get_start_method
#   log_to_stderr
#   parent_process
#   reducer
#   set_executable
#   set_forkserver_preload
#   set_start_method
