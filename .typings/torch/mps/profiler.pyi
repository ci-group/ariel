import contextlib

__all__ = ['start', 'stop', 'profile', 'metal_capture', 'is_metal_capture_enabled', 'is_capturing_metal']

def start(mode: str = 'interval', wait_until_completed: bool = False) -> None:
    '''Start OS Signpost tracing from MPS backend.

    The generated OS Signposts could be recorded and viewed in
    XCode Instruments Logging tool.

    Args:
        mode(str): OS Signpost tracing mode could be "interval", "event",
            or both "interval,event".
            The interval mode traces the duration of execution of the operations,
            whereas event mode marks the completion of executions.
            See document `Recording Performance Data`_ for more info.
        wait_until_completed(bool): Waits until the MPS Stream complete
            executing each encoded GPU operation. This helps generating single
            dispatches on the trace\'s timeline.
            Note that enabling this option would affect the performance negatively.

    .. _Recording Performance Data:
       https://developer.apple.com/documentation/os/logging/recording_performance_data
    '''
def stop() -> None:
    """Stops generating OS Signpost tracing from MPS backend."""
@contextlib.contextmanager
def profile(mode: str = 'interval', wait_until_completed: bool = False):
    '''Context Manager to enabling generating OS Signpost tracing from MPS backend.

    Args:
        mode(str): OS Signpost tracing mode could be "interval", "event",
            or both "interval,event".
            The interval mode traces the duration of execution of the operations,
            whereas event mode marks the completion of executions.
            See document `Recording Performance Data`_ for more info.
        wait_until_completed(bool): Waits until the MPS Stream complete
            executing each encoded GPU operation. This helps generating single
            dispatches on the trace\'s timeline.
            Note that enabling this option would affect the performance negatively.

    .. _Recording Performance Data:
       https://developer.apple.com/documentation/os/logging/recording_performance_data
    '''
def is_metal_capture_enabled() -> bool:
    """Checks if `metal_capture` context manager is usable
    To enable metal capture, set MTL_CAPTURE_ENABLED envvar
    """
def is_capturing_metal() -> bool:
    """Cheks if metal capture is in progress"""
@contextlib.contextmanager
def metal_capture(fname: str):
    """Conext manager that enables capturing of Metal calls into gputrace"""
