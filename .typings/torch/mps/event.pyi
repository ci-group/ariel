from _typeshed import Incomplete

class Event:
    """Wrapper around an MPS event.

    MPS events are synchronization markers that can be used to monitor the
    device's progress, to accurately measure timing, and to synchronize MPS streams.

    Args:
        enable_timing (bool, optional): indicates if the event should measure time
            (default: ``False``)
    """
    __eventId: Incomplete
    def __init__(self, enable_timing: bool = False) -> None: ...
    def __del__(self) -> None: ...
    def record(self) -> None:
        """Records the event in the default stream."""
    def wait(self) -> None:
        """Makes all future work submitted to the default stream wait for this event."""
    def query(self) -> bool:
        """Returns True if all work currently captured by event has completed."""
    def synchronize(self) -> None:
        """Waits until the completion of all work currently captured in this event.
        This prevents the CPU thread from proceeding until the event completes.
        """
    def elapsed_time(self, end_event: Event) -> float:
        """Returns the time elapsed in milliseconds after the event was
        recorded and before the end_event was recorded.
        """
