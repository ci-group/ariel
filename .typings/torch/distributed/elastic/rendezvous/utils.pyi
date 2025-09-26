import weakref
from datetime import timedelta
from threading import Event, Thread
from typing import Any, Callable

__all__ = ['parse_rendezvous_endpoint']

def parse_rendezvous_endpoint(endpoint: str | None, default_port: int) -> tuple[str, int]:
    """Extract the hostname and the port number from a rendezvous endpoint.

    Args:
        endpoint:
            A string in format <hostname>[:<port>].
        default_port:
            The port number to use if the endpoint does not include one.

    Returns:
        A tuple of hostname and port number.
    """

class _PeriodicTimer:
    """Represent a timer that periodically runs a specified function.

    Args:
        interval:
            The interval, in seconds, between each run.
        function:
            The function to run.
    """
    class _Context:
        interval: float
        function: Callable[..., None]
        args: tuple[Any, ...]
        kwargs: dict[str, Any]
        stop_event: Event
    _name: str | None
    _thread: Thread | None
    _finalizer: weakref.finalize | None
    _ctx: _Context
    def __init__(self, interval: timedelta, function: Callable[..., None], *args: Any, **kwargs: Any) -> None: ...
    @property
    def name(self) -> str | None:
        """Get the name of the timer."""
    def set_name(self, name: str) -> None:
        """Set the name of the timer.

        The specified name will be assigned to the background thread and serves
        for debugging and troubleshooting purposes.
        """
    def start(self) -> None:
        """Start the timer."""
    def cancel(self) -> None:
        """Stop the timer at the next opportunity."""
    @staticmethod
    def _run(ctx) -> None: ...
    @staticmethod
    def _stop_thread(thread, stop_event) -> None: ...
