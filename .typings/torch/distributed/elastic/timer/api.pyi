import abc
import threading
from _typeshed import Incomplete
from contextlib import contextmanager
from typing import Any

__all__ = ['TimerRequest', 'TimerClient', 'RequestQueue', 'TimerServer', 'configure', 'expires']

class TimerRequest:
    '''
    Data object representing a countdown timer acquisition and release
    that is used between the ``TimerClient`` and ``TimerServer``.
    A negative ``expiration_time`` should be interpreted as a "release"
    request.

    .. note:: the type of ``worker_id`` is implementation specific.
              It is whatever the TimerServer and TimerClient implementations
              have on to uniquely identify a worker.
    '''
    __slots__: Incomplete
    worker_id: Incomplete
    scope_id: Incomplete
    expiration_time: Incomplete
    def __init__(self, worker_id: Any, scope_id: str, expiration_time: float) -> None: ...
    def __eq__(self, other): ...

class TimerClient(abc.ABC, metaclass=abc.ABCMeta):
    """
    Client library to acquire and release countdown timers by communicating
    with the TimerServer.
    """
    @abc.abstractmethod
    def acquire(self, scope_id: str, expiration_time: float) -> None:
        """
        Acquires a timer for the worker that holds this client object
        given the scope_id and expiration_time. Typically registers
        the timer with the TimerServer.
        """
    @abc.abstractmethod
    def release(self, scope_id: str):
        """
        Releases the timer for the ``scope_id`` on the worker this
        client represents. After this method is
        called, the countdown timer on the scope is no longer in effect.
        """

class RequestQueue(abc.ABC, metaclass=abc.ABCMeta):
    """
    Consumer queue holding timer acquisition/release requests
    """
    @abc.abstractmethod
    def size(self) -> int:
        """
        Returns the size of the queue at the time this method is called.
        Note that by the time ``get`` is called the size of the queue
        may have increased. The size of the queue should not decrease
        until the ``get`` method is called. That is, the following assertion
        should hold:

        size = q.size()
        res = q.get(size, timeout=0)
        assert size == len(res)

        -- or --

        size = q.size()
        res = q.get(size * 2, timeout=1)
        assert size <= len(res) <= size * 2
        """
    @abc.abstractmethod
    def get(self, size: int, timeout: float) -> list[TimerRequest]:
        """
        Gets up to ``size`` number of timer requests in a blocking fashion
        (no more than ``timeout`` seconds).
        """

class TimerServer(abc.ABC, metaclass=abc.ABCMeta):
    """
    Entity that monitors active timers and expires them
    in a timely fashion. This server is responsible for
    reaping workers that have expired timers.
    """
    _request_queue: Incomplete
    _max_interval: Incomplete
    _daemon: Incomplete
    _watchdog_thread: threading.Thread | None
    _stop_signaled: bool
    def __init__(self, request_queue: RequestQueue, max_interval: float, daemon: bool = True) -> None:
        """
        :param request_queue: Consumer ``RequestQueue``
        :param max_interval: max time (in seconds) to wait
                             for an item in the request_queue
        :param daemon: whether to run the watchdog thread as a daemon
        """
    @abc.abstractmethod
    def register_timers(self, timer_requests: list[TimerRequest]) -> None:
        """
        Processes the incoming timer requests and registers them with the server.
        The timer request can either be a acquire-timer or release-timer request.
        Timer requests with a negative expiration_time should be interpreted
        as a release-timer request.
        """
    @abc.abstractmethod
    def clear_timers(self, worker_ids: set[Any]) -> None:
        """
        Clears all timers for the given ``worker_ids``.
        """
    @abc.abstractmethod
    def get_expired_timers(self, deadline: float) -> dict[str, list[TimerRequest]]:
        """
        Returns all expired timers for each worker_id. An expired timer
        is a timer for which the expiration_time is less than or equal to
        the provided deadline.
        """
    @abc.abstractmethod
    def _reap_worker(self, worker_id: Any) -> bool:
        """
        Reaps the given worker. Returns True if the worker has been
        successfully reaped, False otherwise. If any uncaught exception
        is thrown from this method, the worker is considered reaped
        and all associated timers will be removed.
        """
    def _reap_worker_no_throw(self, worker_id: Any) -> bool:
        """
        Wraps ``_reap_worker(worker_id)``, if an uncaught exception is
        thrown, then it considers the worker as reaped.
        """
    def _watchdog_loop(self) -> None: ...
    def _run_watchdog(self) -> None: ...
    def _get_scopes(self, timer_requests): ...
    def start(self) -> None: ...
    def stop(self) -> None: ...

def configure(timer_client: TimerClient):
    """
    Configures a timer client. Must be called before using ``expires``.
    """
@contextmanager
def expires(after: float, scope: str | None = None, client: TimerClient | None = None):
    '''
    Acquires a countdown timer that expires in ``after`` seconds from now,
    unless the code-block that it wraps is finished within the timeframe.
    When the timer expires, this worker is eligible to be reaped. The
    exact meaning of "reaped" depends on the client implementation. In
    most cases, reaping means to terminate the worker process.
    Note that the worker is NOT guaranteed to be reaped at exactly
    ``time.now() + after``, but rather the worker is "eligible" for being
    reaped and the ``TimerServer`` that the client talks to will ultimately
    make the decision when and how to reap the workers with expired timers.

    Usage::

        torch.distributed.elastic.timer.configure(LocalTimerClient())
        with expires(after=10):
            torch.distributed.all_reduce(...)
    '''
