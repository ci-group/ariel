from _typeshed import Incomplete
from concurrent.futures._base import Future
from threading import Event
from typing import TextIO

__all__ = ['tail_logfile', 'TailLog']

def tail_logfile(header: str, file: str, dst: TextIO, finished: Event, interval_sec: float): ...

class TailLog:
    '''
    Tail the given log files.

    The log files do not have to exist when the ``start()`` method is called. The tail-er will gracefully wait until
    the log files are created by the producer and will tail the contents of the
    log files until the ``stop()`` method is called.

    .. warning:: ``TailLog`` will wait indefinitely for the log file to be created!

    Each log file\'s line will be suffixed with a header of the form: ``[{name}{idx}]:``,
    where the ``name`` is user-provided and ``idx`` is the index of the log file
    in the ``log_files`` mapping. ``log_line_prefixes`` can be used to override the
    header for each log file.

    Usage:

    ::

     log_files = {0: "/tmp/0_stdout.log", 1: "/tmp/1_stdout.log"}
     tailer = TailLog("trainer", log_files, sys.stdout).start()
     # actually run the trainers to produce 0_stdout.log and 1_stdout.log
     run_trainers()
     tailer.stop()

     # once run_trainers() start writing the ##_stdout.log files
     # the tailer will print to sys.stdout:
     # >>> [trainer0]:log_line1
     # >>> [trainer1]:log_line1
     # >>> [trainer0]:log_line2
     # >>> [trainer0]:log_line3
     # >>> [trainer1]:log_line2

    .. note:: Due to buffering log lines between files may not necessarily
              be printed out in order. You should configure your application\'s
              logger to suffix each log line with a proper timestamp.

    '''
    _threadpool: Incomplete
    _name: Incomplete
    _dst: Incomplete
    _log_files: Incomplete
    _log_line_prefixes: Incomplete
    _finished_events: dict[int, Event]
    _futs: list[Future]
    _interval_sec: Incomplete
    _stopped: bool
    def __init__(self, name: str, log_files: dict[int, str], dst: TextIO, log_line_prefixes: dict[int, str] | None = None, interval_sec: float = 0.1) -> None: ...
    def start(self) -> TailLog: ...
    def stop(self) -> None: ...
    def stopped(self) -> bool: ...
