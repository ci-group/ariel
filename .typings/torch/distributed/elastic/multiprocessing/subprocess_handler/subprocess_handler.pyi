import signal
import subprocess
from _typeshed import Incomplete

__all__ = ['SubprocessHandler']

class SubprocessHandler:
    """
    Convenience wrapper around python's ``subprocess.Popen``. Keeps track of
    meta-objects associated to the process (e.g. stdout and stderr redirect fds).
    """
    _stdout: Incomplete
    _stderr: Incomplete
    local_rank_id: Incomplete
    proc: subprocess.Popen
    def __init__(self, entrypoint: str, args: tuple, env: dict[str, str], stdout: str | None, stderr: str | None, local_rank_id: int) -> None: ...
    def _popen(self, args: tuple, env: dict[str, str]) -> subprocess.Popen: ...
    def close(self, death_sig: signal.Signals | None = None) -> None: ...
