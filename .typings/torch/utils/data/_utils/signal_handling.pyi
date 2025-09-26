from . import IS_WINDOWS as IS_WINDOWS
from torch._C import _error_if_any_worker_fails as _error_if_any_worker_fails, _remove_worker_pids as _remove_worker_pids, _set_worker_pids as _set_worker_pids, _set_worker_signal_handlers as _set_worker_signal_handlers

_SIGCHLD_handler_set: bool

def _set_SIGCHLD_handler() -> None: ...
