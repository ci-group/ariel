from threading import Thread

_IN_TOPLEVEL_PROCESS: bool

def in_toplevel_process() -> bool: ...
def _async_compile_initializer(orig_ppid: int) -> None: ...

_watchdog_thread: Thread | None
_original_parent: int | None

def has_parent_changed() -> bool: ...
