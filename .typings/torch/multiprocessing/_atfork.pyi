__all__ = ['register_after_fork']

def register_after_fork(func) -> None:
    """Register a callable to be executed in the child process after a fork.

    Note:
        In python < 3.7 this will only work with processes created using the
        ``multiprocessing`` module. In python >= 3.7 it also works with
        ``os.fork()``.

    Args:
        func (function): Function taking no arguments to be called in the child after fork

    """
