from _typeshed import Incomplete

class FileBaton:
    """A primitive, file-based synchronization utility."""
    lock_file_path: Incomplete
    wait_seconds: Incomplete
    fd: Incomplete
    warn_after_seconds: Incomplete
    def __init__(self, lock_file_path, wait_seconds: float = 0.1, warn_after_seconds=None) -> None:
        """
        Create a new :class:`FileBaton`.

        Args:
            lock_file_path: The path to the file used for locking.
            wait_seconds: The seconds to periodically sleep (spin) when
                calling ``wait()``.
            warn_after_seconds: The seconds to wait before showing
                lock file path to warn existing lock file.
        """
    def try_acquire(self):
        """
        Try to atomically create a file under exclusive access.

        Returns:
            True if the file could be created, else False.
        """
    def wait(self) -> None:
        """
        Periodically sleeps for a certain amount until the baton is released.

        The amount of time slept depends on the ``wait_seconds`` parameter
        passed to the constructor.
        """
    def release(self) -> None:
        """Release the baton and removes its file."""
