from typing import Any

__all__ = ['ErrorHandler']

class ErrorHandler:
    """
    Write the provided exception object along with some other metadata about
    the error in a structured way in JSON format to an error file specified by the
    environment variable: ``TORCHELASTIC_ERROR_FILE``. If this environment
    variable is not set, then simply logs the contents of what would have been
    written to the error file.

    This handler may be subclassed to customize the handling of the error.
    Subclasses should override ``initialize()`` and ``record_exception()``.
    """
    def _get_error_file_path(self) -> str | None:
        """
        Return the error file path.

        May return ``None`` to have the structured error be logged only.
        """
    def initialize(self) -> None:
        """
        Call prior to running code that we wish to capture errors/exceptions.

        Typically registers signal/fault handlers. Users can override this
        function to add custom initialization/registrations that aid in
        propagation/information of errors/signals/exceptions/faults.
        """
    def _write_error_file(self, file_path: str, error_msg: str) -> None:
        """Write error message to the file."""
    def record_exception(self, e: BaseException) -> None:
        """
        Write a structured information about the exception into an error file in JSON format.

        If the error file cannot be determined, then logs the content
        that would have been written to the error file.
        """
    def override_error_code_in_rootcause_data(self, rootcause_error_file: str, rootcause_error: dict[str, Any], error_code: int = 0):
        """Modify the rootcause_error read from the file, to correctly set the exit code."""
    def dump_error_file(self, rootcause_error_file: str, error_code: int = 0):
        """Dump parent error file from child process's root cause error and error code."""
    def _rm(self, my_error_file) -> None: ...
