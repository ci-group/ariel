__all__ = ['enable', 'is_enabled', 'tuning_enable', 'tuning_is_enabled', 'record_untuned_enable', 'record_untuned_is_enabled', 'set_max_tuning_duration', 'get_max_tuning_duration', 'set_max_tuning_iterations', 'get_max_tuning_iterations', 'set_filename', 'get_filename', 'get_results', 'get_validators', 'write_file_on_exit', 'write_file', 'read_file', 'tune_gemm_in_file', 'mgpu_tune_gemm_in_file', 'set_rotating_buffer_size', 'get_rotating_buffer_size']

def enable(val: bool = True) -> None:
    """This is the big on/off switch for all TunableOp implementations."""
def is_enabled() -> bool:
    """Returns whether the TunableOp feature is enabled."""
def tuning_enable(val: bool = True) -> None:
    """Enable tuning of TunableOp implementations.

    When enabled, if a tuned entry isn't found, run the tuning step and record
    the entry.
    """
def tuning_is_enabled() -> bool:
    """Returns whether TunableOp implementations can be tuned."""
def record_untuned_enable(val: bool = True) -> None:
    """Enable recording untuned of TunableOp perations for offline tuning.

    When enabled, if a tuned entry isn't found, write it to the untuned file.
    """
def record_untuned_is_enabled() -> bool:
    """Returns whether TunableOp operations are recorded for offline tuning."""
def set_max_tuning_duration(duration: int) -> None:
    """Set max time in milliseconds to spend tuning a given solution.

    If both max tuning duration and iterations are set, the smaller of the two
    will be honored. At minimum 1 tuning iteration will always be run.
    """
def get_max_tuning_duration() -> int:
    """Get max time to spend tuning a given solution."""
def set_max_tuning_iterations(iterations: int) -> None:
    """Set max number of iterations to spend tuning a given solution.

    If both max tuning duration and iterations are set, the smaller of the two
    will be honored. At minimum 1 tuning iteration will always be run.
    """
def get_max_tuning_iterations() -> int:
    """Get max iterations to spend tuning a given solution."""
def set_filename(filename: str, insert_device_ordinal: bool = False) -> None:
    """Set the filename to use for input/output of tuning results.

    If :attr:`insert_device_ordinal` is ``True`` then the current device ordinal
    will be added to the given filename automatically. This can be used in a
    1-process-per-gpu cenario to ensure all processes write to a separate file.
    """
def get_filename() -> str:
    """Get the results filename."""
def get_results() -> tuple[str, str, str, float]:
    """Return all TunableOp results."""
def get_validators() -> tuple[str, str]:
    """Return the TunableOp validators."""
def write_file_on_exit(val: bool) -> None:
    """During Tuning Context destruction, write file to disk.

    This is useful as a final flush of your results to disk if your application
    terminates as result of normal operation or an error. Manual flushing of
    your results can be achieved by manually calling ``write_file()``."""
def write_file(filename: str | None = None) -> bool:
    """Write results to a CSV file.

    If :attr:`filename` is not given, ``get_filename()`` is called.
    """
def read_file(filename: str | None = None) -> bool:
    """Read results from a TunableOp CSV file.

    If :attr:`filename` is not given, ``get_filename()`` is called.
    """
def set_rotating_buffer_size(buffer_size: int) -> None:
    """Set rotating buffer size to this value in MB, if the buffer size is greater than zero.

    If less than zero, query L2 cache size. If equal to zero, means deactivate rotating buffer.
    """
def get_rotating_buffer_size() -> int:
    """Get the rotating buffer size in kilobytes."""
def tune_gemm_in_file(filename: str) -> None:
    """tune GEMM in file."""
def mgpu_tune_gemm_in_file(filename_pattern: str, num_gpus: int) -> None:
    """Process one or more files and distribute work over one or more GPUs."""
