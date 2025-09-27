import contextlib
from collections.abc import Generator

__all__ = ['init', 'start', 'stop', 'profile']

def init(output_file, flags=None, output_mode: str = 'key_value') -> None: ...
def start() -> None:
    """Starts cuda profiler data collection.

    .. warning::
        Raises CudaError in case of it is unable to start the profiler.
    """
def stop() -> None:
    """Stops cuda profiler data collection.

    .. warning::
        Raises CudaError in case of it is unable to stop the profiler.
    """
@contextlib.contextmanager
def profile() -> Generator[None]:
    """
    Enable profiling.

    Context Manager to enabling profile collection by the active profiling tool from CUDA backend.
    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> import torch
        >>> model = torch.nn.Linear(20, 30).cuda()
        >>> inputs = torch.randn(128, 20).cuda()
        >>> with torch.cuda.profiler.profile() as prof:
        ...     model(inputs)
    """
