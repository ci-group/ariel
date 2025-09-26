import types
from _typeshed import Incomplete

def is_available():
    """Return whether PyTorch is built with MKL support."""

VERBOSE_OFF: int
VERBOSE_ON: int

class verbose:
    """
    On-demand oneMKL verbosing functionality.

    To make it easier to debug performance issues, oneMKL can dump verbose
    messages containing execution information like duration while executing
    the kernel. The verbosing functionality can be invoked via an environment
    variable named `MKL_VERBOSE`. However, this methodology dumps messages in
    all steps. Those are a large amount of verbose messages. Moreover, for
    investigating the performance issues, generally taking verbose messages
    for one single iteration is enough. This on-demand verbosing functionality
    makes it possible to control scope for verbose message dumping. In the
    following example, verbose messages will be dumped out for the second
    inference only.

    .. highlight:: python
    .. code-block:: python

        import torch
        model(data)
        with torch.backends.mkl.verbose(torch.backends.mkl.VERBOSE_ON):
            model(data)

    Args:
        level: Verbose level
            - ``VERBOSE_OFF``: Disable verbosing
            - ``VERBOSE_ON``:  Enable verbosing
    """
    enable: Incomplete
    def __init__(self, enable) -> None: ...
    def __enter__(self): ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: types.TracebackType | None): ...
