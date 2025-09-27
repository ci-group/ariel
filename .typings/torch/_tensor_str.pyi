import contextlib
import dataclasses
from _typeshed import Incomplete
from collections.abc import Generator
from torch import inf as inf
from typing import Any

@dataclasses.dataclass
class __PrinterOptions:
    precision: int = ...
    threshold: float = ...
    edgeitems: int = ...
    linewidth: int = ...
    sci_mode: bool | None = ...

PRINT_OPTS: Incomplete

def set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=None) -> None:
    """Set options for printing. Items shamelessly taken from NumPy

    Args:
        precision: Number of digits of precision for floating point output
            (default = 4).
        threshold: Total number of array elements which trigger summarization
            rather than full `repr` (default = 1000).
        edgeitems: Number of array items in summary at beginning and end of
            each dimension (default = 3).
        linewidth: The number of characters per line for the purpose of
            inserting line breaks (default = 80). Thresholded matrices will
            ignore this parameter.
        profile: Sane defaults for pretty printing. Can override with any of
            the above options. (any one of `default`, `short`, `full`)
        sci_mode: Enable (True) or disable (False) scientific notation. If
            None (default) is specified, the value is defined by
            `torch._tensor_str._Formatter`. This value is automatically chosen
            by the framework.

    Example::

        >>> # Limit the precision of elements
        >>> torch.set_printoptions(precision=2)
        >>> torch.tensor([1.12345])
        tensor([1.12])
        >>> # Limit the number of elements shown
        >>> torch.set_printoptions(threshold=5)
        >>> torch.arange(10)
        tensor([0, 1, 2, ..., 7, 8, 9])
        >>> # Restore defaults
        >>> torch.set_printoptions(profile='default')
        >>> torch.tensor([1.12345])
        tensor([1.1235])
        >>> torch.arange(10)
        tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    """
def get_printoptions() -> dict[str, Any]:
    """Gets the current options for printing, as a dictionary that
    can be passed as ``**kwargs`` to set_printoptions().
    """
@contextlib.contextmanager
def printoptions(**kwargs) -> Generator[None]:
    """Context manager that temporarily changes the print options.  Accepted
    arguments are same as :func:`set_printoptions`."""
def tensor_totype(t): ...

class _Formatter:
    floating_dtype: Incomplete
    int_mode: bool
    sci_mode: bool
    max_width: int
    def __init__(self, tensor) -> None: ...
    def width(self): ...
    def format(self, value): ...

def _scalar_str(self, formatter1, formatter2=None): ...
def _vector_str(self, indent, summarize, formatter1, formatter2=None): ...
def _tensor_str_with_formatter(self, indent, summarize, formatter1, formatter2=None): ...
def _tensor_str(self, indent): ...
def _add_suffixes(tensor_str, suffixes, indent, force_newline): ...
def get_summarized_data(self): ...
def _str_intern(inp, *, tensor_contents=None): ...
def _functorch_wrapper_str_intern(tensor, *, tensor_contents=None): ...
def _str(self, *, tensor_contents=None): ...
