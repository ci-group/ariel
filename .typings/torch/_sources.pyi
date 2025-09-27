import ast
import functools
from _typeshed import Incomplete
from torch._C import ErrorReport as ErrorReport
from torch._C._jit_tree_views import SourceRangeFactory as SourceRangeFactory
from typing import Any, NamedTuple

def get_source_lines_and_file(obj: Any, error_msg: str | None = None) -> tuple[list[str], int, str | None]:
    """
    Wrapper around inspect.getsourcelines and inspect.getsourcefile.

    Returns: (sourcelines, file_lino, filename)
    """
def normalize_source_lines(sourcelines: list[str]) -> list[str]:
    """
    This helper function accepts a list of source lines. It finds the
    indentation level of the function definition (`def`), then it indents
    all lines in the function body to a point at or greater than that
    level. This allows for comments and continued string literals that
    are at a lower indentation than the rest of the code.
    Args:
        sourcelines: function source code, separated into lines by
                        the '
' character
    Returns:
        A list of source lines that have been correctly aligned
    """

class SourceContext(SourceRangeFactory):
    uses_true_division: Incomplete
    filename: Incomplete
    funcname: Incomplete
    def __init__(self, source, filename, file_lineno, leading_whitespace_len, uses_true_division: bool = True, funcname=None) -> None: ...

@functools.cache
def make_source_context(*args): ...
def fake_range(): ...

class ParsedDef(NamedTuple):
    ast: ast.Module
    ctx: SourceContext
    source: str
    filename: str | None
    file_lineno: int

def parse_def(fn): ...
