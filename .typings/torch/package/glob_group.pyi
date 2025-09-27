from _typeshed import Incomplete
from collections.abc import Iterable

GlobPattern = str | Iterable[str]

class GlobGroup:
    '''A set of patterns that candidate strings will be matched against.

    A candidate is composed of a list of segments separated by ``separator``, e.g. "foo.bar.baz".

    A pattern contains one or more segments. Segments can be:
        - A literal string (e.g. "foo"), which matches exactly.
        - A string containing a wildcard (e.g. "torch*", or "foo*baz*"). The wildcard matches
          any string, including the empty string.
        - A double wildcard ("**"). This matches against zero or more complete segments.

    Examples:
        ``torch.**``: matches ``torch`` and all its submodules, e.g. ``torch.nn`` and ``torch.nn.functional``.
        ``torch.*``: matches ``torch.nn`` or ``torch.functional``, but not ``torch.nn.functional``.
        ``torch*.**``: matches ``torch``, ``torchvision``, and all their submodules.

    A candidates will match the ``GlobGroup`` if it matches any of the ``include`` patterns and
    none of the ``exclude`` patterns.

    Args:
        include (Union[str, Iterable[str]]): A string or list of strings,
            each representing a pattern to be matched against. A candidate
            will match if it matches *any* include pattern
        exclude (Union[str, Iterable[str]]): A string or list of strings,
            each representing a pattern to be matched against. A candidate
            will be excluded from matching if it matches *any* exclude pattern.
        separator (str): A string that delimits segments in candidates and
            patterns. By default this is "." which corresponds to how modules are
            named in Python. Another common value for this is "/", which is
            the Unix path separator.
    '''
    _dbg: Incomplete
    include: Incomplete
    exclude: Incomplete
    separator: Incomplete
    def __init__(self, include: GlobPattern, *, exclude: GlobPattern = (), separator: str = '.') -> None: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def matches(self, candidate: str) -> bool: ...
    @staticmethod
    def _glob_list(elems: GlobPattern, separator: str = '.'): ...
    @staticmethod
    def _glob_to_re(pattern: str, separator: str = '.'): ...
