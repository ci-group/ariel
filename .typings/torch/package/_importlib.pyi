from _typeshed import Incomplete

_zip_searchorder: Incomplete

def _normalize_line_endings(source): ...
def _resolve_name(name, package, level):
    """Resolve a relative module name to an absolute one."""
def _sanity_check(name, package, level) -> None:
    '''Verify arguments are "sane".'''
def _calc___package__(globals):
    """Calculate what __package__ should be.

    __package__ is not guaranteed to be defined or could be set to None
    to represent that its proper value is unknown.

    """
def _normalize_path(path):
    """Normalize a path by ensuring it is a string.

    If the resulting string contains path separators, an exception is raised.
    """
