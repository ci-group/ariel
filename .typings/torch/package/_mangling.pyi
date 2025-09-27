from _typeshed import Incomplete

_mangle_index: int

class PackageMangler:
    """
    Used on import, to ensure that all modules imported have a shared mangle parent.
    """
    _mangle_index: Incomplete
    _mangle_parent: Incomplete
    def __init__(self) -> None: ...
    def mangle(self, name) -> str: ...
    def demangle(self, mangled: str) -> str:
        """
        Note: This only demangles names that were mangled by this specific
        PackageMangler. It will pass through names created by a different
        PackageMangler instance.
        """
    def parent_name(self): ...

def is_mangled(name: str) -> bool: ...
def demangle(name: str) -> str:
    """
    Note: Unlike PackageMangler.demangle, this version works on any
    mangled name, irrespective of which PackageMangler created it.
    """
def get_mangle_prefix(name: str) -> str: ...
