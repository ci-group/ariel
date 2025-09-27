from .._mangling import is_mangled as is_mangled
from typing import Any

def is_from_package(obj: Any) -> bool:
    """
    Return whether an object was loaded from a package.

    Note: packaged objects from externed modules will return ``False``.
    """
