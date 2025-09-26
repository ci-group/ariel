import os
from typing import AnyStr

__all__ = ['fspath']

def fspath(path: AnyStr | os.PathLike[AnyStr]) -> AnyStr: ...
