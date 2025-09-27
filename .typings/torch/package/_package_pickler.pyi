from .importer import Importer as Importer, ObjMismatchError as ObjMismatchError, ObjNotFoundError as ObjNotFoundError, sys_importer as sys_importer
from _typeshed import Incomplete
from pickle import _Pickler

class _PyTorchLegacyPickler(_Pickler):
    _persistent_id: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def persistent_id(self, obj): ...

class PackagePickler(_PyTorchLegacyPickler):
    """Package-aware pickler.

    This behaves the same as a normal pickler, except it uses an `Importer`
    to find objects and modules to save.
    """
    importer: Incomplete
    dispatch: Incomplete
    def __init__(self, importer: Importer, *args, **kwargs) -> None: ...
    def save_global(self, obj, name=None) -> None: ...

def create_pickler(data_buf, importer, protocol: int = 4): ...
