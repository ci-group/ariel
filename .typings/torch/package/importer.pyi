import abc
from abc import ABC, abstractmethod
from types import ModuleType
from typing import Any

__all__ = ['ObjNotFoundError', 'ObjMismatchError', 'Importer', 'OrderedImporter']

class ObjNotFoundError(Exception):
    """Raised when an importer cannot find an object by searching for its name."""
class ObjMismatchError(Exception):
    """Raised when an importer found a different object with the same name as the user-provided one."""

class Importer(ABC, metaclass=abc.ABCMeta):
    """Represents an environment to import modules from.

    By default, you can figure out what module an object belongs by checking
    __module__ and importing the result using __import__ or importlib.import_module.

    torch.package introduces module importers other than the default one.
    Each PackageImporter introduces a new namespace. Potentially a single
    name (e.g. 'foo.bar') is present in multiple namespaces.

    It supports two main operations:
        import_module: module_name -> module object
        get_name: object -> (parent module name, name of obj within module)

    The guarantee is that following round-trip will succeed or throw an ObjNotFoundError/ObjMisMatchError.
        module_name, obj_name = env.get_name(obj)
        module = env.import_module(module_name)
        obj2 = getattr(module, obj_name)
        assert obj1 is obj2
    """
    modules: dict[str, ModuleType]
    @abstractmethod
    def import_module(self, module_name: str) -> ModuleType:
        """Import `module_name` from this environment.

        The contract is the same as for importlib.import_module.
        """
    def get_name(self, obj: Any, name: str | None = None) -> tuple[str, str]:
        """Given an object, return a name that can be used to retrieve the
        object from this environment.

        Args:
            obj: An object to get the module-environment-relative name for.
            name: If set, use this name instead of looking up __name__ or __qualname__ on `obj`.
                This is only here to match how Pickler handles __reduce__ functions that return a string,
                don't use otherwise.
        Returns:
            A tuple (parent_module_name, attr_name) that can be used to retrieve `obj` from this environment.
            Use it like:
                mod = importer.import_module(parent_module_name)
                obj = getattr(mod, attr_name)

        Raises:
            ObjNotFoundError: we couldn't retrieve `obj by name.
            ObjMisMatchError: we found a different object with the same name as `obj`.
        """
    def whichmodule(self, obj: Any, name: str) -> str:
        """Find the module name an object belongs to.

        This should be considered internal for end-users, but developers of
        an importer can override it to customize the behavior.

        Taken from pickle.py, but modified to exclude the search into sys.modules
        """

class _SysImporter(Importer):
    """An importer that implements the default behavior of Python."""
    def import_module(self, module_name: str): ...
    def whichmodule(self, obj: Any, name: str) -> str: ...

class OrderedImporter(Importer):
    '''A compound importer that takes a list of importers and tries them one at a time.

    The first importer in the list that returns a result "wins".
    '''
    _importers: list[Importer]
    def __init__(self, *args) -> None: ...
    def _is_torchpackage_dummy(self, module):
        """Returns true iff this module is an empty PackageNode in a torch.package.

        If you intern `a.b` but never use `a` in your code, then `a` will be an
        empty module with no source. This can break cases where we are trying to
        re-package an object after adding a real dependency on `a`, since
        OrderedImportere will resolve `a` to the dummy package and stop there.

        See: https://github.com/pytorch/pytorch/pull/71520#issuecomment-1029603769
        """
    def import_module(self, module_name: str) -> ModuleType: ...
    def whichmodule(self, obj: Any, name: str) -> str: ...
