import inspect
import torch
import types
from .file_structure_representation import Directory
from .glob_group import GlobPattern
from .importer import Importer
from _typeshed import Incomplete
from collections.abc import Generator
from torch.types import FileLike
from typing import Any, Callable

__all__ = ['PackageImporter']

class PackageImporter(Importer):
    '''Importers allow you to load code written to packages by :class:`PackageExporter`.
    Code is loaded in a hermetic way, using files from the package
    rather than the normal python import system. This allows
    for the packaging of PyTorch model code and data so that it can be run
    on a server or used in the future for transfer learning.

    The importer for packages ensures that code in the module can only be loaded from
    within the package, except for modules explicitly listed as external during export.
    The file ``extern_modules`` in the zip archive lists all the modules that a package externally depends on.
    This prevents "implicit" dependencies where the package runs locally because it is importing
    a locally-installed package, but then fails when the package is copied to another machine.
    '''
    modules: dict[str, types.ModuleType]
    zip_reader: Any
    filename: str
    root: Incomplete
    extern_modules: Incomplete
    patched_builtins: Incomplete
    _mangler: Incomplete
    storage_context: Any
    last_map_location: Incomplete
    Unpickler: Incomplete
    def __init__(self, file_or_buffer: FileLike | torch._C.PyTorchFileReader, module_allowed: Callable[[str], bool] = ...) -> None:
        """Open ``file_or_buffer`` for importing. This checks that the imported package only requires modules
        allowed by ``module_allowed``

        Args:
            file_or_buffer: a file-like object (has to implement :meth:`read`, :meth:`readline`, :meth:`tell`, and :meth:`seek`),
                a string, or an ``os.PathLike`` object containing a filename.
            module_allowed (Callable[[str], bool], optional): A method to determine if a externally provided module
                should be allowed. Can be used to ensure packages loaded do not depend on modules that the server
                does not support. Defaults to allowing anything.

        Raises:
            ImportError: If the package will use a disallowed module.
        """
    def import_module(self, name: str, package=None):
        """Load a module from the package if it hasn't already been loaded, and then return
        the module. Modules are loaded locally
        to the importer and will appear in ``self.modules`` rather than ``sys.modules``.

        Args:
            name (str): Fully qualified name of the module to load.
            package ([type], optional): Unused, but present to match the signature of importlib.import_module. Defaults to ``None``.

        Returns:
            types.ModuleType: The (possibly already) loaded module.
        """
    def load_binary(self, package: str, resource: str) -> bytes:
        '''Load raw bytes.

        Args:
            package (str): The name of module package (e.g. ``"my_package.my_subpackage"``).
            resource (str): The unique name for the resource.

        Returns:
            bytes: The loaded data.
        '''
    def load_text(self, package: str, resource: str, encoding: str = 'utf-8', errors: str = 'strict') -> str:
        '''Load a string.

        Args:
            package (str): The name of module package (e.g. ``"my_package.my_subpackage"``).
            resource (str): The unique name for the resource.
            encoding (str, optional): Passed to ``decode``. Defaults to ``\'utf-8\'``.
            errors (str, optional): Passed to ``decode``. Defaults to ``\'strict\'``.

        Returns:
            str: The loaded text.
        '''
    def load_pickle(self, package: str, resource: str, map_location=None) -> Any:
        '''Unpickles the resource from the package, loading any modules that are needed to construct the objects
        using :meth:`import_module`.

        Args:
            package (str): The name of module package (e.g. ``"my_package.my_subpackage"``).
            resource (str): The unique name for the resource.
            map_location: Passed to `torch.load` to determine how tensors are mapped to devices. Defaults to ``None``.

        Returns:
            Any: The unpickled object.
        '''
    def id(self):
        """
        Returns internal identifier that torch.package uses to distinguish :class:`PackageImporter` instances.
        Looks like::

            <torch_package_0>
        """
    def file_structure(self, *, include: GlobPattern = '**', exclude: GlobPattern = ()) -> Directory:
        '''Returns a file structure representation of package\'s zipfile.

        Args:
            include (Union[List[str], str]): An optional string e.g. ``"my_package.my_subpackage"``, or optional list of strings
                for the names of the files to be included in the zipfile representation. This can also be
                a glob-style pattern, as described in :meth:`PackageExporter.mock`

            exclude (Union[List[str], str]): An optional pattern that excludes files whose name match the pattern.

        Returns:
            :class:`Directory`
        '''
    def python_version(self):
        """Returns the version of python that was used to create this package.

        Note: this function is experimental and not Forward Compatible. The plan is to move this into a lock
        file later on.

        Returns:
            :class:`Optional[str]` a python version e.g. 3.8.9 or None if no version was stored with this package
        """
    def _read_extern(self): ...
    def _make_module(self, name: str, filename: str | None, is_package: bool, parent: str): ...
    def _load_module(self, name: str, parent: str): ...
    def _compile_source(self, fullpath: str, mangled_filename: str): ...
    def get_source(self, module_name) -> str: ...
    def get_resource_reader(self, fullname): ...
    def _install_on_parent(self, parent: str, name: str, module: types.ModuleType): ...
    def _do_find_and_load(self, name): ...
    def _find_and_load(self, name): ...
    def _gcd_import(self, name, package=None, level: int = 0):
        """Import and return the module based on its name, the package the call is
        being made from, and the level adjustment.

        This function represents the greatest common denominator of functionality
        between import_module and __import__. This includes setting __package__ if
        the loader did not.

        """
    def _handle_fromlist(self, module, fromlist, *, recursive: bool = False):
        """Figure out what __import__ should return.

        The import_ parameter is a callable which takes the name of module to
        import. It is required to decouple the function from assuming importlib's
        import implementation is desired.

        """
    def __import__(self, name, globals=None, locals=None, fromlist=(), level: int = 0): ...
    def _get_package(self, package):
        """Take a package name or module object and return the module.

        If a name, the module is imported.  If the passed or imported module
        object is not a package, raise an exception.
        """
    def _zipfile_path(self, package, resource=None): ...
    def _get_or_create_package(self, atoms: list[str]) -> _PackageNode | _ExternNode: ...
    def _add_file(self, filename: str):
        """Assembles a Python module out of the given file. Will ignore files in the .data directory.

        Args:
            filename (str): the name of the file inside of the package archive to be added
        """
    def _add_extern(self, extern_name: str): ...

class _PathNode: ...

class _PackageNode(_PathNode):
    source_file: Incomplete
    children: dict[str, _PathNode]
    def __init__(self, source_file: str | None) -> None: ...

class _ModuleNode(_PathNode):
    __slots__: Incomplete
    source_file: Incomplete
    def __init__(self, source_file: str) -> None: ...

class _ExternNode(_PathNode): ...
_orig_getfile = inspect.getfile

class _PackageResourceReader:
    """Private class used to support PackageImporter.get_resource_reader().

    Confirms to the importlib.abc.ResourceReader interface. Allowed to access
    the innards of PackageImporter.
    """
    importer: Incomplete
    fullname: Incomplete
    def __init__(self, importer, fullname) -> None: ...
    def open_resource(self, resource): ...
    def resource_path(self, resource): ...
    def is_resource(self, name): ...
    def contents(self) -> Generator[Incomplete]: ...
