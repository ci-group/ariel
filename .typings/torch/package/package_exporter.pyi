import types
from ._digraph import DiGraph
from .glob_group import GlobGroup, GlobPattern
from .importer import Importer
from _typeshed import Incomplete
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from torch.types import FileLike
from torch.utils.hooks import RemovableHandle
from typing import Any, IO

__all__ = ['PackagingErrorReason', 'EmptyMatchError', 'PackagingError', 'PackageExporter']

class _ModuleProviderAction(Enum):
    """Represents one of the actions that :class:`PackageExporter` can take on a module.

    See :meth:`PackageExporter.extern` and friends for a description of what the actions do.
    """
    INTERN = 1
    EXTERN = 2
    MOCK = 3
    DENY = 4
    REPACKAGED_MOCK_MODULE = 5
    SKIP = 6

class PackagingErrorReason(Enum):
    """Listing of different reasons a dependency may fail to package.

    This enum is used to provide good error messages when
    :class:`PackagingError` is raised.
    """
    def __repr__(self) -> str: ...
    IS_EXTENSION_MODULE = 'Module is a C extension module. torch.package supports Python modules only.'
    NO_DUNDER_FILE = 'Module had no __file__ defined.'
    SOURCE_FILE_NOT_FOUND = 'Module had a __file__, but we could not find it in your filesystem.'
    DEPENDENCY_RESOLUTION_FAILED = 'Dependency resolution failed.'
    NO_ACTION = 'Module did not match against any action pattern. Extern, mock, or intern it.'
    DENIED = 'Module was denied by a pattern.'
    MOCKED_BUT_STILL_USED = 'Module was mocked out, but is still being used in the package. Please intern or extern the mocked modules if objects are supposed to be in the package.'

@dataclass
class _PatternInfo:
    """Holds :class:`PackageExporter`-specific info about how to execute matches against"""
    action: _ModuleProviderAction
    allow_empty: bool
    was_matched: bool
    def __init__(self, action, allow_empty) -> None: ...

class EmptyMatchError(Exception):
    """This is an exception that is thrown when a mock or extern is marked as
    ``allow_empty=False``, and is not matched with any module during packaging.
    """

class PackagingError(Exception):
    """This exception is raised when there is an issue with exporting a package.
    ``PackageExporter`` will attempt to gather up all the errors and present
    them to you at once.
    """
    dependency_graph: Incomplete
    def __init__(self, dependency_graph: DiGraph, debug: bool = False) -> None: ...

class PackageExporter:
    '''Exporters allow you to write packages of code, pickled Python data, and
    arbitrary binary and text resources into a self-contained package.

    Imports can load this code in a hermetic way, such that code is loaded
    from the package rather than the normal Python import system. This allows
    for the packaging of PyTorch model code and data so that it can be run
    on a server or used in the future for transfer learning.

    The code contained in packages is copied file-by-file from the original
    source when it is created, and the file format is a specially organized
    zip file. Future users of the package can unzip the package, and edit the code
    in order to perform custom modifications to it.

    The importer for packages ensures that code in the module can only be loaded from
    within the package, except for modules explicitly listed as external using :meth:`extern`.
    The file ``extern_modules`` in the zip archive lists all the modules that a package externally depends on.
    This prevents "implicit" dependencies where the package runs locally because it is importing
    a locally-installed package, but then fails when the package is copied to another machine.

    When source code is added to the package, the exporter can optionally scan it
    for further code dependencies (``dependencies=True``). It looks for import statements,
    resolves relative references to qualified module names, and performs an action specified by the user
    (See: :meth:`extern`, :meth:`mock`, and :meth:`intern`).
    '''
    importer: Importer
    debug: Incomplete
    buffer: IO[bytes] | None
    zip_file: Incomplete
    _written_files: set[str]
    serialized_reduces: dict[int, Any]
    dependency_graph: Incomplete
    script_module_serializer: Incomplete
    storage_context: Incomplete
    _extern_hooks: OrderedDict
    _mock_hooks: OrderedDict
    _intern_hooks: OrderedDict
    patterns: dict[GlobGroup, _PatternInfo]
    _unique_id: int
    def __init__(self, f: FileLike, importer: Importer | Sequence[Importer] = ..., debug: bool = False) -> None:
        """
        Create an exporter.

        Args:
            f: The location to export to. Can be a  ``string``/``Path`` object containing a filename
                or a binary I/O object.
            importer: If a single Importer is passed, use that to search for modules.
                If a sequence of importers are passed, an ``OrderedImporter`` will be constructed out of them.
            debug: If set to True, add path of broken modules to PackagingErrors.
        """
    def save_source_file(self, module_name: str, file_or_directory: str, dependencies: bool = True):
        '''Adds the local file system ``file_or_directory`` to the source package to provide the code
        for ``module_name``.

        Args:
            module_name (str): e.g. ``"my_package.my_subpackage"``, code will be saved to provide code for this package.
            file_or_directory (str): the path to a file or directory of code. When a directory, all python files in the directory
                are recursively copied using :meth:`save_source_file`. If a file is named ``"/__init__.py"`` the code is treated
                as a package.
            dependencies (bool, optional): If ``True``, we scan the source for dependencies.
        '''
    def get_unique_id(self) -> str:
        """Get an id. This id is guaranteed to only be handed out once for this package."""
    def _get_dependencies(self, src: str, module_name: str, is_package: bool) -> list[str]:
        """Return all modules that this source code depends on.

        Dependencies are found by scanning the source code for import-like statements.

        Arguments:
            src: The Python source code to analyze for dependencies.
            module_name: The name of the module that ``src`` corresponds to.
            is_package: Whether this module should be treated as a package.
                See :py:meth:`save_source_string` for more info.

        Returns:
            A list containing modules detected as direct dependencies in
            ``src``.  The items in the list are guaranteed to be unique.
        """
    def save_source_string(self, module_name: str, src: str, is_package: bool = False, dependencies: bool = True):
        """Adds ``src`` as the source code for ``module_name`` in the exported package.

        Args:
            module_name (str): e.g. ``my_package.my_subpackage``, code will be saved to provide code for this package.
            src (str): The Python source code to save for this package.
            is_package (bool, optional): If ``True``, this module is treated as a package. Packages are allowed to have submodules
                (e.g. ``my_package.my_subpackage.my_subsubpackage``), and resources can be saved inside them. Defaults to ``False``.
            dependencies (bool, optional): If ``True``, we scan the source for dependencies.
        """
    def _write_source_string(self, module_name: str, src: str, is_package: bool = False):
        """Write ``src`` as the source code for ``module_name`` in the zip archive.

        Arguments are otherwise the same as for :meth:`save_source_string`.
        """
    def _import_module(self, module_name: str): ...
    def _module_exists(self, module_name: str) -> bool: ...
    def _get_source_of_module(self, module: types.ModuleType) -> str | None: ...
    def add_dependency(self, module_name: str, dependencies: bool = True):
        """Given a module, add it to the dependency graph according to patterns
        specified by the user.
        """
    def save_module(self, module_name: str, dependencies: bool = True):
        """Save the code for ``module`` into the package. Code for the module is resolved using the ``importers`` path to find the
        module object, and then using its ``__file__`` attribute to find the source code.

        Args:
            module_name (str): e.g. ``my_package.my_subpackage``, code will be saved to provide code
                for this package.
            dependencies (bool, optional): If ``True``, we scan the source for dependencies.
        """
    def _intern_module(self, module_name: str, dependencies: bool):
        """Adds the module to the dependency graph as an interned module,
        along with any metadata needed to write it out to the zipfile at serialization time.
        """
    def save_pickle(self, package: str, resource: str, obj: Any, dependencies: bool = True, pickle_protocol: int = 3):
        '''Save a python object to the archive using pickle. Equivalent to :func:`torch.save` but saving into
        the archive rather than a stand-alone file. Standard pickle does not save the code, only the objects.
        If ``dependencies`` is true, this method will also scan the pickled objects for which modules are required
        to reconstruct them and save the relevant code.

        To be able to save an object where ``type(obj).__name__`` is ``my_module.MyObject``,
        ``my_module.MyObject`` must resolve to the class of the object according to the ``importer`` order. When saving objects that
        have previously been packaged, the importer\'s ``import_module`` method will need to be present in the ``importer`` list
        for this to work.

        Args:
            package (str): The name of module package this resource should go in (e.g. ``"my_package.my_subpackage"``).
            resource (str): A unique name for the resource, used to identify it to load.
            obj (Any): The object to save, must be picklable.
            dependencies (bool, optional): If ``True``, we scan the source for dependencies.
        '''
    def save_text(self, package: str, resource: str, text: str):
        '''Save text data to the package.

        Args:
            package (str): The name of module package this resource should go it (e.g. ``"my_package.my_subpackage"``).
            resource (str): A unique name for the resource, used to identify it to load.
            text (str): The contents to save.
        '''
    def save_binary(self, package, resource, binary: bytes):
        '''Save raw bytes to the package.

        Args:
            package (str): The name of module package this resource should go it (e.g. ``"my_package.my_subpackage"``).
            resource (str): A unique name for the resource, used to identify it to load.
            binary (str): The data to save.
        '''
    def register_extern_hook(self, hook: ActionHook) -> RemovableHandle:
        """Registers an extern hook on the exporter.

        The hook will be called each time a module matches against an :meth:`extern` pattern.
        It should have the following signature::

            hook(exporter: PackageExporter, module_name: str) -> None

        Hooks will be called in order of registration.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                A handle that can be used to remove the added hook by calling
                ``handle.remove()``.
        """
    def register_mock_hook(self, hook: ActionHook) -> RemovableHandle:
        """Registers a mock hook on the exporter.

        The hook will be called each time a module matches against a :meth:`mock` pattern.
        It should have the following signature::

            hook(exporter: PackageExporter, module_name: str) -> None

        Hooks will be called in order of registration.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                A handle that can be used to remove the added hook by calling
                ``handle.remove()``.
        """
    def register_intern_hook(self, hook: ActionHook) -> RemovableHandle:
        """Registers an intern hook on the exporter.

        The hook will be called each time a module matches against an :meth:`intern` pattern.
        It should have the following signature::

            hook(exporter: PackageExporter, module_name: str) -> None

        Hooks will be called in order of registration.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                A handle that can be used to remove the added hook by calling
                ``handle.remove()``.
        """
    def intern(self, include: GlobPattern, *, exclude: GlobPattern = (), allow_empty: bool = True):
        '''Specify modules that should be packaged. A module must match some ``intern`` pattern in order to be
        included in the package and have its dependencies processed recursively.

        Args:
            include (Union[List[str], str]): A string e.g. "my_package.my_subpackage", or list of strings
                for the names of the modules to be externed. This can also be a glob-style pattern, as described in :meth:`mock`.

            exclude (Union[List[str], str]): An optional pattern that excludes some patterns that match the include string.

            allow_empty (bool): An optional flag that specifies whether the intern modules specified by this call
                to the ``intern`` method must be matched to some module during packaging. If an ``intern`` module glob
                pattern is added with ``allow_empty=False``, and :meth:`close` is called (either explicitly or via ``__exit__``)
                before any modules match that pattern, an exception is thrown. If ``allow_empty=True``, no such exception is thrown.

        '''
    def mock(self, include: GlobPattern, *, exclude: GlobPattern = (), allow_empty: bool = True):
        '''Replace some required modules with a mock implementation.  Mocked modules will return a fake
        object for any attribute accessed from it. Because we copy file-by-file, the dependency resolution will sometimes
        find files that are imported by model files but whose functionality is never used
        (e.g. custom serialization code or training helpers).
        Use this function to mock this functionality out without having to modify the original code.

        Args:
            include (Union[List[str], str]): A string e.g. ``"my_package.my_subpackage"``, or list of strings
                for the names of the modules to be mocked out. Strings can also be a glob-style pattern
                string that may match multiple modules. Any required dependencies that match this pattern
                string will be mocked out automatically.

                Examples :
                    ``\'torch.**\'`` -- matches ``torch`` and all submodules of torch, e.g. ``\'torch.nn\'``
                    and ``\'torch.nn.functional\'``

                    ``\'torch.*\'`` -- matches ``\'torch.nn\'`` or ``\'torch.functional\'``, but not
                    ``\'torch.nn.functional\'``

            exclude (Union[List[str], str]): An optional pattern that excludes some patterns that match the include string.
                e.g. ``include=\'torch.**\', exclude=\'torch.foo\'`` will mock all torch packages except ``\'torch.foo\'``,
                Default: is ``[]``.

            allow_empty (bool): An optional flag that specifies whether the mock implementation(s) specified by this call
                to the :meth:`mock` method must be matched to some module during packaging. If a mock is added with
                ``allow_empty=False``, and :meth:`close` is called (either explicitly or via ``__exit__``) and the mock has
                not been matched to a module used by the package being exported, an exception is thrown.
                If ``allow_empty=True``, no such exception is thrown.

        '''
    def extern(self, include: GlobPattern, *, exclude: GlobPattern = (), allow_empty: bool = True):
        '''Include ``module`` in the list of external modules the package can import.
        This will prevent dependency discovery from saving
        it in the package. The importer will load an external module directly from the standard import system.
        Code for extern modules must also exist in the process loading the package.

        Args:
            include (Union[List[str], str]): A string e.g. ``"my_package.my_subpackage"``, or list of strings
                for the names of the modules to be externed. This can also be a glob-style pattern, as
                described in :meth:`mock`.

            exclude (Union[List[str], str]): An optional pattern that excludes some patterns that match the
                include string.

            allow_empty (bool): An optional flag that specifies whether the extern modules specified by this call
                to the ``extern`` method must be matched to some module during packaging. If an extern module glob
                pattern is added with ``allow_empty=False``, and :meth:`close` is called (either explicitly or via
                ``__exit__``) before any modules match that pattern, an exception is thrown. If ``allow_empty=True``,
                no such exception is thrown.

        '''
    def deny(self, include: GlobPattern, *, exclude: GlobPattern = ()):
        '''Blocklist modules who names match the given glob patterns from the list of modules the package can import.
        If a dependency on any matching packages is found, a :class:`PackagingError` is raised.

        Args:
            include (Union[List[str], str]): A string e.g. ``"my_package.my_subpackage"``, or list of strings
                for the names of the modules to be externed. This can also be a glob-style pattern, as described in :meth:`mock`.

            exclude (Union[List[str], str]): An optional pattern that excludes some patterns that match the include string.
        '''
    def _persistent_id(self, obj): ...
    def __enter__(self): ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: types.TracebackType | None) -> None: ...
    def _write(self, filename, str_or_bytes) -> None: ...
    def _validate_dependency_graph(self) -> None: ...
    def _write_mock_file(self) -> None: ...
    def _execute_dependency_graph(self) -> None:
        """Takes a finalized dependency graph describing how to package all
        modules and executes it, writing to the ZIP archive.
        """
    def _write_python_version(self) -> None:
        """Writes the python version that the package was created with to .data/python_version"""
    def close(self) -> None:
        '''Write the package to the filesystem. Any calls after :meth:`close` are now invalid.
        It is preferable to use resource guard syntax instead::

            with PackageExporter("file.zip") as e:
                ...
        '''
    def _finalize_zip(self) -> None:
        """Called at the very end of packaging to leave the zipfile in a closed but valid state."""
    def _filename(self, package, resource): ...
    def _can_implicitly_extern(self, module_name: str): ...
    def dependency_graph_string(self) -> str:
        """Returns digraph string representation of dependencies in package.

        Returns:
            A string representation of dependencies in package.
        """
    def _nodes_with_action_type(self, action: _ModuleProviderAction | None) -> list[str]: ...
    def externed_modules(self) -> list[str]:
        """Return all modules that are currently externed.

        Returns:
            A list containing the names of modules which will be
            externed in this package.
        """
    def interned_modules(self) -> list[str]:
        """Return all modules that are currently interned.

        Returns:
            A list containing the names of modules which will be
            interned in this package.
        """
    def mocked_modules(self) -> list[str]:
        """Return all modules that are currently mocked.

        Returns:
            A list containing the names of modules which will be
            mocked in this package.
        """
    def denied_modules(self) -> list[str]:
        """Return all modules that are currently denied.

        Returns:
            A list containing the names of modules which will be
            denied in this package.
        """
    def get_rdeps(self, module_name: str) -> list[str]:
        """Return a list of all modules which depend on the module ``module_name``.

        Returns:
            A list containing the names of modules which depend on ``module_name``.
        """
    def all_paths(self, src: str, dst: str) -> str:
        """Return a dot representation of the subgraph
           that has all paths from src to dst.

        Returns:
            A dot representation containing all paths from src to dst.
            (https://graphviz.org/doc/info/lang.html)
        """
