import types
from collections.abc import Iterable, Mapping
from typing import Any

class FileWatcher:
    """Detects file changes by polling ``os.stat``.

  Paths are resolved with ``os.path.realpath`` so that edits to the workspace
  file behind a runfiles symlink are detected.
  """
    def __init__(self, paths: Iterable[str] = (), poll_interval: float = 0.0) -> None:
        """Initializes the watcher.

    Args:
      paths: Files to start watching.
      poll_interval: Minimum seconds between two polls that check the files;
        ``poll()`` calls in between return nothing.
    """
    def watch(self, path: str) -> str:
        """Starts watching ``path``; returns the resolved path."""
    def poll(self) -> set[str]:
        """Returns the resolved paths that changed since the previous poll."""

def source_path(module: types.ModuleType) -> str | None:
    """Returns the resolved ``.py`` source of a module, or None if it has none."""
def module_dir(name: str) -> str | None:
    """Returns the directory of a loaded module's source, if it has one."""
def local_imports(module: types.ModuleType) -> list[types.ModuleType]:
    """Returns the ``.py`` modules in the same directory that ``module`` uses."""
def reload_module(module: types.ModuleType) -> None:
    """Re-executes a module's current source in place."""
def reload_order(deps: Mapping[str, set[str]], roots: Iterable[str]) -> list[str]:
    """Returns ``roots`` and their importers, each after the modules it imports.

  Args:
    deps: Mapping from each watched module name to the watched module names it
      imports.
    roots: Module names whose source changed (or previously failed to reload).
  """
def current_class(cls) -> type[Any] | None:
    """Returns the class now bound to ``cls``'s module and qualified name."""
def class_name(cls) -> str:
    """Returns the fully qualified ``module.QualName`` of ``cls``."""
def same_default(a: Any, b: Any) -> bool:
    """Returns whether two class-attribute default values are equal."""
