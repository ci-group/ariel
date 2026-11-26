import os
from _typeshed import Incomplete
from collections.abc import Iterable
from mujoco.studio import messages as messages, viewer_handle as viewer_handle, viewer_protocol as viewer_protocol

SimInitEvent: Incomplete
ViewerInitEvent: Incomplete
ThemeEvent: Incomplete

class SimPlugin:
    """Base class for simulation-side plugins.

  Automatically caches ``self.handle`` (``self._handle``) when ``SimInitEvent``
  is dispatched; stored as a class attribute default so subclasses needn't call
  ``super().__init__()``.
  """
    @property
    def handle(self) -> viewer_handle.ViewerHandle: ...
    @handle.setter
    def handle(self, value: viewer_handle.ViewerHandle | None) -> None: ...

class ViewerPlugin:
    """Base class for viewer-side plugins.

  Automatically caches ``self.viewer`` (``self._viewer``) when
  ``ViewerInitEvent`` is dispatched; stored as a class attribute default so
  subclasses needn't call ``super().__init__()``.
  """
    @property
    def viewer(self) -> viewer_protocol.Viewer: ...
    @viewer.setter
    def viewer(self, value: viewer_protocol.Viewer | None) -> None: ...
Plugin = SimPlugin | ViewerPlugin

def split_plugins(plugins: Iterable[Plugin] | None = None) -> tuple[list[ViewerPlugin], list[SimPlugin]]:
    """Splits a sequence of plugins into ``(viewer_plugins, sim_plugins)``.

  Args:
    plugins: Iterable of ``ViewerPlugin`` and ``SimPlugin`` instances.

  Returns:
    A ``(viewer_plugins, sim_plugins)`` tuple preserving input order.

  Raises:
    TypeError: If an item is neither a ``ViewerPlugin`` nor a ``SimPlugin``,
      inherits from both, or is a class rather than an instance.
  """

PLUGIN_PACKAGE: str

def load_plugins(paths: Iterable[str | os.PathLike[str]] | None = None) -> list[Plugin]:
    """Imports ``.py`` files and instantiates every plugin class defined in them.

  A file ``dir/name.py`` is imported as ``studio_plugins.name``, and ``dir`` is
  appended to ``sys.path`` so the file can import helper files next to it
  (``import helper``). Every ``ViewerPlugin`` or ``SimPlugin`` subclass defined
  in the file (not merely imported into it) is instantiated with ``cls()``, in
  definition order. The file runs inside the current process, so it can only
  import packages present in the running binary's environment.

  Args:
    paths: Paths to ``.py`` files defining plugin classes. Empty entries are
      skipped.

  Returns:
    The plugin instances, in file order then definition order.

  Raises:
    FileNotFoundError: If a path is not a file.
    ImportError: If a file cannot be imported, or another file with the same
      name was loaded already.
    ValueError: If a file defines no ``ViewerPlugin`` or ``SimPlugin`` subclass.
  """
