import dataclasses
from _typeshed import Incomplete
from collections.abc import Iterable
from mujoco.studio import hot_reload_utils as utils, messages as messages, plugin_registry as plugin_registry
from typing import Any

FileWatcher = utils.FileWatcher
logger: Incomplete

@dataclasses.dataclass(frozen=True)
class Options:
    """What hot reloading covers; ``parse()`` reads it from ``--hot_reload``.

  Attributes:
    model: Whether the shown model file (with its includes and assets) is
      reloaded when saved.
    plugins: Which registered plugins are reloaded when their module is saved:
      all of them (True), none (False), or those whose class has one of the
      given names.
  """
    model: bool = ...
    plugins: bool | frozenset[str] = ...
    @classmethod
    def parse(cls, text: str) -> Options:
        """Parses a ``--hot_reload`` value.

    ``all`` covers plugins and the model, ``plugins`` and ``model`` one of the
    two, and anything else names a plugin class to reload, e.g.
    ``MySimPlugin,MyViewerPlugin,...``. Items combine with commas
    (``model,MySimPlugin``).

    Args:
      text: The flag value; empty for nothing.

    Raises:
      ValueError: If an item is neither a keyword nor a valid class name.
    """
    def reloads(self, plugin: Any) -> bool:
        """Returns whether ``plugin`` is reloaded when its module is saved."""

@dataclasses.dataclass
class _RegistryState:
    """Per-registry bookkeeping."""
    host: Any
    seen_generation: int
    swapped: set[int] = dataclasses.field(default_factory=set)
    logged: set[tuple[int, str]] = dataclasses.field(default_factory=set)

class Reloader:
    """Watches plugin modules and hot-reloads them into plugin registries.

  One instance serves every registry in the process, since modules are shared.
  Each registry is updated only from its own ``tick()`` call, i.e. on its own
  thread and between frames or steps.
  """
    generation: int
    def __init__(self, options: Options = ..., poll_interval: float = ...) -> None: ...
    @property
    def options(self) -> Options:
        """What this reloader covers."""
    @property
    def watched_modules(self) -> set[str]:
        """Names of the modules that are reloaded when their source changes."""
    @property
    def poll_interval(self) -> float:
        """Minimum seconds between two checks of the watched files."""
    def tick(self, registry: plugin_registry.PluginRegistry, host: Any) -> None:
        """Reloads changed modules and updates ``registry``'s plugins.

    Args:
      registry: The registry to update.
      host: The object that owns ``registry`` (a Viewer or ViewerHandle). Its
        modules are framework code and are never reloaded.
    """

def enable(what: str = 'all') -> None:
    """Configures hot reloading for this process ('' disables it).

  Calling ``enable`` again updates what is reloaded.

  Args:
    what: What to reload; see ``Options.parse``.
  """
def watches_model() -> bool:
    """Returns whether the model file shown is reloaded when saved."""
def watch_files(paths: Iterable[str]) -> FileWatcher | None:
    """Returns a watcher for the files a model is built from, if enabled.

  The watcher polls at the reloader's rate, so polling it every step is cheap.

  Args:
    paths: The files to watch.
  """
def tick(registry: plugin_registry.PluginRegistry, host: Any) -> None:
    """Hot-reloads changed plugin modules into ``registry``, if enabled.

  Call once per frame or step, never from inside a handler.

  Args:
    registry: The registry to update.
    host: The object that owns ``registry`` (a Viewer or ViewerHandle).
  """
