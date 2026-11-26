from mujoco.studio import messages as messages
from typing import Any, Callable, NamedTuple

Handler = Callable[..., Any]
HandlerWrapper = Callable[[Any, Handler], Handler]

class PrioritizedHandler(NamedTuple):
    """A registered handler and its priority; higher priorities run first."""
    priority: int
    handler: Handler

class PluginRegistry:
    """Runtime registry of discovered handlers from plugin instances.

  The set of plugins is fixed at construction: there is no late registration.
  ``hot_reload`` relies on this — it scans ``plugins`` once per registry, and a
  reload swaps the class of an existing instance and calls ``rebuild_handlers``,
  but never adds an instance. Plugins therefore need to be registered up front;
  see the hot-reload design goal in ``hot_reload.py``.
  """
    plugins: list[Any]
    handlers: dict[Any, list[PrioritizedHandler]]
    def __init__(self, plugins: list[Any] | None = None) -> None: ...
    def rebuild_handlers(self, wrap: HandlerWrapper | None = None) -> None:
        """Discovers and registers handlers for all plugins in ``self.plugins``.

    Args:
      wrap: Optional ``wrap(plugin, handler)`` returning the callable to
        register in place of ``handler`` (e.g. to log its exceptions).
    """
    def dispatch(self, message: Any) -> None:
        """Dispatches a message to registered handlers in priority order."""
