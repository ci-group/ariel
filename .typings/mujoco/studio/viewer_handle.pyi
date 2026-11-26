import contextlib
import dataclasses
import mujoco
import os
from _typeshed import Incomplete
from mujoco.studio import endpoints as endpoints, hot_reload as hot_reload, messages as messages, parser as parser, plugin_registry as plugin_registry
from typing import Any, Callable, Iterable

logger: Incomplete
IsAliveFn = Callable[[], bool]
ShutdownFn = Callable[[float], None]
ForwardErrorFn = Callable[[], None]

@dataclasses.dataclass(frozen=True)
class SimInitEvent(messages.Event):
    """Lifecycle event dispatched locally once when ``ViewerHandle`` initializes.

  ``plugin.SimPlugin`` caches ``event.handle`` as ``self.handle`` above
  ``CRITICAL`` priority, so ``self.handle`` is already set when subclass
  handlers run (raising aborts the launch and shuts down the viewer).
  """
    handle: ViewerHandle

class ViewerHandle:
    """A handle for interacting with a running viewer application from the sim."""
    model: mujoco.MjModel | None
    data: mujoco.MjData | None
    model_path: str
    def __init__(self, sim_endpoint: endpoints.SimEndpoint, *, sim_plugins: list[Any] | None = None, is_alive_fn: IsAliveFn | None = None, shutdown_fn: ShutdownFn | None = None, forward_error_fn: ForwardErrorFn | None = None) -> None:
        """Initializes the ViewerHandle.

    Args:
      sim_endpoint: The endpoint to use for communication with the viewer.
      sim_plugins: Optional list of plugin instances for sim-side processing,
        are classes with methods decorated with ``@handler``.
      is_alive_fn: Optional liveness check; without one the viewer is assumed to
        be running until ``close()`` is called.
      shutdown_fn: Optional launcher-owned shutdown hook, called by ``close()``.
      forward_error_fn: Optional hook that raises any unhandled exception from
        the viewer thread.
    """
    def close(self, reason: str = '') -> None:
        """Signals the viewer to exit and waits for it to shut down.

    Re-raises any unhandled exception that terminated the viewer thread, so a
    viewer-side failure surfaces on the sim side rather than being lost.

    Args:
      reason: Why the viewer is being closed, when this is not a normal exit
        (see ExitEvent.reason). The web viewer shows it in the browser.
    """
    def __enter__(self) -> ViewerHandle: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None: ...
    def is_running(self) -> bool:
        """Returns True while the viewer is open.

    Closes the handle once the viewer has gone away, so this is also where an
    unhandled viewer-thread exception is re-raised (see ``close()``).
    """
    def send_to_viewer(self, message: messages.Message) -> None:
        """Sends an event or snapshot message to the viewer process.

    Args:
      message: The message to send.
    """
    def send_if_changes(self, *make_messages: messages.MakeMessage) -> contextlib.AbstractContextManager[None]:
        """Sends the messages that change during the ``with`` block to the viewer.

    Calls each ``make_message()`` on entering and on leaving the block and
    sends those that differ. Nothing is sent if the handle is closed or its
    model is replaced during the block.
    """
    def dispatch(self, message: messages.Message) -> None:
        """Dispatches a message to sim-side handlers in priority order."""
    def set_model(self, model: mujoco.MjSpec | mujoco.MjModel, path: str = '', data: mujoco.MjData | None = None, *, watch: Iterable[str | os.PathLike[str]] = ()) -> None:
        """Publishes ``model`` to sim plugins and the viewer as the current model.

    Can be called from a ``LoadModelEvent`` handler (which returns ``True`` to
    consume the event; see ``LoadModelEvent``) or from any other sim-side
    handler (e.g. a GUI event handler)::

      @messages.handler
      def on_load_model(self, event: messages.LoadModelEvent) -> bool:
        spec = mujoco.MjSpec()
        ...
        self.handle.set_model(spec)
        return True

    Args:
      model: The new model, or an ``MjSpec`` to compile.
      path: The file that produced ``model``, if any; see ``ModelEvent.path``.
      data: The data for ``model``; a fresh ``MjData`` is created when None.
      watch: Extra files read by a Python scene while building ``model`` (e.g.
        a base XML loaded with ``MjSpec.from_file('robot.xml')`` to make the
        initial spec). A model file loaded via ``path`` is already watched
        automatically; ``watch`` lets a Python scene (where ``path`` is empty)
        also trigger ``reload()`` when those files or their ``<include>``s and
        assets are saved.
    """
    def reload(self) -> None:
        """Reloads the current model, as the Reload button does.

    Dispatches ``LoadModelEvent(path=self.model_path)``: a model file is parsed
    again, or a sim plugin that handles ``LoadModelEvent`` builds the model
    again (e.g. after a GUI event or ``HotReloadEvent``). Call on the sim
    thread.
    """
    def sync(self, model: mujoco.MjModel | None = None, data: mujoco.MjData | None = None) -> tuple[mujoco.MjModel | None, mujoco.MjData | None]:
        """Syncs the simulation with the viewer and returns the updated sim state.

    This method processes incoming messages from the viewer, dispatches
    ``PreStepEvent``, ``StepEvent``, and ``PostStepEvent`` so sim-side plugins
    can prepare, advance, and observe the simulation, and sends the resulting
    simulation state to the viewer as a snapshot.

    ``sync`` itself neither advances physics nor sleeps. When using
    ``run_sim_loop``, include a ``StepEvent`` plugin in ``sim_plugins`` (e.g.
    ``step_control.StepControl()`` for real-time-paced CPU stepping, or your
    own plugin to step differently) that also paces each iteration to avoid
    busy-spinning. Alternatively, a user script can write its own loop around
    ``sync`` and step and pace the simulation directly (see
    ``sample/custom_sim_loop.py``).

    Sim-side edits belong in handlers of those three events: ``sync`` checks
    ``model.opt`` for changes across ``PreStepEvent``, ``StepEvent``, and
    ``PostStepEvent`` and sends an ``MjOptionSnapshot`` to the viewer when it
    was modified.

    Args:
      model: Optional model owned by the caller's loop. If None, keeps
        simulating the model already on the handle (from ``set_model`` or
        ``LoadModelEvent``).
      data: The data for ``model``; ignored when ``model`` is None.

    Returns:
      The current model and data; rebind both, they may be new objects (e.g.
      after the viewer sends a ModelEvent).
    """

def run_sim_loop(handle: ViewerHandle) -> None:
    """Runs the standard simulation sync loop until the viewer closes.

  Publish a model before calling this (``handle.set_model`` or a
  ``LoadModelEvent``), or include a sim plugin that does; until one arrives the
  viewer shows an empty scene.

  With this loop, all sim-side behavior—stepping, pacing, and edits to
  ``model``, ``data``, or plugin state—belongs in ``sim_plugins`` handling
  ``PreStepEvent``, ``StepEvent``, and ``PostStepEvent``, which ``sync``
  dispatches inside its ``model.opt`` change detection. For an example of a
  script that owns its loop, stepping and pacing itself, see
  ``sample/custom_sim_loop.py``.
  """
