from _typeshed import Incomplete
from mujoco.studio import messages as messages, plugin as plugin, sim as sim

class StepControl(plugin.SimPlugin):
    """Default sim plugin that steps CPU physics with real-time pacing.

  Pass your own plugin instead to step differently (e.g. a GPU simulation).

  Pacing: this plugin owns real-time pacing. It steps as much sim time as fits
  the wall-clock budget of one loop iteration, then sleeps 1ms per StepEvent.
  The sleep matters because advance() returns at once when no step is due yet
  or while paused, and sync() itself never sleeps: without it, a bare
  ``while handle.is_running(): handle.sync()`` loop spins at 100% CPU and
  starves the viewer threads of the GIL. The 1ms yield matches the physics
  loops of the simulate app and of the native Studio app; advance() catches up
  on the next iteration. It lives in this plugin rather than in advance() so
  that advance() stays non-blocking for callers on render threads (the browser
  build, scripts with their own frame loop). A plugin that replaces this one
  must at minimum pace each iteration the same way.

  Reporting: a ``StepControlSnapshot`` goes to the viewer whenever a setting
  (pause, speed, noise) changes, and whenever the achieved speed has moved by
  ``_MEASURED_SPEED_DEADBAND``.
  """
    step_control: Incomplete
    def __init__(self) -> None: ...
