import dataclasses
from _typeshed import Incomplete
from mujoco.studio import messages as messages, plugin as plugin, sim as sim, ux as ux

@dataclasses.dataclass(frozen=True)
class SimHistorySnapshot(messages.Snapshot):
    """Bidirectional snapshot carrying history-buffer state.

  The simulation owns an authoritative history buffer (``sim.SimHistory``) that
  records a frame after every physics step.  The viewer keeps a local mirror
  of the buffer's size, index, and head timestamp in ``ux_state`` so its
  scrubber widget can render a timeline, but never stores authoritative frame
  data — the displayed state always comes from the sim via ``StateSnapshot``.

  This single message type is used in both directions on separate directional
  channels, so there is no slot collision in the latest-wins SnapshotChannel:

  Viewer-to-sim (scrub request):
    The viewer sets only ``index`` (the desired history offset) and leaves
    ``size``, ``head_time``, and ``tail_time`` at their defaults.  The sim
    loads the requested frame into its own model/data, requests a pause, and
    the result reaches the viewer as the next StateSnapshot.

  Sim-to-viewer (buffer description):
    Sent whenever a frame is recorded, history is reset, or a scrub request is
    applied.  ``size``, ``index``, ``head_time``, and ``tail_time`` are
    populated so the viewer can render the scrubber timeline against the sim's
    authoritative index space.

  Attributes:
    size: Total number of frames stored in the sim's history.
    head_time: Simulation timestamp at the head of the history buffer (index 0).
    tail_time: Simulation timestamp of the oldest frame in the history buffer
      (index ``1 - size``).
    index: History offset in the sim's index space; 0 is the most recent state,
      negative values go further into the past.
  """
    size: int = ...
    head_time: float = ...
    tail_time: float = ...
    index: int = ...
    @classmethod
    def from_ux_state(cls, ux_state: ux.UxState) -> SimHistorySnapshot | None:
        """Packs the current scrub index of ``ux_state`` into a snapshot."""

class SimHistory(plugin.SimPlugin):
    """Sim-side plugin for simulation history recording and scrubbing.

  Owns the authoritative history buffer, records a frame after every physics
  step, handles scrub requests from the viewer, and broadcasts the buffer
  state as a ``SimHistorySnapshot`` whenever the buffer changes or is scrubbed.

  The buffer holds a bounded number of frames (``sim.SimHistory``), so at one
  frame per step it covers only a second or two of a large model stepped at
  500 Hz. Pass ``min_seconds`` to record every Nth step instead, with N chosen
  per model so the buffer spans at least that much simulated time; models
  small enough to fit are still recorded every step.
  """
    sim_history: Incomplete
    def __init__(self, *, min_seconds: float = 0.0) -> None:
        """Initializes the plugin.

    Args:
      min_seconds: Simulated seconds the history should span at least, by
        recording only every Nth step when the buffer is too small to hold
        every step. 0 records every step.
    """
    @property
    def frame_interval(self) -> float:
        """Simulated seconds between recorded frames for the current model."""
