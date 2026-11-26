from mujoco.studio import messages as vp

class ViewerEndpoint:
    """Viewer-side endpoint that routes Messages to/from the Sim.

  Outgoing (viewer-to-sim): EventChannel and SnapshotChannel.
  Incoming (sim-to-viewer): EventChannel and SnapshotChannel.

  Each direction uses its own channel instances, so a Snapshot type sent in both
  directions occupies separate per-type slots with no collision.
  """
    def __init__(self, *, s2v_snapshot: vp.SnapshotChannel, s2v_events: vp.EventChannel, v2s_events: vp.EventChannel, v2s_snapshot: vp.SnapshotChannel) -> None: ...
    def send_to_sim(self, message: vp.Message) -> None: ...
    def get_sim_events(self) -> list[vp.Event]:
        """Returns all pending events from the simulation."""
    def get_sim_snapshots(self) -> list[vp.Snapshot]:
        """Returns all pending latest snapshots from the simulation, one per type."""
    def close(self) -> None:
        """Close all channels owned by this endpoint (idempotent)."""

class SimEndpoint:
    """Sim-side endpoint that routes Messages to/from the Viewer.

  Outgoing (sim-to-viewer): EventChannel and SnapshotChannel.
  Incoming (viewer-to-sim): EventChannel and SnapshotChannel.

  Each direction uses its own channel instances, so a Snapshot type sent in both
  directions occupies separate per-type slots with no collision.
  """
    def __init__(self, *, s2v_snapshot: vp.SnapshotChannel, s2v_events: vp.EventChannel, v2s_events: vp.EventChannel, v2s_snapshot: vp.SnapshotChannel) -> None: ...
    def send_to_viewer(self, message: vp.Message) -> None: ...
    def get_viewer_events(self) -> list[vp.Event]:
        """Returns all pending events from the viewer."""
    def get_viewer_snapshots(self) -> list[vp.Snapshot]:
        """Returns all pending latest snapshots from the viewer, one per type."""
    def reclaim_to_viewer(self, snapshot_type: type[vp.Snapshot]) -> vp.Snapshot | None:
        """Reclaims an unconsumed outgoing snapshot of ``snapshot_type``, if any."""
    def close(self) -> None:
        """Close all channels owned by this endpoint."""

def make_endpoints(*, s2v_snapshot: vp.SnapshotChannel, s2v_events: vp.EventChannel, v2s_events: vp.EventChannel, v2s_snapshot: vp.SnapshotChannel) -> tuple[ViewerEndpoint, SimEndpoint]:
    """Returns viewer and simulation endpoints."""
