"""Blueprint trajectory pipeline, step 1/3 — draw a trajectory on a scratchpad.

Opens a matplotlib canvas (top-down XY view, NED frame) where you freehand-draw
a flight path with the mouse. The drawing is resampled by arc length into a
sequence of waypoint "gates" whose yaw is the local path tangent — the standard
waypoint-track formulation used in RL drone racing (Song et al., "Autonomous
Drone Racing with Deep Reinforcement Learning", 2021; Kaufmann et al.,
"Champion-level drone racing using deep reinforcement learning", Nature 2023).
The drone flies the track at a constant altitude.

Controls (interactive mode):
  * drag        draw a stroke (multiple strokes are concatenated in order)
  * u           undo last stroke
  * c           clear everything
  * enter       resample -> preview waypoints + headings -> save and exit

Output NPZ (default __data__/blueprint_traj/trajectory.npz):
  raw_xy     (M, 2)  the drawn polyline, NED x/y
  gates_pos  (G, 3)  waypoints in NED (z = -altitude)
  gate_yaw   (G,)    heading at each waypoint (path tangent)
  start_pos  (3,)    1 m behind waypoint 0 along its tangent
  altitude, spacing, closed                  scalars/flags

Headless demo shapes (no GUI, useful for testing the downstream scripts):
    uv run examples/spear/library/42a_draw_trajectory.py --demo figure8

Then train with 42b_train_blueprint_traj.py and visualize with
42c_visualize_blueprint_traj.py.
"""

import argparse
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_OUT = Path(__file__).parents[3] / "__data__" / "blueprint_traj" / "trajectory.npz"


# ---------------------------------------------------------------------------
# Resampling: drawn polyline -> evenly spaced waypoints with tangent yaw
# ---------------------------------------------------------------------------

def _dedupe(points: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    keep = [0]
    for i in range(1, len(points)):
        if np.linalg.norm(points[i] - points[keep[-1]]) > eps:
            keep.append(i)
    return points[keep]


def _smooth(points: np.ndarray, window: int, closed: bool) -> np.ndarray:
    """Moving-average smoothing. Wraps for closed tracks, clamps for open."""
    if window <= 1 or len(points) < 3:
        return points
    k = window // 2
    if closed:
        padded = np.concatenate([points[-k:], points, points[:k]])
    else:
        padded = np.concatenate(
            [np.repeat(points[:1], k, axis=0), points,
             np.repeat(points[-1:], k, axis=0)]
        )
    kernel = np.ones(2 * k + 1) / (2 * k + 1)
    out = np.stack(
        [np.convolve(padded[:, d], kernel, mode="valid") for d in range(2)],
        axis=1,
    )
    return out


def resample_track(
    raw_xy: np.ndarray,
    spacing: float,
    closed: bool,
    smooth_window: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (waypoints (G,2), yaw (G,)) evenly spaced along the drawing.

    Yaw is the local tangent direction (central differences), which makes
    each waypoint a "gate" the drone should cross moving along the path —
    matching TorchDroneGateEnv's crossing test (projection onto the gate
    normal flips sign).
    """
    pts = _dedupe(np.asarray(raw_xy, dtype=np.float64))
    if len(pts) < 2:
        raise ValueError("need at least 2 distinct drawn points")
    pts = _smooth(pts, smooth_window, closed)
    if closed:
        pts = np.concatenate([pts, pts[:1]])

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total < 2 * spacing:
        raise ValueError(
            f"drawn path is only {total:.2f} m long; need >= {2 * spacing:.2f} m "
            f"(2x spacing). Draw a longer path or lower --spacing."
        )

    n_wp = max(int(round(total / spacing)), 2)
    # For a closed loop the last sample would coincide with the first.
    s_wp = (np.linspace(0.0, total, n_wp, endpoint=False) if closed
            else np.linspace(0.0, total, n_wp))
    wp = np.stack(
        [np.interp(s_wp, s, pts[:, d]) for d in range(2)], axis=1,
    )

    # Tangent yaw via central differences (wrap-around for closed loops).
    if closed:
        nxt = np.roll(wp, -1, axis=0)
        prv = np.roll(wp, 1, axis=0)
        d = nxt - prv
    else:
        d = np.empty_like(wp)
        d[1:-1] = wp[2:] - wp[:-2]
        d[0] = wp[1] - wp[0]
        d[-1] = wp[-1] - wp[-2]
    yaw = np.arctan2(d[:, 1], d[:, 0])
    return wp, yaw


def build_track_arrays(
    wp_xy: np.ndarray, yaw: np.ndarray, altitude: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(gates_pos NED, gate_yaw, start_pos) from 2D waypoints."""
    gates_pos = np.column_stack(
        [wp_xy, np.full(len(wp_xy), -abs(altitude))]
    ).astype(np.float32)
    gate_yaw = yaw.astype(np.float32)
    start_pos = (
        gates_pos[0]
        - np.array([math.cos(yaw[0]), math.sin(yaw[0]), 0.0]) * 1.0
    ).astype(np.float32)
    return gates_pos, gate_yaw, start_pos


def save_track(
    out: Path, raw_xy, gates_pos, gate_yaw, start_pos,
    altitude: float, spacing: float, closed: bool,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        raw_xy=np.asarray(raw_xy, dtype=np.float32),
        gates_pos=gates_pos,
        gate_yaw=gate_yaw,
        start_pos=start_pos,
        altitude=np.float32(altitude),
        spacing=np.float32(spacing),
        closed=np.bool_(closed),
    )
    length = float(
        np.linalg.norm(np.diff(gates_pos[:, :2], axis=0), axis=1).sum()
    )
    print(f"[draw] saved {len(gates_pos)} waypoints "
          f"(track ~{length:.1f} m, closed={closed}) -> {out}")


def plot_preview(ax, raw_xy, gates_pos, gate_yaw, start_pos, closed: bool):
    raw = np.asarray(raw_xy)
    ax.plot(raw[:, 0], raw[:, 1], "-", color="0.75", lw=1, label="drawing")
    wp = gates_pos[:, :2]
    loop = np.concatenate([wp, wp[:1]]) if closed else wp
    ax.plot(loop[:, 0], loop[:, 1], "o-", color="tab:orange", ms=4,
            lw=1.5, label="waypoints")
    ax.quiver(wp[:, 0], wp[:, 1], np.cos(gate_yaw), np.sin(gate_yaw),
              color="tab:blue", scale=25, width=3e-3, label="heading")
    ax.plot(*start_pos[:2], "s", color="tab:green", ms=9, label="start")
    ax.legend(loc="upper right", fontsize=8)


# ---------------------------------------------------------------------------
# Headless demo shapes (also used for pipeline smoke tests)
# ---------------------------------------------------------------------------

def demo_shape(name: str, scale: float = 2.5, n: int = 400) -> tuple[np.ndarray, bool]:
    t = np.linspace(0.0, 2.0 * math.pi, n)
    if name == "circle":
        return np.stack([scale * np.cos(t), scale * np.sin(t)], axis=1), True
    if name == "figure8":
        return np.stack(
            [scale * np.sin(t), scale * np.sin(t) * np.cos(t)], axis=1
        ), True
    if name == "slalom":
        x = np.linspace(-scale, scale, n)
        return np.stack([x, 0.8 * scale * np.sin(3.0 * x)], axis=1), False
    raise ValueError(f"unknown demo shape {name!r}")


# ---------------------------------------------------------------------------
# Interactive scratchpad
# ---------------------------------------------------------------------------

class Scratchpad:
    def __init__(self, args):
        self.args = args
        self.strokes: list[list[tuple[float, float]]] = []
        self._drawing = False
        self.saved = False

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        b = args.bounds
        self.ax.set_xlim(-b, b)
        self.ax.set_ylim(-b, b)
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel("x [m]  (NED north)")
        self.ax.set_ylabel("y [m]  (NED east)")
        self._set_title("draw the flight path")

        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def _set_title(self, msg: str):
        self.ax.set_title(
            f"Trajectory scratchpad — {msg}\n"
            "drag: draw   u: undo stroke   c: clear   enter: save"
        )
        self.fig.canvas.draw_idle()

    def all_points(self) -> np.ndarray:
        pts = [p for s in self.strokes for p in s]
        return np.asarray(pts, dtype=np.float64).reshape(-1, 2)

    def on_press(self, ev):
        if ev.inaxes is not self.ax or self.saved:
            return
        self._drawing = True
        self.strokes.append([(ev.xdata, ev.ydata)])
        (line,) = self.ax.plot(ev.xdata, ev.ydata, "-", color="k", lw=2)
        line.set_gid("stroke")
        self._cur_line = line

    def on_motion(self, ev):
        if not self._drawing or ev.inaxes is not self.ax:
            return
        self.strokes[-1].append((ev.xdata, ev.ydata))
        xs, ys = zip(*self.strokes[-1])
        self._cur_line.set_data(xs, ys)
        self.fig.canvas.draw_idle()

    def on_release(self, _ev):
        self._drawing = False

    def _stroke_lines(self):
        return [ln for ln in self.ax.lines if ln.get_gid() == "stroke"]

    def on_key(self, ev):
        if self.saved:
            return
        if ev.key == "c":
            self.strokes.clear()
            for ln in self._stroke_lines():
                ln.remove()
            self._set_title("cleared — draw the flight path")
        elif ev.key == "u" and self.strokes:
            self.strokes.pop()
            lines = self._stroke_lines()
            if lines:
                lines[-1].remove()
            self._set_title("stroke undone")
        elif ev.key == "enter":
            self.finish()

    def finish(self):
        pts = self.all_points()
        try:
            wp, yaw = resample_track(
                pts, self.args.spacing, self.args.closed, self.args.smooth,
            )
        except ValueError as e:
            self._set_title(f"ERROR: {e}")
            return
        gates_pos, gate_yaw, start_pos = build_track_arrays(
            wp, yaw, self.args.altitude,
        )
        save_track(
            self.args.out, pts, gates_pos, gate_yaw, start_pos,
            self.args.altitude, self.args.spacing, self.args.closed,
        )
        plot_preview(self.ax, pts, gates_pos, gate_yaw, start_pos,
                     self.args.closed)
        png = self.args.out.with_suffix(".png")
        self.fig.savefig(png, dpi=140)
        print(f"[draw] preview -> {png}")
        self.saved = True
        self._set_title(f"saved {len(gates_pos)} waypoints — close window")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--spacing", type=float, default=0.8,
                   help="arc-length distance between waypoints [m] "
                        "(gate_size in the env is 1.5 m)")
    p.add_argument("--altitude", type=float, default=1.5,
                   help="constant flight altitude [m] (NED z = -altitude)")
    p.add_argument("--closed", action="store_true",
                   help="treat the drawing as a closed loop (lap track)")
    p.add_argument("--smooth", type=int, default=7,
                   help="moving-average window over the raw drawing")
    p.add_argument("--bounds", type=float, default=4.0,
                   help="half-size of the drawing canvas [m]")
    p.add_argument("--demo", choices=["circle", "figure8", "slalom"],
                   default=None,
                   help="skip the GUI; generate this shape headlessly")
    args = p.parse_args()

    if args.demo is not None:
        matplotlib.use("Agg")
        raw, closed = demo_shape(args.demo)
        closed = closed or args.closed
        wp, yaw = resample_track(raw, args.spacing, closed, args.smooth)
        gates_pos, gate_yaw, start_pos = build_track_arrays(
            wp, yaw, args.altitude,
        )
        save_track(args.out, raw, gates_pos, gate_yaw, start_pos,
                   args.altitude, args.spacing, closed)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
        plot_preview(ax, raw, gates_pos, gate_yaw, start_pos, closed)
        ax.set_title(f"demo track: {args.demo}")
        png = args.out.with_suffix(".png")
        fig.savefig(png, dpi=140)
        print(f"[draw] preview -> {png}")
        return

    # matplotlib defaults to the non-interactive Agg backend in this env;
    # switch to a GUI backend or the canvas never appears.
    if matplotlib.get_backend().lower().startswith("agg"):
        for backend in ("QtAgg", "TkAgg", "GTK4Agg"):
            try:
                plt.switch_backend(backend)
                break
            except ImportError:
                continue
        else:
            raise SystemExit(
                "no interactive matplotlib backend available (tried TkAgg, "
                "QtAgg, GTK4Agg). Install tkinter (python3-tk) or use --demo."
            )

    pad = Scratchpad(args)
    plt.show()
    if not pad.saved:
        print("[draw] window closed without saving (press enter to save)")


if __name__ == "__main__":
    main()
