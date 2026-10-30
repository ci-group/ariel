"""Replay a minimal CMA-ES hover controller (trained by 35c_hover_cmaes_minimal.py)
in MuJoCo. Loads the 9-ish-dim param vector + blueprint and plays back the
torch dynamics with the same linear feedback law used during training.

Run:
    uv run examples/spear/35d_replay_cmaes_minimal.py \\
        --run-dir __data__/hover_cmaes_min/<TIMESTAMP> --view
"""

import argparse
import sys
import time as _time
from pathlib import Path

import mujoco
import numpy as np
import torch

from ariel.body_phenotypes.drone.backends import blueprint_to_mjspec, blueprint_to_propellers
from ariel.body_phenotypes.drone.blueprint import DroneBlueprint
from ariel.simulation.drone.drone_configuration import DroneConfiguration
from ariel.simulation.drone.dynamics_params import derive_reference_params
from ariel.simulation.environments import SimpleFlatWorld
from ariel.simulation.tasks.torch_drone_gate_env import _build_torch_dynamics
from ariel.utils.video_recorder import VideoRecorder

# Canonical hover prior (same instance type 35c uses for training).
sys.path.insert(0, str(Path(__file__).parent / "library"))
from prior_controller import HoverPrior, N_GAINS  # noqa: E402

HOVER_TARGET_NED = np.array([0.0, 0.0, -1.5], dtype=np.float32)
GRAVITY = 9.81

parser = argparse.ArgumentParser(description="Replay minimal CMA-ES hover in MuJoCo")
parser.add_argument("--run-dir", required=True,
                    help="Directory from 35c_hover_cmaes_minimal.py "
                         "(blueprint.json + best_params.npy)")
parser.add_argument("--rollout-time", type=float, default=10.0)
parser.add_argument("--dt", type=float, default=0.01)
parser.add_argument("--device", default="cpu")
parser.add_argument("--deterministic-init", action="store_true",
                    help="Start exactly at target (no init perturbation)")
parser.add_argument("--out", default=None)
parser.add_argument("--view", action="store_true")
# Video quality knobs.  Defaults give HD@60fps with H.264 — visually a big
# step up from the previous 720×540 / 30fps / mp4v output.  Use --width
# 1920 --height 1080 for full-HD; drop to 1280×720 if disk space matters.
parser.add_argument("--width", type=int, default=1280, help="Video width (px)")
parser.add_argument("--height", type=int, default=720, help="Video height (px)")
parser.add_argument("--fps", type=int, default=60, help="Video framerate")
parser.add_argument("--video-quality", type=int, default=9,
                    help="imageio quality 0–10 (only used if imageio-ffmpeg available)")
parser.add_argument("--camera", choices=["follow", "orbit", "fixed"],
                    default="follow",
                    help="follow=track drone, orbit=slowly circle, fixed=static")
args = parser.parse_args()

run_dir = Path(args.run_dir)
bp_path = run_dir / "blueprint.json"
pr_path = run_dir / "best_params.npy"
if not bp_path.exists() or not pr_path.exists():
    raise FileNotFoundError(f"Need {bp_path} and {pr_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fitness curve plot — if 35c saved a fitness.txt, render it next to the
# replay artifacts. Mean ± std band + per-gen max + best-ever overlay.
# ─────────────────────────────────────────────────────────────────────────────

def _plot_fitness(fit_path: Path, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")   # headless-safe; we save, never show interactively
    import matplotlib.pyplot as plt

    data = np.loadtxt(fit_path, delimiter="\t")
    if data.ndim == 1:
        data = data[None, :]
    gen, gmax, gmed, gmean, gstd, gmin, gbest = data.T

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    ax.fill_between(gen, gmean - gstd, gmean + gstd,
                    alpha=0.22, color="C0", label="mean ± std")
    ax.plot(gen, gmean, color="C0", lw=1.4, label="mean")
    ax.plot(gen, gmax,  color="C1", lw=1.0, alpha=0.8, label="per-gen max")
    ax.plot(gen, gbest, color="C3", lw=1.6, label="best-ever")
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness  (cumulative dense reward)")
    ax.set_title(f"CMA-ES hover training — {fit_path.parent.name}")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


fit_path = run_dir / "fitness.txt"
if fit_path.exists():
    plot_out = run_dir / "fitness_curve.png"
    _plot_fitness(fit_path, plot_out)
    print(f"Fitness plot → {plot_out}")
else:
    print(f"(no fitness.txt at {fit_path} — skipping plot)")


# ─────────────────────────────────────────────────────────────────────────────
# Build params; the mixer and u_hover live in `prior_controller.HoverPrior`.
# ─────────────────────────────────────────────────────────────────────────────

def _build_params(propellers):
    cfg = DroneConfiguration(propellers)
    params = derive_reference_params(
        propellers=cfg.propellers, mass=float(cfg.mass),
        inertia=np.asarray(cfg.inertia_matrix),
        prop_size=propellers[0].get("propsize", 2),
        gravity=GRAVITY,
    )
    return params, cfg.num_motors


# ─────────────────────────────────────────────────────────────────────────────
# Load + roll out
# ─────────────────────────────────────────────────────────────────────────────

bp = DroneBlueprint.load_json(bp_path)
print(bp.summary())
propellers = blueprint_to_propellers(bp, convention="ned")
N = len(propellers)

params, _ = _build_params(propellers)
dev = torch.device(args.device)
dtype = torch.float32
dyn = _build_torch_dynamics(params, N, GRAVITY, dev, dtype)
prior = HoverPrior(
    propellers=propellers, params=params,
    target_ned=HOVER_TARGET_NED.tolist(),
    gravity=GRAVITY, action_scale=0.4,
    device=dev, dtype=dtype,
)
target = prior.target

theta = np.load(pr_path).astype(np.float32)
# Backwards compat: legacy runs saved N+4 params (no k_yaw_rate). Newer
# runs save N+5. Pad with k_yaw_rate=0 if loading a legacy file so the
# replay still works (drone will yaw-drift as before).
if theta.shape[0] == N + 4:
    print("Legacy params (N+4) — padding k_yaw_rate=0 for backward-compat.")
    theta = np.concatenate([theta, np.zeros(1, dtype=np.float32)])
assert theta.shape[0] == prior.param_dim, (
    f"Param shape {theta.shape} doesn't match expected ({prior.param_dim},) "
    f"for N={N}"
)
print(f"trim={theta[:N].round(3).tolist()}  "
      f"k_alt_p={theta[N+0]:+.3f}  k_alt_d={theta[N+1]:+.3f}  "
      f"k_tilt={theta[N+2]:+.3f}  k_rate={theta[N+3]:+.3f}  "
      f"k_yaw_rate={theta[N+4]:+.3f}")

params_t = torch.tensor(theta, device=dev, dtype=dtype)

# Initial state (1 env)
state_dim = 12 + N
s = torch.zeros((1, state_dim), device=dev, dtype=dtype)
s[0, 0:3] = target
if not args.deterministic_init:
    s[0, 0:3] += (torch.rand(3, device=dev, dtype=dtype) - 0.5) * 0.2
    s[0, 6:8] = (torch.rand(2, device=dev, dtype=dtype) - 0.5) * 0.1

steps = int(args.rollout_time / args.dt) + 1
pos_ned   = np.zeros((steps, 3), dtype=np.float32)
euler_log = np.zeros((steps, 3), dtype=np.float32)
act_log   = np.zeros((steps, N), dtype=np.float32)

for i in range(steps):
    pos_ned[i]   = s[0, 0:3].cpu().numpy()
    euler_log[i] = s[0, 6:9].cpu().numpy()

    action = prior.prior_action(s, params_t.unsqueeze(0))
    act_log[i] = action[0].cpu().numpy()

    sd = dyn(s.T, action.T).T
    s = s + args.dt * sd

deg = np.degrees
print(
    f"\nRollout {steps} steps ({steps*args.dt:.1f}s)\n"
    f"  Alt (NED z): [{pos_ned[:,2].min():.2f}, {pos_ned[:,2].max():.2f}]  target={HOVER_TARGET_NED[2]:.2f}\n"
    f"  XY drift:    x=[{pos_ned[:,0].min():.2f},{pos_ned[:,0].max():.2f}]  "
    f"y=[{pos_ned[:,1].min():.2f},{pos_ned[:,1].max():.2f}]\n"
    f"  Tilt  (deg): roll [{deg(euler_log[:,0]).min():.1f}, {deg(euler_log[:,0]).max():.1f}]  "
    f"pitch [{deg(euler_log[:,1]).min():.1f}, {deg(euler_log[:,1]).max():.1f}]\n"
    f"  Actions mean: {act_log.mean(axis=0).round(3)}"
)

# NED → ENU + euler→quat (same conversion as 35b / 17)
pos_enu = pos_ned.copy()
pos_enu[:, 2] = -pos_enu[:, 2]
phi, theta_e, psi = euler_log[:, 0], euler_log[:, 1], euler_log[:, 2]
cy, sy = np.cos(psi / 2), np.sin(psi / 2)
cp, sp = np.cos(theta_e / 2), np.sin(theta_e / 2)
cr, sr = np.cos(phi / 2), np.sin(phi / 2)
quat_enu = np.stack([
    cr * cp * cy + sr * sp * sy,
    sr * cp * cy - cr * sp * sy,
    -(cr * sp * cy + sr * cp * sy),
    cr * cp * sy - sr * sp * cy,
], axis=-1)

# ─────────────────────────────────────────────────────────────────────────────
# MuJoCo scene
# ─────────────────────────────────────────────────────────────────────────────

arm_lengths = [bp.payload(a).length for a in bp.children(bp.root_id)]  # type: ignore[arg-type]
mean_arm_len = float(np.mean(arm_lengths)) if arm_lengths else 0.06
drone_spec = blueprint_to_mjspec(
    bp, motor_mass=0.01, arm_mass=0.034 * mean_arm_len, body_name="hover_quad",
)
world = SimpleFlatWorld()
world.spawn(drone_spec,
            position=(float(pos_enu[0, 0]), float(pos_enu[0, 1]), float(pos_enu[0, 2])),
            correct_collision_with_floor=False)
target_body = world.spec.worldbody.add_body(name="hover_target",
                                            pos=[0.0, 0.0, -float(HOVER_TARGET_NED[2])])
target_body.add_geom(name="hover_target_geom",
                     type=mujoco.mjtGeom.mjGEOM_SPHERE,
                     size=[0.06, 0.0, 0.0], rgba=(0.1, 0.9, 0.1, 0.8),
                     contype=0, conaffinity=0)
m = world.spec.compile()
d = mujoco.MjData(m)
m.opt.timestep = args.dt


def _set_pose(idx):
    p, q = pos_enu[idx], quat_enu[idx]
    d.qpos[0:3] = [float(p[0]), float(p[1]), float(p[2])]
    d.qpos[3:7] = [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)


if args.view:
    import mujoco.viewer
    print("Launching MuJoCo viewer …")
    _set_pose(0)
    with mujoco.viewer.launch_passive(m, d) as viewer:
        idx = 0
        while viewer.is_running() and idx < len(pos_enu):
            t0 = _time.time()
            _set_pose(idx)
            viewer.sync()
            slack = args.dt - (_time.time() - t0)
            if slack > 0:
                _time.sleep(slack)
            idx += 1
else:
    out = Path(args.out) if args.out else (run_dir / "replay.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    W, H, FPS = args.width, args.height, args.fps

    # ── Camera ──────────────────────────────────────────────────────────
    # MjvCamera in FREE mode tracking the drone gives much better framing
    # than the default model camera (which is usually static at the world
    # origin and squashes everything to a corner).
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 3.0          # metres from look-at point
    cam.elevation = -20.0       # negative tilts down toward subject
    cam.azimuth = 45.0          # initial heading (deg); orbit mode rotates this
    cam.lookat[:] = [0.0, 0.0, -float(HOVER_TARGET_NED[2])]

    # ── Float-accurate frame timing ─────────────────────────────────────
    # Old code did int(1 / (fps·dt)) frames-per-step which silently drops
    # frames and produces visible stutter when fps·dt isn't an integer
    # divisor (e.g. 30·0.01 = 0.3 → rounds to 3 sim-steps per frame → real
    # playback is 33fps not 30fps).  Resample by sim-time instead.
    total_time = (len(pos_enu) - 1) * args.dt
    n_frames = max(1, int(round(total_time * FPS)) + 1)
    frame_times = np.linspace(0.0, total_time, n_frames)

    print(f"Rendering {n_frames} frames @ {W}×{H} {FPS}fps "
          f"(camera={args.camera}) …")

    frames: list[np.ndarray] = []
    with mujoco.Renderer(m, width=W, height=H) as renderer:
        for f_idx, t in enumerate(frame_times):
            # Nearest sim sample for this video frame (no interp — at 60fps
            # with dt=0.01 the gap is at most one sim step, ~10 ms, which
            # is visually imperceptible for hover).
            sim_idx = min(int(round(t / args.dt)), len(pos_enu) - 1)
            _set_pose(sim_idx)

            if args.camera == "follow":
                cam.lookat[:] = [float(pos_enu[sim_idx, 0]),
                                 float(pos_enu[sim_idx, 1]),
                                 float(pos_enu[sim_idx, 2])]
            elif args.camera == "orbit":
                cam.lookat[:] = [float(pos_enu[sim_idx, 0]),
                                 float(pos_enu[sim_idx, 1]),
                                 float(pos_enu[sim_idx, 2])]
                cam.azimuth = 45.0 + 20.0 * t   # slow 20°/s orbit
            # "fixed" leaves cam alone

            renderer.update_scene(d, camera=cam)
            frames.append(renderer.render())

    # ── Encode ──────────────────────────────────────────────────────────
    # Prefer imageio-ffmpeg + libx264: dramatically better visual quality
    # than OpenCV's mp4v at the same file size.  Fall back to the project's
    # VideoRecorder (mp4v / mp4) if imageio-ffmpeg isn't installed.
    encoded = False
    try:
        import imageio.v3 as iio   # type: ignore[import-not-found]
        iio.imwrite(
            str(out),
            np.stack(frames, axis=0),
            fps=FPS,
            codec="libx264",
            quality=args.video_quality,         # 0–10, higher = better
            pixelformat="yuv420p",              # broadly compatible (QuickTime, browsers)
            macro_block_size=1,                 # don't pad dimensions
            ffmpeg_log_level="error",
        )
        encoded = True
        print(f"Video (libx264 q={args.video_quality}) → {out}")
    except ImportError:
        print("imageio-ffmpeg not installed; falling back to OpenCV/mp4v "
              "(install with: uv pip install imageio[ffmpeg] for better quality)")
    except Exception as e:
        print(f"imageio encode failed ({e!s}); falling back to OpenCV/mp4v")

    if not encoded:
        rec = VideoRecorder(file_name=out.stem, output_folder=out.parent,
                            width=W, height=H, fps=FPS)
        for f in frames:
            rec.write(frame=f)
        rec.release()
        print(f"Video (mp4v) → {out}")

print("Done.")
