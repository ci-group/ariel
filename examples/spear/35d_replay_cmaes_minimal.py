"""Replay a minimal CMA-ES hover controller (trained by 35c_hover_cmaes_minimal.py)
in MuJoCo. Loads the 9-ish-dim param vector + blueprint and plays back the
torch dynamics with the same linear feedback law used during training.

Run:
    uv run examples/spear/35d_replay_cmaes_minimal.py \\
        --run-dir __data__/hover_cmaes_min/<TIMESTAMP> --view
"""

import argparse
import math
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
# Same dynamics + mixer as 35c. Kept inline for the same reason as 35b.
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


def _compute_u_hover(params, num_motors):
    k_w, k, w_min, w_max = params["k_w"], params["k"], params["w_min"], params["w_max"]
    W_hover = math.sqrt(GRAVITY / (k_w * num_motors))
    z = float(np.clip((W_hover - w_min) / (w_max - w_min), 0.0, 1.0))
    disc = (1.0 - k) ** 2 + 4.0 * k * z * z
    U_hover = (-(1.0 - k) + math.sqrt(max(disc, 0.0))) / (2.0 * k)
    return float(np.clip(2.0 * U_hover - 1.0, -1.0, 1.0))


def _tilt_mixer(propellers):
    # Sign convention must match 35c_hover_cmaes_minimal.py::_tilt_mixer.
    # Pitch column is NEGATIVE cos(phi): k_q_signed = +x·k_f/Iyy in the
    # dynamics, so a front motor (+x) with theta>0 (nose up in NED) needs
    # LESS thrust to recover. Roll column is +sin(phi) since k_p_signed
    # already absorbs the -y_i sign for roll. See docstring in 35c.
    mix = np.zeros((len(propellers), 2), dtype=np.float32)
    for i, p in enumerate(propellers):
        pos = np.asarray(p["loc"], dtype=np.float32)
        phi = math.atan2(float(pos[1]), float(pos[0]))
        mix[i, 0] = -math.cos(phi)   # pitch
        mix[i, 1] =  math.sin(phi)   # roll
    return mix


# ─────────────────────────────────────────────────────────────────────────────
# Load + roll out
# ─────────────────────────────────────────────────────────────────────────────

bp = DroneBlueprint.load_json(bp_path)
print(bp.summary())
propellers = blueprint_to_propellers(bp, convention="ned")
N = len(propellers)

theta = np.load(pr_path).astype(np.float32)
assert theta.shape[0] == N + 4, f"Param shape {theta.shape} doesn't match N={N}+4"
trim    = torch.tensor(theta[:N],     dtype=torch.float32)
k_alt_p = float(theta[N + 0])
k_alt_d = float(theta[N + 1])
k_tilt  = float(theta[N + 2])
k_rate  = float(theta[N + 3])
print(f"trim={theta[:N].round(3).tolist()}  k_alt_p={k_alt_p:.3f}  "
      f"k_alt_d={k_alt_d:.3f}  k_tilt={k_tilt:.3f}  k_rate={k_rate:.3f}")

params, _ = _build_params(propellers)
dev = torch.device(args.device)
dtype = torch.float32
dyn = _build_torch_dynamics(params, N, GRAVITY, dev, dtype)
u_hover = _compute_u_hover(params, N)
mix = torch.tensor(_tilt_mixer(propellers), device=dev, dtype=dtype)
target = torch.tensor(HOVER_TARGET_NED, device=dev, dtype=dtype)
trim = trim.to(dev)

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

    z_err      = s[:, 2:3] - target[2]
    vz         = s[:, 5:6]
    roll       = s[:, 6:7]
    pitch      = s[:, 7:8]
    roll_rate  = s[:, 9:10]
    pitch_rate = s[:, 10:11]
    alt_cmd  = k_alt_p * z_err - k_alt_d * vz
    att_cmd  = k_tilt * (mix[:, 0].unsqueeze(0) * pitch +
                         mix[:, 1].unsqueeze(0) * roll)
    rate_cmd = k_rate * (mix[:, 0].unsqueeze(0) * pitch_rate +
                         mix[:, 1].unsqueeze(0) * roll_rate)
    action = trim.unsqueeze(0) + alt_cmd + att_cmd + rate_cmd
    action = (u_hover + action.clamp(-1.0, 1.0) * 0.4).clamp(-1.0, 1.0)
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
    rec = VideoRecorder(file_name=out.stem, output_folder=out.parent,
                        width=720, height=540, fps=30)
    steps_per_frame = max(1, int(round(1.0 / (rec.fps * args.dt))))
    with mujoco.Renderer(m, width=rec.width, height=rec.height) as renderer:
        for idx in range(0, len(pos_enu), steps_per_frame):
            _set_pose(idx)
            renderer.update_scene(d)
            rec.write(frame=renderer.render())
    rec.release()
    print(f"Video → {out}")

print("Done.")
