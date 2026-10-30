"""Blueprint trajectory pipeline, step 2/3 — train PPO to fly a drawn track.

Loads the waypoint track produced by 42a_draw_trajectory.py and trains a PPO
policy on the canonical blueprint hex (same morphology as the 40-series
scripts: even 60-degree azimuths, planar arms, vertical motors) to fly it.

Formulation — waypoint gates + progress reward, following the RL drone-racing
literature:
  * Song et al., "Autonomous Drone Racing with Deep Reinforcement Learning"
    (IROS 2021): reward is progress toward the next gate (d_old - d_new) plus
    a bonus on gate passage; observation contains the next few gates relative
    to the body. TorchDroneGateEnv implements exactly this, so the drawn
    waypoints are fed in as a gate track (yaw = path tangent).
  * Kaufmann et al., "Champion-level drone racing using deep reinforcement
    learning" (Nature 2023): same progress formulation, initialization at
    random gates along the track for uniform state coverage — we keep the
    env's `initialize_at_random_gates=True` during training and evaluate
    from the fixed start.
  * Shaping terms (upright bonus, yaw-rate penalty, velocity-toward-gate,
    altitude floor) reuse the values tuned for the trajectory tasks in
    envs/residual_drone_env.py.

Run:
    uv run examples/spear/library/42a_draw_trajectory.py            # draw first
    uv run examples/spear/library/42b_train_blueprint_traj.py \\
        --steps 5_000_000 --num-envs 16

Outputs (default __data__/blueprint_traj/runs/<timestamp>/):
    policy.zip, vecnormalize.pkl, trajectory.npz (copy), config.json

Visualize with 42c_visualize_blueprint_traj.py.
"""

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from ariel.body_phenotypes.drone.backends import (
    blueprint_to_mjspec,  # noqa: F401 — re-exported for 42c
    blueprint_to_propellers,
)
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
from ariel.simulation.tasks.torch_drone_gate_env import TorchDroneGateEnv

# Canonical hex constants — match 40_morph_break_analysis.py.
N_MOTORS = 6
ARM_LENGTH = 0.15
CORE_MASS = 0.5
PROP_SIZE = 2

# Shaping constants tuned for the trajectory tasks in
# envs/residual_drone_env.py (autoresearch loop).
UPRIGHT_BONUS = 0.002
EXTRA_YAW_RATE_PEN = 0.005
VELOCITY_REWARD_COEF = 0.03
ALTITUDE_FLOOR_Z = -0.5
ALTITUDE_FLOOR_COEF = 0.5

DATA_ROOT = Path(__file__).parents[3] / "__data__" / "blueprint_traj"
DEFAULT_TRAJ = DATA_ROOT / "trajectory.npz"


def canonical_genome() -> np.ndarray:
    """Regular hex: even azimuths, planar arms, vertical motors, 3ccw+3cw."""
    az = np.radians(np.arange(N_MOTORS) * 60.0).astype(np.float32)
    g = np.zeros((N_MOTORS, 6), dtype=np.float32)
    g[:, 0] = ARM_LENGTH
    g[:, 1] = az
    g[:, 2] = 0.0
    g[:, 3] = az
    g[:, 4] = 0.0
    g[:, 5] = np.array([0, 1, 0, 1, 0, 1], dtype=np.float32)
    return g


def build_blueprint_and_propellers():
    bp = spherical_angular_to_blueprint(
        canonical_genome(), core_mass=CORE_MASS, propsize=PROP_SIZE,
    )
    props = blueprint_to_propellers(bp, convention="ned")
    return bp, props


class BlueprintTrajEnv(TorchDroneGateEnv):
    """TorchDroneGateEnv with hover-equivalent motor-state initialization.

    The parent resets motor state w to 0 (deterministic start) or U[-1, 1]
    (random-gate start). w is *normalized* rotor speed: w=0 is mid-throttle,
    several times hover thrust for this hex, which can flip the drone before
    the policy acts. Reset motors to the hover-equivalent value instead
    (same fix as ResidualDroneEnv / 36_build_hover_library).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        p = self.drone_sim.params
        w_lo, w_hi = float(p["w_min"]), float(p["w_max"])
        W_hover = math.sqrt(self.drone_sim.g / (p["k_w"] * self.num_motors))
        self._w_hover_norm = float(
            (2.0 * W_hover - (w_hi + w_lo)) / max(w_hi - w_lo, 1e-6)
        )

    def _reset_envs(self, mask: torch.Tensor) -> None:
        super()._reset_envs(mask)
        if mask.any():
            self.world_states[mask, 12:12 + self.num_motors] = self._w_hover_norm


def load_track(path: Path) -> dict:
    d = np.load(path)
    return {
        "gates_pos": d["gates_pos"].astype(np.float32),
        "gate_yaw": d["gate_yaw"].astype(np.float32),
        "start_pos": d["start_pos"].astype(np.float32),
        "raw_xy": d["raw_xy"].astype(np.float32),
        "altitude": float(d["altitude"]),
        "closed": bool(d["closed"]),
    }


def make_env(track: dict, num_envs: int, seed: int, device: str,
             gates_ahead: int, max_steps: int,
             random_init: bool) -> BlueprintTrajEnv:
    _bp, props = build_blueprint_and_propellers()
    g = track["gates_pos"]
    margin = 3.0
    x_bounds = (float(g[:, 0].min() - margin), float(g[:, 0].max() + margin))
    y_bounds = (float(g[:, 1].min() - margin), float(g[:, 1].max() + margin))
    return BlueprintTrajEnv(
        num_envs=num_envs,
        propellers=props,
        gates_pos=g,
        gate_yaw=track["gate_yaw"],
        start_pos=track["start_pos"],
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z_bounds=(-4.0, 0.5),
        gates_ahead=gates_ahead,
        initialize_at_random_gates=random_init,
        seed=seed,
        device=device,
        max_steps=max_steps,
        upright_bonus=UPRIGHT_BONUS,
        extra_yaw_rate_pen=EXTRA_YAW_RATE_PEN,
        velocity_reward_coef=VELOCITY_REWARD_COEF,
        altitude_floor_z=ALTITUDE_FLOOR_Z,
        altitude_floor_coef=ALTITUDE_FLOOR_COEF,
    )


def evaluate_from_start(model, vecnorm, track: dict, seed: int, device: str,
                        gates_ahead: int, max_steps: int) -> dict:
    """One deterministic episode from the fixed start; returns track stats."""
    from stable_baselines3.common.vec_env import VecNormalize

    raw = make_env(track, num_envs=1, seed=seed, device=device,
                   gates_ahead=gates_ahead, max_steps=max_steps,
                   random_init=False)
    env = VecNormalize(raw, training=False, norm_obs=vecnorm.norm_obs,
                       norm_reward=False, clip_obs=vecnorm.clip_obs)
    env.obs_rms = vecnorm.obs_rms

    obs = env.reset()
    total_r, gates, steps = 0.0, 0, 0
    pos_trace = []
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        env.step_async(action)
        obs, r, dones, infos = env.step_wait()
        total_r += float(r[0])
        steps += 1
        pos_trace.append(raw.world_states[0, 0:3].cpu().numpy().copy())
        if bool(dones[0]):
            # snapshot taken before the auto-reset inside step_wait
            gates = int(infos[0]["num_gates_passed"][0])
            break
    else:
        gates = int(raw.num_gates_passed[0])
    n_g = len(track["gates_pos"])
    return {
        "gates_passed": gates,
        "num_gates": n_g,
        "completion": gates / n_g,
        "steps": steps,
        "reward": total_r,
        "pos_trace": np.asarray(pos_trace),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJ)
    p.add_argument("--steps", type=int, default=5_000_000)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--gates-ahead", type=int, default=2,
                   help="future waypoints visible in the observation")
    p.add_argument("--max-steps", type=int, default=1500,
                   help="episode length (dt=0.01 -> 15 s)")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", type=Path,
                   default=DATA_ROOT / "runs" / time.strftime("%Y%m%d_%H%M%S"))
    args = p.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import VecNormalize

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.trajectory.exists():
        raise SystemExit(
            f"trajectory file missing: {args.trajectory}\n"
            "Draw one first:  uv run examples/spear/library/42a_draw_trajectory.py"
        )
    track = load_track(args.trajectory)
    length = float(np.linalg.norm(
        np.diff(track["gates_pos"][:, :2], axis=0), axis=1).sum())
    print(f"[train] track: {len(track['gates_pos'])} waypoints, "
          f"~{length:.1f} m, altitude={track['altitude']:.1f} m, "
          f"closed={track['closed']}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.trajectory, out_dir / "trajectory.npz")
    (out_dir / "config.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v
         for k, v in vars(args).items()}, indent=2))

    raw = make_env(track, num_envs=args.num_envs, seed=args.seed,
                   device=args.device, gates_ahead=args.gates_ahead,
                   max_steps=args.max_steps, random_init=True)
    print(f"[train] canonical hex: mass={raw.drone_sim.mass:.3f} kg  "
          f"obs_dim={raw.observation_space.shape[0]}  "
          f"w_hover_norm={raw._w_hover_norm:+.3f}")
    env = VecNormalize(raw, norm_obs=True, norm_reward=True,
                       clip_obs=10.0, clip_reward=10.0, gamma=args.gamma)

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        n_steps=args.n_steps,
        batch_size=max((args.n_steps * args.num_envs) // 8, 64),
        n_epochs=10,
        gamma=args.gamma,
        gae_lambda=0.95,
        learning_rate=args.lr,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        max_grad_norm=0.5,
        seed=args.seed,
        device=args.device,
        verbose=1,
    )

    ckpt_freq = max(args.n_steps, 250_000 // args.num_envs)
    t0 = time.time()
    model.learn(
        total_timesteps=args.steps,
        callback=CheckpointCallback(
            save_freq=ckpt_freq, save_path=str(out_dir / "checkpoints"),
            name_prefix="ppo", save_vecnormalize=True,
        ),
    )
    elapsed = time.time() - t0
    print(f"[train] {args.steps:,} steps in {elapsed:.0f}s "
          f"({args.steps / max(elapsed, 1e-9):.0f} sps)")

    model.save(str(out_dir / "policy.zip"))
    env.save(str(out_dir / "vecnormalize.pkl"))

    stats = evaluate_from_start(
        model, env, track, seed=123, device=args.device,
        gates_ahead=args.gates_ahead, max_steps=args.max_steps,
    )
    print(f"[eval] from fixed start: {stats['gates_passed']}/{stats['num_gates']} "
          f"waypoints ({100 * stats['completion']:.0f}%)  "
          f"steps={stats['steps']}  reward={stats['reward']:+.1f}")
    print(f"[train] saved -> {out_dir}")


if __name__ == "__main__":
    main()
