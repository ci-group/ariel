"""P2 — Kaufmann et al. 2023, *Champion-level drone racing using deep reinforcement learning* (Nature, "Swift").

Ports two mechanisms from the paper:

1. **Random initialisation along the track** — kept as-is; the parent
   `TorchDroneGateEnv` already does this via `initialize_at_random_gates`.
   Confirmed in `42b_train_blueprint_traj.py` (random_init=True during
   training, False during eval).
2. **Track curriculum** — instead of training on ONE fixed track (42b),
   sample a fresh random smooth closed track *per worker* and *per
   episode*. This is Swift's "distribution of tracks" strategy: the
   policy has to learn gate-relative flying, not memorise a trajectory.

The track generator produces closed loops from random control points on a
ring, then reuses `42a_draw_trajectory.resample_track` /
`build_track_arrays` (imported via common — no duplication) to obtain
waypoint gates with tangent-yaw. Held-out tracks share the generator with
a different seed range.

Evaluation reports **waypoints/sec** (lap-time proxy) on K held-out
generated tracks AND on any user-drawn track from 42a.

Usage:
    uv run examples/spear/library/paper_impls/p2_kaufmann2023_swift.py --smoke
    uv run examples/spear/library/paper_impls/p2_kaufmann2023_swift.py \\
        --mode multi --steps 5_000_000 --num-envs 16
    uv run examples/spear/library/paper_impls/p2_kaufmann2023_swift.py \\
        --mode single   # single-fixed-track baseline for comparison
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from common import (
    dump_config, dump_results, load_42a, load_42b, new_out_dir,
)

_t42a = load_42a()
_t42b = load_42b()
BlueprintTrajEnv = _t42b.BlueprintTrajEnv


# ─────────────────────────────────────────────────────────────────────────────
# Random smooth closed-track generator
# ─────────────────────────────────────────────────────────────────────────────

def random_closed_track(rng: np.random.RandomState,
                        n_gates: int = 20,
                        n_control: int = 6,
                        base_radius: float = 2.5,
                        radial_jitter: float = 1.0,
                        altitude: float = 1.5) -> dict:
    """Sample a random closed track with EXACTLY `n_gates` waypoints.

    Control points are placed at evenly spaced angles with
    `r_i = base_radius + U(-jitter, jitter)`. Reuses
    `42a_draw_trajectory.resample_track` with `spacing = perimeter/n_gates`
    so the returned track has a fixed observation shape regardless of the
    random radii — the multi-track VecEnv needs this to swap tracks
    without rebuilding gate tensors.
    """
    angles = np.linspace(0.0, 2 * math.pi, n_control, endpoint=False)
    radii = base_radius + rng.uniform(-radial_jitter, radial_jitter, size=n_control)

    dense = np.linspace(0.0, 2 * math.pi, 400, endpoint=False)
    dr = np.interp(dense, angles, radii, period=2 * math.pi)
    raw_xy = np.stack([dr * np.cos(dense), dr * np.sin(dense)], axis=1)

    # Perimeter of the closed dense polyline determines spacing exactly.
    diffs = np.diff(np.concatenate([raw_xy, raw_xy[:1]]), axis=0)
    perimeter = float(np.linalg.norm(diffs, axis=1).sum())
    spacing = perimeter / n_gates

    wp, yaw = _t42a.resample_track(raw_xy, spacing, closed=True, smooth_window=9)
    # resample_track may return n_gates±1; force exactly n_gates by
    # trimming or wrapping.
    if len(wp) > n_gates:
        wp, yaw = wp[:n_gates], yaw[:n_gates]
    elif len(wp) < n_gates:
        pad = n_gates - len(wp)
        wp = np.concatenate([wp, wp[-1:].repeat(pad, axis=0)])
        yaw = np.concatenate([yaw, yaw[-1:].repeat(pad)])
    gpos, gyaw, spos = _t42a.build_track_arrays(wp, yaw, altitude)
    return {
        "gates_pos": gpos, "gate_yaw": gyaw, "start_pos": spos,
        "raw_xy": raw_xy.astype(np.float32),
        "altitude": float(altitude), "closed": True,
        "control_radii": radii.astype(np.float32),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-track worker: swaps the gate track on every episode reset
# ─────────────────────────────────────────────────────────────────────────────

class MultiTrackBlueprintEnv(BlueprintTrajEnv):
    """`BlueprintTrajEnv` that re-samples a fresh random track on every
    reset (including per-env resets inside `_reset_envs`). Each env-slot
    keeps ONE track for the length of an episode (so gate indexing / gate
    passing stays consistent within an episode); on `done` for that slot a
    new track is generated for that slot only.

    Implementation: keep a stack of per-slot tracks (gate_pos, gate_yaw,
    prev_gate) and rebuild the gate tensors whenever a done occurs. This
    is possible because the parent stores gate tensors as `self.gate_pos_t`
    etc., and the observation code reads from them on each step.
    """

    def __init__(self, *args, track_rng_seed: int = 0,
                 track_kwargs: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._track_rng = np.random.RandomState(int(track_rng_seed))
        self._track_kwargs = track_kwargs or {}
        # Track store: this env's construction used ONE track; we'll swap
        # it on the fly. Since gate_pos_t etc. are per-env-batch shared in
        # TorchDroneGateEnv, resampling means replacing the entire tensor
        # (all workers of THIS env instance share it). To make per-slot
        # tracks work we set num_envs=1 externally.
        if self.num_envs != 1:
            raise ValueError(
                "MultiTrackBlueprintEnv requires num_envs=1 (one worker per "
                "process). Use a `MultiTrackVecEnv` that stacks these."
            )

    def _sample_new_track(self) -> dict:
        return random_closed_track(self._track_rng, **self._track_kwargs)

    def _install_track(self, track: dict) -> None:
        """Replace gate tensors + precomputed relative offsets in place."""
        g = torch.tensor(track["gates_pos"], device=self.dev, dtype=self.dtype)
        y = torch.tensor(track["gate_yaw"],  device=self.dev, dtype=self.dtype)
        s = torch.tensor(track["start_pos"], device=self.dev, dtype=self.dtype)
        # If the number of gates changes, we need to rebuild size-dependent
        # buffers (num_gates, gate_pos_rel_t, obs slots). Keep it simple:
        # regenerate only if same G; else raise. Pin via `n_control`.
        if g.shape[0] != self.gate_pos_t.shape[0]:
            raise RuntimeError(
                f"track has {g.shape[0]} gates; env was built for "
                f"{self.gate_pos_t.shape[0]}. Fix `n_control` / `spacing`."
            )
        self.gate_pos_t = g
        self.gate_yaw_t = y
        self.start_pos_t = s
        # Recompute the gate-frame relative offsets used by the obs code.
        G = int(g.shape[0])
        gpr = np.zeros((G, 3), dtype=np.float32)
        gyr = np.zeros(G, dtype=np.float32)
        _gpos = track["gates_pos"]
        _gyaw = track["gate_yaw"]
        for i in range(G):
            gpr[i] = _gpos[i] - _gpos[i - 1]
            R2 = np.array([
                [ np.cos(_gyaw[i - 1]), np.sin(_gyaw[i - 1])],
                [-np.sin(_gyaw[i - 1]), np.cos(_gyaw[i - 1])],
            ])
            gpr[i, 0:2] = R2 @ gpr[i, 0:2]
            dy = _gyaw[i] - _gyaw[i - 1]
            gyr[i] = (dy + math.pi) % (2 * math.pi) - math.pi
        self.gate_pos_rel_t = torch.tensor(gpr, device=self.dev, dtype=self.dtype)
        self.gate_yaw_rel_t = torch.tensor(gyr, device=self.dev, dtype=self.dtype)

    def _reset_envs(self, mask: torch.Tensor) -> None:
        # Any reset (initial or per-episode done) swaps the track first,
        # then delegates to the parent for the actual state init.
        if bool(mask.any()):
            self._install_track(self._sample_new_track())
        super()._reset_envs(mask)


class MultiTrackVecEnv(VecEnv):
    """Stack of `MultiTrackBlueprintEnv` instances, one per worker slot.

    Delegates the SB3 VecEnv interface to per-slot 1-env instances so each
    slot has an independent track that resamples on episode boundaries.
    """

    def __init__(self, num_slots: int, track_kwargs: dict, seed: int = 0,
                 device: str = "cpu", max_steps: int = 1500,
                 gates_ahead: int = 2):
        self.num_slots = num_slots
        self.envs: list[MultiTrackBlueprintEnv] = []
        for i in range(num_slots):
            init_track = random_closed_track(
                np.random.RandomState(seed + 1_000 + i), **track_kwargs,
            )
            _bp, props = _t42b.build_blueprint_and_propellers()
            g = init_track["gates_pos"]
            margin = 3.0
            env = MultiTrackBlueprintEnv(
                num_envs=1,
                propellers=props,
                gates_pos=g,
                gate_yaw=init_track["gate_yaw"],
                start_pos=init_track["start_pos"],
                x_bounds=(-margin - abs(g[:, 0]).max(),
                           margin + abs(g[:, 0]).max()),
                y_bounds=(-margin - abs(g[:, 1]).max(),
                           margin + abs(g[:, 1]).max()),
                z_bounds=(-4.0, 0.5),
                gates_ahead=gates_ahead,
                initialize_at_random_gates=True,
                seed=seed + i,
                device=device,
                max_steps=max_steps,
                upright_bonus=_t42b.UPRIGHT_BONUS,
                extra_yaw_rate_pen=_t42b.EXTRA_YAW_RATE_PEN,
                velocity_reward_coef=_t42b.VELOCITY_REWARD_COEF,
                altitude_floor_z=_t42b.ALTITUDE_FLOOR_Z,
                altitude_floor_coef=_t42b.ALTITUDE_FLOOR_COEF,
                track_rng_seed=seed + i,
                track_kwargs=track_kwargs,
            )
            self.envs.append(env)
        obs_space = self.envs[0].observation_space
        act_space = self.envs[0].action_space
        VecEnv.__init__(self, num_slots, obs_space, act_space)
        self._actions_buffer = None

    def reset(self):
        obs_list = [e.reset()[0] for e in self.envs]
        return np.stack(obs_list, axis=0).astype(np.float32)

    def step_async(self, actions):
        self._actions_buffer = actions

    def step_wait(self):
        obs, rs, ds, infos = [], [], [], []
        for i, e in enumerate(self.envs):
            e.step_async(self._actions_buffer[i:i+1])
            o, r, d, info = e.step_wait()
            obs.append(o[0]); rs.append(float(r[0])); ds.append(bool(d[0]))
            infos.append(info[0])
        return (
            np.stack(obs, axis=0).astype(np.float32),
            np.asarray(rs, dtype=np.float32),
            np.asarray(ds, dtype=bool),
            infos,
        )

    def close(self): pass
    def seed(self, seed=None): return []
    def get_attr(self, attr_name, indices=None):
        return [getattr(e, attr_name, None) for e in self.envs]
    def set_attr(self, attr_name, value, indices=None): pass
    def env_method(self, method_name, *args, indices=None, **kwargs): return []
    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_slots
    def render(self, mode="human"): return {}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation on held-out tracks
# ─────────────────────────────────────────────────────────────────────────────

def eval_on_track(model, vecnorm_stats, track: dict, seed: int, device: str,
                  gates_ahead: int, max_steps: int, dt: float = 0.01) -> dict:
    """Deterministic single-episode rollout on ONE fixed track from
    `start_pos`. Reports waypoints/sec and completion fraction."""
    raw = _t42b.make_env(
        track, num_envs=1, seed=seed, device=device,
        gates_ahead=gates_ahead, max_steps=max_steps, random_init=False,
    )
    env = VecNormalize(raw, training=False, norm_obs=True, norm_reward=False,
                       clip_obs=10.0)
    env.obs_rms = vecnorm_stats
    obs = env.reset()
    total_r = 0.0
    steps = 0
    for t in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        env.step_async(action)
        obs, r, dones, infos = env.step_wait()
        total_r += float(r[0])
        steps = t + 1
        if bool(dones[0]):
            # step_counts is reset to 0 by _reset_envs before we see it;
            # use our own counter and pull the pre-reset gate count from info.
            gates = int(infos[0]["num_gates_passed"][0])
            break
    else:
        gates = int(raw.num_gates_passed[0])
    return {
        "gates_passed": gates,
        "num_gates": raw.num_gates,
        "completion": gates / max(raw.num_gates, 1),
        "steps": steps,
        "reward": total_r,
        "waypoints_per_sec": gates / max(steps * dt, 1e-9),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["single", "multi"], default="multi")
    p.add_argument("--n-gates", type=int, default=20,
                   help="EXACT number of waypoints per random track (fixed)")
    p.add_argument("--n-control", type=int, default=6,
                   help="radial control points sampled uniformly on the ring")
    p.add_argument("--base-radius", type=float, default=2.5)
    p.add_argument("--radial-jitter", type=float, default=1.0)
    p.add_argument("--altitude", type=float, default=1.5)
    p.add_argument("--steps", type=int, default=2_000_000)
    p.add_argument("--num-envs", type=int, default=10)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--gates-ahead", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--n-eval-tracks", type=int, default=5)
    p.add_argument("--user-track", type=Path, default=None,
                   help="optional path to a 42a trajectory.npz for extra eval")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.steps = 30_000
        args.num_envs = 4
        args.n_steps = 256
        args.max_steps = 400
        args.n_eval_tracks = 2

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    out_dir = new_out_dir(f"p2_kaufmann2023_swift", args.mode)
    dump_config(out_dir, vars(args))
    print(f"[P2] out_dir: {out_dir}  mode={args.mode}")

    track_kwargs = dict(
        n_gates=args.n_gates, n_control=args.n_control,
        base_radius=args.base_radius, radial_jitter=args.radial_jitter,
        altitude=args.altitude,
    )

    if args.mode == "multi":
        raw = MultiTrackVecEnv(
            num_slots=args.num_envs, track_kwargs=track_kwargs,
            seed=args.seed, device=args.device, max_steps=args.max_steps,
            gates_ahead=args.gates_ahead,
        )
    else:
        # single-fixed-track baseline: generate one and reuse across workers
        base_track = random_closed_track(
            np.random.RandomState(args.seed + 500), **track_kwargs,
        )
        raw = _t42b.make_env(
            base_track, num_envs=args.num_envs, seed=args.seed,
            device=args.device, gates_ahead=args.gates_ahead,
            max_steps=args.max_steps, random_init=True,
        )

    env = VecNormalize(raw, norm_obs=True, norm_reward=True,
                       clip_obs=10.0, clip_reward=10.0, gamma=0.99)
    model = PPO(
        "MlpPolicy", env,
        policy_kwargs=dict(net_arch=[256, 256]),
        n_steps=args.n_steps,
        batch_size=max((args.n_steps * args.num_envs) // 8, 64),
        n_epochs=10, gamma=0.99, gae_lambda=0.95,
        learning_rate=3e-4, clip_range=0.2, ent_coef=0.0,
        max_grad_norm=0.5, seed=args.seed, device=args.device, verbose=1,
    )
    callbacks = [
        CheckpointCallback(
            save_freq=max(args.n_steps, 250_000 // env.num_envs),
            save_path=str(out_dir / "checkpoints"),
            name_prefix="ppo", save_vecnormalize=True,
        ),
    ]
    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=False)
    elapsed = time.time() - t0
    print(f"[P2] trained {args.steps:,} steps in {elapsed:.0f}s")

    # ── Eval on N held-out tracks (fresh seeds, not overlapping training) ──
    eval_seed_base = args.seed + 100_000
    print(f"\n[P2] eval on {args.n_eval_tracks} held-out tracks:")
    per_track = []
    for i in range(args.n_eval_tracks):
        t_rng = np.random.RandomState(eval_seed_base + i)
        track = random_closed_track(t_rng, **track_kwargs)
        stats = eval_on_track(
            model, env.obs_rms, track, seed=eval_seed_base + i,
            device=args.device, gates_ahead=args.gates_ahead,
            max_steps=args.max_steps,
        )
        stats["track_idx"] = i
        per_track.append(stats)
        print(f"  track {i}: gates={stats['gates_passed']}/{stats['num_gates']} "
              f"({100*stats['completion']:.0f}%)  wp/s={stats['waypoints_per_sec']:.2f}  "
              f"r={stats['reward']:+.1f}")

    user_track_stats = None
    if args.user_track is not None and args.user_track.exists():
        user_track = _t42b.load_track(args.user_track)
        user_track_stats = eval_on_track(
            model, env.obs_rms, user_track, seed=args.seed + 999_999,
            device=args.device, gates_ahead=args.gates_ahead,
            max_steps=args.max_steps,
        )
        print(f"\n[P2] user track {args.user_track.name}: "
              f"gates={user_track_stats['gates_passed']}/{user_track_stats['num_gates']} "
              f"({100*user_track_stats['completion']:.0f}%)  "
              f"wp/s={user_track_stats['waypoints_per_sec']:.2f}")

    mean_comp = float(np.mean([r["completion"] for r in per_track]))
    mean_wps = float(np.mean([r["waypoints_per_sec"] for r in per_track]))
    print(f"\n[P2] mean completion on held-out={mean_comp:.2f}  "
          f"mean wp/s={mean_wps:.2f}")

    model.save(str(out_dir / "policy.zip"))
    env.save(str(out_dir / "vecnormalize.pkl"))
    dump_results(out_dir, {
        "mode": args.mode,
        "mean_completion": mean_comp,
        "mean_waypoints_per_sec": mean_wps,
        "per_track": per_track,
        "user_track": user_track_stats,
        "train_seconds": elapsed,
    })
    print(f"[P2] saved -> {out_dir}")


if __name__ == "__main__":
    main()
