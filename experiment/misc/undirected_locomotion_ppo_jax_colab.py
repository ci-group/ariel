"""Undirected locomotion for the ARIEL ``insect_small`` morphology — Colab edition.

Standalone script that trains a PPO policy with mujoco-playground + brax on
a Colab GPU. Handles its own dependency installation (ariel from GitHub,
mujoco pinned to 3.6.0 for mjx compatibility, brax, mujoco_playground,
mediapy).

Usage
-----
In a Colab notebook, drop this file in and run::

    !python undirected_locomotion_ppo_jax_colab.py --num-timesteps 20000000

Or paste the content into a single cell and run it. The first run spends
~2 min on pip installs, then starts training on the GPU.

Tested on
---------
- Colab free (T4, 16 GB VRAM)
- Colab Pro / Pro+ (L4, A100)

If you OOM on T4, drop ``--num-envs`` down to 4096 or 2048.
"""

from __future__ import annotations

import os
import subprocess
import sys

# ============================================================================ #
#                       Dependency bootstrap (Colab)                           #
# ============================================================================ #
_REQUIRED = {
    "mujoco": "3.6.0",
    "mujoco-mjx": "3.6.0",
    "brax": None,  # any version
    "mujoco_playground": None,
    "mediapy": None,
    "ml_collections": None,
}


def _pip_install(*pkgs: str) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "-q", *pkgs]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd)


def _ensure_deps() -> None:
    """Install anything missing. Idempotent; safe to re-run."""
    try:
        import mujoco  # noqa: F401
        import brax  # noqa: F401
        import mujoco_playground  # noqa: F401
        import mediapy  # noqa: F401
        import ariel  # noqa: F401

        # Verify mujoco is pinned to 3.6.0 (mjx/warp compat requirement).
        import mujoco as _mj
        if _mj.__version__ != "3.6.0":
            raise ImportError(f"mujoco=={_mj.__version__}, need 3.6.0")
        return
    except ImportError:
        pass

    print("Installing deps (one-time; ~2 min)…", flush=True)
    _pip_install("mujoco==3.6.0", "mujoco-mjx==3.6.0", "warp-lang==1.17.0")
    _pip_install("brax", "mujoco_playground", "mediapy", "ml_collections")
    _pip_install(
        "git+https://github.com/ci-group/ariel.git@guszti-kevin-aron"
    )
    # mujoco pin can get bumped by transitive deps — re-pin:
    _pip_install("mujoco==3.6.0", "mujoco-mjx==3.6.0", "warp-lang==1.17.0")


_ensure_deps()

# ============================================================================ #
#                       XLA / MuJoCo rendering config                          #
# ============================================================================ #
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Async malloc reduces GPU memory fragmentation (recommended by XLA for OOMs).
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("MUJOCO_GL", "egl")

# ============================================================================ #
#                                Imports                                       #
# ============================================================================ #
import argparse
import functools
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import mediapy as media
import mujoco
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground import wrapper
from mujoco_playground._src import locomotion, mjx_env

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.insect import insect_small
from ariel.simulation.environments import SimpleFlatWorld


ENV_NAME = "ArielInsectUndirectedLocomotion"
TORSO_BODY_NAME = "robot1_core"


# ============================================================================ #
#                              MuJoCo model build                              #
# ============================================================================ #
def _build_insect_mj_model(sim_dt: float) -> mujoco.MjModel:
    """Compile the ariel insect spawned on a flat world."""
    world = SimpleFlatWorld(load_precompiled=False)
    body = insect_small()
    world.spawn(body.spec, position=[0, 0, 0.1])

    def _find_body(parent, name):
        for b in getattr(parent, "bodies", []):
            if b.name == name:
                return b
            found = _find_body(b, name)
            if found is not None:
                return found
        return None

    core_body = _find_body(world.spec.worldbody, TORSO_BODY_NAME)
    if core_body is not None:
        core_body.add_camera(
            name="track",
            mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM,
            pos=[0.0, -1.4, 0.9],
            xyaxes=[1, 0, 0, 0, 0.6, 0.8],
        )
    else:
        world.spec.worldbody.add_camera(
            name="track",
            pos=[0.0, -1.4, 0.9],
            xyaxes=[1, 0, 0, 0, 0.6, 0.8],
        )

    mj_model = world.spec.compile()
    mj_model.opt.timestep = sim_dt
    # Playground-style small solver (CPU defaults blow up MJX JIT).
    mj_model.opt.iterations = 4
    mj_model.opt.ls_iterations = 8
    mj_model.vis.global_.offwidth = 1280
    mj_model.vis.global_.offheight = 720

    # Contact filtering: robot<->floor only, no self-collisions. ariel modules
    # default to contype=conaffinity=1 so every geom-pair is a collision
    # candidate; whitelisting floor-vs-robot speeds up MJX ~60×.
    floor_id = mj_model.geom("floor").id
    for i in range(mj_model.ngeom):
        if i == floor_id:
            mj_model.geom_contype[i] = 1
            mj_model.geom_conaffinity[i] = 2
        else:
            mj_model.geom_contype[i] = 2
            mj_model.geom_conaffinity[i] = 1
    return mj_model


# ============================================================================ #
#                               Env definition                                 #
# ============================================================================ #
def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,
        episode_length=500,  # 10 s of sim time
        action_repeat=1,
        action_scale=jp.pi / 2.0,
        upright_termination_z=-0.3,
        reward_config=config_dict.create(
            scales=config_dict.create(
                forward_speed=1.0,
                alive=0.05,
                upright=0.1,
                action_rate=-0.01,
                torques=-1e-4,
                joint_vel=-1e-4,
            ),
        ),
        impl="jax",
        njmax=200,
        naconmax=12_000,
    )


class UndirectedLocomotionInsect(mjx_env.MjxEnv):
    """Reward displacement from origin in any direction."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)

        self._mj_model = _build_insect_mj_model(self._config.sim_dt)
        self._mjx_model = mjx_env.put_model(self._mj_model, impl=self._config.impl)
        self._xml_path = ""

        self._init_qpos = jp.array(self._mj_model.qpos0)
        self._torso_body_id = self._mj_model.body(TORSO_BODY_NAME).id
        self._nu = self._mj_model.nu
        self._nv = self._mj_model.nv

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, qpos_rng = jax.random.split(rng)

        qpos = self._init_qpos
        joint_noise = 0.05 * jax.random.uniform(
            qpos_rng, (self._nu,), minval=-1.0, maxval=1.0
        )
        qpos = qpos.at[7:].set(qpos[7:] + joint_noise)

        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            qvel=jp.zeros(self._nv),
            ctrl=jp.zeros(self._nu),
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        info = {
            "rng": rng,
            "last_act": jp.zeros(self._nu),
            "step": jp.int32(0),
        }
        metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward_config.scales.keys()}
        metrics["displacement"] = jp.zeros(())

        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        motor_targets = jp.clip(
            action * self._config.action_scale,
            -self._config.action_scale,
            self._config.action_scale,
        )
        data = mjx_env.step(
            self.mjx_model, state.data, motor_targets, self.n_substeps
        )

        obs = self._get_obs(data, state.info)
        rewards = self._compute_rewards(data, action, state.info)
        reward = jp.sum(
            jp.array([
                v * self._config.reward_config.scales[k]
                for k, v in rewards.items()
            ])
        ) * self.dt

        up_proj = data.xmat[self._torso_body_id, 2, 2]
        done = up_proj < self._config.upright_termination_z
        nan_state = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = done | nan_state

        reward = jp.where(jp.isnan(reward) | nan_state, 0.0, reward)
        reward = jp.clip(reward, -10.0, 10.0)

        state.info["last_act"] = action
        state.info["step"] = state.info["step"] + 1
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
        state.metrics["displacement"] = jp.linalg.norm(data.qpos[:2])

        return state.replace(
            data=data,
            obs=obs,
            reward=reward,
            done=done.astype(jp.float32),
        )

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> Dict[str, jax.Array]:
        joint_angles = data.qpos[7:]
        joint_vel = data.qvel[6:]
        up = data.xmat[self._torso_body_id, :, 2]
        linvel_world = data.qvel[0:3]
        angvel_world = data.qvel[3:6]
        torso_rot = data.xmat[self._torso_body_id].reshape(-1)
        phase = jp.array([
            jp.sin(2.0 * jp.pi * data.time),
            jp.cos(2.0 * jp.pi * data.time),
        ])

        state = jp.concatenate([
            jp.array([data.qpos[2]]),
            torso_rot,
            joint_angles,
            joint_vel,
            linvel_world,
            angvel_world,
            up,
            info["last_act"],
            phase,
        ])
        state = jp.nan_to_num(state, nan=0.0, posinf=10.0, neginf=-10.0)
        state = jp.clip(state, -10.0, 10.0)
        return {"state": state}

    def _compute_rewards(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
    ) -> dict[str, jax.Array]:
        xy_speed = jp.linalg.norm(data.qvel[0:2])
        up_proj = data.xmat[self._torso_body_id, 2, 2]

        return {
            "forward_speed": xy_speed,
            "alive": jp.ones(()),
            "upright": jp.clip(up_proj, 0.0, 1.0),
            "action_rate": jp.sum(jp.square(action - info["last_act"])),
            "torques": jp.sum(jp.square(data.actuator_force)),
            "joint_vel": jp.sum(jp.square(data.qvel[6:])),
        }


locomotion.register_environment(
    ENV_NAME, UndirectedLocomotionInsect, default_config
)


# ============================================================================ #
#                                PPO config                                    #
# ============================================================================ #
def ppo_config(num_timesteps: int, num_envs: int, batch_size: int) -> config_dict.ConfigDict:
    return config_dict.create(
        num_timesteps=num_timesteps,
        num_evals=10,
        reward_scaling=1.0,
        episode_length=500,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=8,
        num_minibatches=16,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        num_envs=num_envs,
        num_eval_envs=min(128, num_envs // 8),
        batch_size=batch_size,
        max_grad_norm=1.0,
        num_resets_per_eval=10,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(64, 64),
            value_hidden_layer_sizes=(128, 128),
            policy_obs_key="state",
            value_obs_key="state",
        ),
    )


# ============================================================================ #
#                                  Training                                    #
# ============================================================================ #
def train(args: argparse.Namespace) -> tuple[Any, Any, Path]:
    print(f"jax backend: {jax.default_backend()}  devices: {jax.devices()}")

    env_cfg = default_config()
    env = UndirectedLocomotionInsect(env_cfg)
    eval_env = UndirectedLocomotionInsect(env_cfg)

    rl_cfg = ppo_config(args.num_timesteps, args.num_envs, args.batch_size)

    out_dir = (
        Path(args.outdir)
        / f"insect_undirected-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "env_config.json").write_text(json.dumps(env_cfg.to_dict(), indent=2))
    (out_dir / "ppo_config.json").write_text(json.dumps(rl_cfg.to_dict(), indent=2))
    print(f"logging to {out_dir}")

    history: dict[str, list[float]] = {"steps": [], "reward": [], "reward_std": []}
    times = [time.monotonic()]

    def progress(num_steps: int, metrics: dict[str, Any]) -> None:
        times.append(time.monotonic())
        reward = float(metrics.get("eval/episode_reward", float("nan")))
        reward_std = float(metrics.get("eval/episode_reward_std", float("nan")))
        history["steps"].append(int(num_steps))
        history["reward"].append(reward)
        history["reward_std"].append(reward_std)
        print(
            f"[{num_steps:>10}] reward={reward:+.3f} ± {reward_std:.3f}"
            f"  elapsed={times[-1] - times[0]:7.1f}s",
            flush=True,
        )

    train_params = dict(rl_cfg)
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **train_params.pop("network_factory")
    )
    num_eval_envs = train_params.pop("num_eval_envs")

    train_fn = functools.partial(
        ppo.train,
        **train_params,
        network_factory=network_factory,
        seed=args.seed,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_eval_envs=num_eval_envs,
    )

    make_inference_fn, params, _ = train_fn(
        environment=env,
        eval_env=eval_env,
        progress_fn=progress,
    )

    if len(times) > 1:
        print(f"time to jit: {times[1] - times[0]:.1f}s")
        print(f"time to train: {times[-1] - times[1]:.1f}s")

    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))

    if history["steps"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        steps = np.array(history["steps"])
        reward = np.array(history["reward"])
        reward_std = np.array(history["reward_std"])
        ax.plot(steps, reward, color="tab:blue")
        ax.fill_between(steps, reward - reward_std, reward + reward_std, color="tab:blue", alpha=0.25)
        ax.set_xlabel("environment steps")
        ax.set_ylabel("eval episode reward")
        ax.set_title("Insect undirected locomotion (PPO)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "reward_curve.png", dpi=140)
        plt.close(fig)

    return make_inference_fn, params, out_dir


# ============================================================================ #
#                              Rollout and render                              #
# ============================================================================ #
def render_rollout(
    make_inference_fn: Any,
    params: Any,
    out_dir: Path,
    episode_length: int,
    seed: int,
) -> None:
    env = UndirectedLocomotionInsect(default_config())
    inference_fn = make_inference_fn(params, deterministic=True)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(inference_fn)

    rng = jax.random.PRNGKey(seed + 100)
    state = jit_reset(rng)
    rollout = [state]
    trajectory = [(float(state.data.qpos[0]), float(state.data.qpos[1]))]

    for _ in range(episode_length):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state)
        trajectory.append((float(state.data.qpos[0]), float(state.data.qpos[1])))
        if bool(state.done):
            break

    final_disp = float(jp.linalg.norm(state.data.qpos[:2]))
    print(f"rollout length: {len(rollout)}  final displacement: {final_disp:.3f} m")

    render_every = 2
    fps = 1.0 / env.dt / render_every
    traj = rollout[::render_every]
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    frames = env.render(
        traj, camera="track", scene_option=scene_option, height=480, width=640
    )
    video_path = out_dir / "rollout.mp4"
    media.write_video(str(video_path), frames, fps=fps)
    print(f"saved video → {video_path}")

    xs = np.array([p[0] for p in trajectory])
    ys = np.array([p[1] for p in trajectory])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xs, ys, "b-", linewidth=2, label="trajectory")
    ax.plot(xs[0], ys[0], "go", markersize=10, label="start")
    ax.plot(xs[-1], ys[-1], "r*", markersize=14, label="end")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"insect undirected locomotion — displacement {final_disp:.2f} m")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path = out_dir / "trajectory.png"
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)
    print(f"saved trajectory plot → {plot_path}")


# ============================================================================ #
#                                   Main                                       #
# ============================================================================ #
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-timesteps", type=int, default=20_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4096,
        help="Parallel envs. T4 (16GB): 4096–8192. A100: 16384+.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="PPO minibatch size. Scale with num_envs.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="undirected_locomotion_ppo_jax",
    )
    parser.add_argument("--episode-length", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    make_inference_fn, params, out_dir = train(args)
    render_rollout(make_inference_fn, params, out_dir, args.episode_length, args.seed)


if __name__ == "__main__":
    main()
