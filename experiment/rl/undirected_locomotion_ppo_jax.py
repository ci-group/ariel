"""Undirected locomotion for the ARIEL ``insect_small`` morphology.

Trains a PPO policy with mujoco-playground + brax on GPU. The task reward
encourages movement in any direction away from the origin, hence
"undirected" locomotion. The custom environment is a thin subclass of
``mujoco_playground._src.mjx_env.MjxEnv`` that wraps an ariel-built MJCF.

Must be run from the ``mujoco_playground`` virtualenv, which already has
jax (CUDA), brax and mujoco_playground installed. ariel is installed as an
editable package into that venv.

Example
-------
``/home/user/Desktop/EvoDevo/mujoco_playground/.venv/bin/python \
    experiment/undirected_locomotion_ppo_jax.py --num-timesteps 20000000``
"""

from __future__ import annotations

# XLA/MuJoCo configuration — must happen before jax and brax import.
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Async malloc reduces GPU memory fragmentation (recommended by XLA for OOMs).
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("MUJOCO_GL", "egl")

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
from brax.io import model
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
FLOOR_BODY_NAME = "floor"
# Six leg-tip bricks (see insect_small in prebuilt_robots/insect.py). The
# spine bricks (brick_0, brick_1) and the core are intentionally excluded —
# touching the ground with those is not "stepping".
FOOT_BODY_NAMES = tuple(f"robot1_brick_{i}brick" for i in range(2, 8))

# ---- Batch-size and solver-buffer budgets ---------------------------------- #
# All sizing is derived from NUM_ENVS so changing it doesn't silently under- or
# over-allocate the solver buffers. The per-env numbers come from measurements
# in `njmax_naconmax_progress.md` (nefc peak ≈144 for random control; CCD peak
# ≈135 observed during training). Keep a ~10% safety margin above observed.
NUM_ENVS = 6144                                       # ~8-9 GB VRAM target on 12 GiB GPU
NJMAX_PER_ENV = 800                                   # ~5.5× random-control nefc peak
NACONMAX_PER_ENV = 320                                # contact-pool slack per env
NACCDMAX_PER_ENV = 100                                # dropped from 150: 135-peak was during hover exploit, walking generates fewer CCDs; frees ~1.7 GB of CCD workspace

# ---- PPO batching, derived from NUM_ENVS ----------------------------------- #
# brax asserts: batch_size * num_minibatches % num_envs == 0 (data from num_envs
# parallel rollouts is split into num_minibatches minibatches of batch_size).
# We fix the ratio at BATCH_RATIO (number of SGD passes per rollout) and derive
# batch_size, so changing NUM_ENVS never desyncs the assertion.
NUM_MINIBATCHES = 16
BATCH_RATIO = 6                                       # historical k; batch_size*num_minibatches = BATCH_RATIO*NUM_ENVS
BATCH_SIZE = BATCH_RATIO * NUM_ENVS // NUM_MINIBATCHES


# ============================================================================ #
#                              MuJoCo model build                              #
# ============================================================================ # 
def _build_insect_mj_model(sim_dt: float) -> mujoco.MjModel:
    """Compile the ariel insect spawned on a flat world."""
    world = SimpleFlatWorld(load_precompiled=False)
    body = insect_small()
    world.spawn(body.spec, position=[0, 0, 0.1])

    # Tracking camera that follows the robot torso centre-of-mass.
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
    # implicitfast integrates damping implicitly: much more stable with
    # stiff contacts than Euler, same per-step cost.
    mj_model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # 6/12 is the usual sweet spot for legged MJX: enough iterations that
    # feet don't sink through the floor, still fast to compile.
    mj_model.opt.iterations = 6
    mj_model.opt.ls_iterations = 12
    mj_model.vis.global_.offwidth = 1280
    mj_model.vis.global_.offheight = 720

    # Enable self-collisions AND floor contacts. Without this, robot limbs
    # pass through each other. We still want to avoid phantom contacts at
    # joint attachments (parent-child module boundaries), so we bitmask:
    #   - floor:   contype=1, conaffinity=1  (collides with everything)
    #   - robot:   contype=1, conaffinity=1  (collides with floor + distant limbs)
    # Adjacent modules are auto-excluded by MuJoCo via the body hierarchy
    # when the modules were attached (ariel's attach_body uses welds/joints).
    for i in range(mj_model.ngeom):
        mj_model.geom_contype[i] = 1
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
        action_scale=jp.pi / 2.0,  # policy outputs in [-1, 1] → ±90° joint target
        # NOTE: motor-target low-pass filtering is now modeled at the actuator
        # level in ariel's HingeModule (dyntype=FILTEREXACT, τ=50 ms), so no
        # python-side filter is needed here.
        upright_termination_z=-0.3,  # terminate when torso 'up' projection drops below this
        reward_config=config_dict.create(
            # xy_speed threshold below which the stall penalty activates.
            stall_speed=0.05,
            # Reward is positive when a foot stays airborne for AIR_TIME_TARGET
            # before touching down. Shorter → encourages fast stepping; longer
            # → longer swings. Go1 uses 0.1 s.
            air_time_target=0.1,
            scales=config_dict.create(
                forward_speed=5.0,     # strong push for locomotion
                stall=-3.0,            # penalize staying nearly stationary
                upright=0.01,          # mild upright bonus
                action_rate=-0.1,      # softened from -0.5: give π/2 action range room to explore
                torques=-1e-4,         # energy
                joint_vel=-5e-4,       # softened from -5e-3: give π/2 action range room to explore
                lin_vel_z=-1.0,        # anti-hover: 2× to counter flutter re-emergence with soft action_rate
                ang_vel_xy=-0.2,       # anti-flutter: 4× for same reason
                feet_air_time=1.0,     # reward natural stepping (air→ground events)
            ),
        ),
        impl="warp",
        njmax=NJMAX_PER_ENV,                           # per-world
        naconmax=NUM_ENVS * NACONMAX_PER_ENV,          # total across all worlds
        naccdmax=NUM_ENVS * NACCDMAX_PER_ENV,          # total across all worlds
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
        self._xml_path = ""  # programmatically generated

        self._init_qpos = jp.array(self._mj_model.qpos0)
        self._torso_body_id = self._mj_model.body(TORSO_BODY_NAME).id
        self._nu = self._mj_model.nu
        self._nv = self._mj_model.nv

        # Foot bodies for proximity-based contact detection. We use xpos[z]
        # rather than iterating data._impl.contact__*, because the WARP contact
        # pool is shared across all worlds — filtering by worldid under vmap is
        # fragile, while xpos is cleanly per-world.
        self._foot_body_ids = jp.array(
            [self._mj_model.body(n).id for n in FOOT_BODY_NAMES]
        )
        self._n_feet = int(self._foot_body_ids.shape[0])
        # Brick half-width is 0.05 (BRICK_DIMENSIONS); foot COM sits ~0.05
        # above its geom bottom. 0.055 → "within 0.5 cm of floor".
        self._foot_contact_z = 0.055

    # ---- MjxEnv interface ------------------------------------------------- #
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

    # ---- Rollout ---------------------------------------------------------- #
    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, qpos_rng = jax.random.split(rng)

        qpos = self._init_qpos
        # Small random hinge perturbations so successive seeds explore
        # different initial configurations.
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
            naccdmax=self._config.naccdmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        info = {
            "rng": rng,
            "last_act": jp.zeros(self._nu),
            "step": jp.int32(0),
            "feet_air_time": jp.zeros(self._n_feet),
            "last_contact": jp.zeros(self._n_feet, dtype=bool),
        }
        metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward_config.scales.keys()}
        metrics["displacement"] = jp.zeros(())

        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        motor_targets = jp.clip(
            action * self._config.action_scale, -self._config.action_scale, self._config.action_scale
        )
        data = mjx_env.step(
            self.mjx_model, state.data, motor_targets, self.n_substeps
        )

        # Per-foot ground contact. A contact slot is active when dist < 0;
        # it counts for foot i if the pair involves foot_geom_ids[i] + floor.
        contact = self._feet_contact(data)
        air_time_prev = state.info["feet_air_time"]
        air_time = (air_time_prev + self.dt) * (~contact)
        # "first_contact": foot just transitioned from in-air to touching down.
        first_contact = contact & (air_time_prev > 0.0)

        obs = self._get_obs(data, state.info)
        rewards = self._compute_rewards(data, action, state.info, air_time_prev, first_contact)
        reward = jp.sum(
            jp.array([
                v * self._config.reward_config.scales[k]
                for k, v in rewards.items()
            ])
        ) * self.dt

        # Terminate when the body flips over (upright projection < threshold).
        up_proj = data.xmat[self._torso_body_id, 2, 2]
        done = up_proj < self._config.upright_termination_z
        nan_state = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = done | nan_state

        # NaN guard: never let NaN rewards leak into brax PPO (poisons GAE).
        reward = jp.where(jp.isnan(reward) | nan_state, 0.0, reward)
        reward = jp.clip(reward, -10.0, 10.0)

        # Bookkeeping.
        state.info["last_act"] = action
        state.info["step"] = state.info["step"] + 1
        state.info["feet_air_time"] = air_time
        state.info["last_contact"] = contact
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
        state.metrics["displacement"] = jp.linalg.norm(data.qpos[:2])

        return state.replace(
            data=data,
            obs=obs,
            reward=reward,
            done=done.astype(jp.float32),
        )

    def _feet_contact(self, data: mjx.Data) -> jax.Array:
        """Boolean mask of shape (n_feet,): True where foot-i is near floor."""
        foot_z = data.xpos[self._foot_body_ids, 2]
        return foot_z < self._foot_contact_z

    # ---- Observations and rewards ---------------------------------------- #
    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> Dict[str, jax.Array]:
        joint_angles = data.qpos[7:]
        joint_vel = data.qvel[6:]

        up = data.xmat[self._torso_body_id, :, 2]            # body-up vector (3,)
        linvel_world = data.qvel[0:3]
        angvel_world = data.qvel[3:6]

        # xmat is 3x3, flatten the rotation of the torso so the policy sees orientation.
        torso_rot = data.xmat[self._torso_body_id].reshape(-1)  # (9,)

        phase = jp.array([
            jp.sin(2.0 * jp.pi * data.time),
            jp.cos(2.0 * jp.pi * data.time),
        ])

        state = jp.concatenate([
            jp.array([data.qpos[2]]),  # torso height
            torso_rot,                 # orientation
            joint_angles,
            joint_vel,
            linvel_world,
            angvel_world,
            up,
            info["last_act"],
            phase,
        ])
        # Guard against physics blow-ups leaking into the critic/policy.
        state = jp.nan_to_num(state, nan=0.0, posinf=10.0, neginf=-10.0)
        state = jp.clip(state, -10.0, 10.0)
        return {"state": state}

    def _compute_rewards(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        air_time_prev: jax.Array,
        first_contact: jax.Array,
    ) -> dict[str, jax.Array]:
        xy_speed = jp.linalg.norm(data.qvel[0:2])
        up_proj = data.xmat[self._torso_body_id, 2, 2]
        stall_speed = self._config.reward_config.stall_speed
        air_time_target = self._config.reward_config.air_time_target

        # feet_air_time: sum over feet of (air_time - target) at the moment a
        # foot lands. Rewards swings close to `target`; zero if no landing
        # events. Gated by xy_speed > stall_speed so the policy can't farm the
        # bonus by tapping feet in place (go1 does the same gating via cmd_norm).
        moving = (xy_speed > stall_speed).astype(jp.float32)
        feet_air_time_reward = (
            jp.sum((air_time_prev - air_time_target) * first_contact) * moving
        )

        return {
            "forward_speed": xy_speed,
            "stall": jp.clip(stall_speed - xy_speed, 0.0, stall_speed),
            "upright": jp.clip(up_proj, 0.0, 1.0),
            "action_rate": jp.sum(jp.square(action - info["last_act"])),
            "torques": jp.sum(jp.square(data.actuator_force)),
            "joint_vel": jp.sum(jp.square(data.qvel[6:])),
            "lin_vel_z": jp.square(data.qvel[2]),
            "ang_vel_xy": jp.sum(jp.square(data.qvel[3:5])),
            "feet_air_time": feet_air_time_reward,
        }


# Register with the mujoco_playground locomotion suite so the standard
# registry.get_default_config / registry.load pipeline works.
locomotion.register_environment(
    ENV_NAME, UndirectedLocomotionInsect, default_config
)


# ============================================================================ #
#                                PPO config                                    #
# ============================================================================ #
def ppo_config(num_timesteps: int) -> config_dict.ConfigDict:
    return config_dict.create(
        num_timesteps=num_timesteps,
        num_evals=10,
        reward_scaling=1.0,
        episode_length=500,  # must match env default_config().episode_length
        normalize_observations=True,
        action_repeat=1,
        unroll_length=8,
        num_minibatches=NUM_MINIBATCHES,
        num_updates_per_batch=4,
        discounting=0.99,  # horizon ≈1.9 s at ctrl_dt=0.02
        learning_rate=3e-4,
        entropy_cost=1e-2,
        num_envs=NUM_ENVS,
        num_eval_envs=128,
        batch_size=BATCH_SIZE,
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

    rl_cfg = ppo_config(args.num_timesteps)

    env_cfg = default_config()
    env = UndirectedLocomotionInsect(env_cfg)
    # Eval env runs with far fewer worlds (num_eval_envs vs NUM_ENVS). Shrink
    # its naconmax/naccdmax to match — otherwise it keeps a full training-sized
    # CCD workspace alive during eval and OOMs the GPU.
    eval_cfg = default_config()
    eval_cfg.naconmax = int(rl_cfg.num_eval_envs) * NACONMAX_PER_ENV
    eval_cfg.naccdmax = int(rl_cfg.num_eval_envs) * NACCDMAX_PER_ENV
    eval_env = UndirectedLocomotionInsect(eval_cfg)

    out_dir = (
        Path(args.outdir)
        / f"insect_undirected-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "env_config.json").write_text(json.dumps(env_cfg.to_dict(), indent=2))
    (out_dir / "eval_env_config.json").write_text(json.dumps(eval_cfg.to_dict(), indent=2))
    (out_dir / "ppo_config.json").write_text(json.dumps(rl_cfg.to_dict(), indent=2))
    print(f"logging to {out_dir}")

    history: dict[str, list[float]] = {"steps": [], "reward": [], "reward_std": []}
    times = [time.monotonic()]
    # Each entry is (step, params_as_numpy_pytree); captured at init + every eval.
    # Consumed by experiment/plotting/fitness_landscape_ppo_pca.py.
    snapshots: list[tuple[int, Any]] = []

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

    def snapshot_fn(current_step: int, _make_policy: Any, params_tuple: Any) -> None:
        # Copy to host-side numpy so the entry survives beyond the next training step.
        snapshots.append(
            (int(current_step), jax.tree_util.tree_map(np.asarray, params_tuple))
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
        policy_params_fn=snapshot_fn,
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
    # Save params so rerun_undirected_locomotion_ppo_jax.py can replay the rollout
    # without retraining.
    model.save_params(str(out_dir / "params.pkl"), params)
    # Save the per-eval snapshot trajectory for fitness_landscape_ppo_pca.py.
    model.save_params(str(out_dir / "params_snapshots.pkl"), snapshots)

    # Reward curve.
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
    # Shrink the solver buffers for the single-env rollout. The training
    # config sizes buffers for NUM_ENVS worlds; replaying it here would compile
    # a second WARP graph alongside training's, OOMing the GPU.
    rollout_cfg = default_config()
    rollout_cfg.naconmax = NACONMAX_PER_ENV           # single env
    rollout_cfg.naccdmax = NACCDMAX_PER_ENV           # single env
    env = UndirectedLocomotionInsect(rollout_cfg)
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

    # Video.
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

    # Trajectory plot.
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
    parser.add_argument("--num-timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(Path(__file__).parent.parent / "__data__" / "undirected_locomotion_ppo_jax"),
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=500,
        help="Episode length for the post-training rollout.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    make_inference_fn, params, out_dir = train(args)
    # Rollout needs ~15 MB of extra GPU space to compile a fresh WARP graph.
    # Training keeps buffers pinned via cuda_malloc_async's pool, so retry
    # after an explicit cache + GC pass. If it still OOMs, log and move on —
    # training results are already saved to out_dir.
    import gc
    jax.clear_caches()
    gc.collect()
    try:
        render_rollout(make_inference_fn, params, out_dir, args.episode_length, args.seed)
    except Exception as e:
        print(f"render_rollout skipped: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
