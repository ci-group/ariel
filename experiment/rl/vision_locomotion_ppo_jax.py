"""Vision-based directed locomotion: find and walk to a red ball by sight.

Extends ``directed_locomotion_ppo_jax.py``. The robot's core is a head with
an RGB camera ("eye") looking along the head axis. The policy no longer gets
the target's coordinates — it must *detect* the target in its camera image:

* Target = red ball. ``n_distractors`` balls of other colours (blue, green,
  …) are placed at other bearings, so the policy has to discriminate by
  colour, not just find "the object".
* Actor (policy) input: camera frames (``pixels/view_0``, a stack of the last
  ``frame_stack`` RGB frames) + proprioception (``state``: joints,
  velocities, gravity, last action, clock). No target information.
* Critic input: camera + proprioception + the TRUE target bearing/distance
  (``privileged_state``). The critic only exists during training, so giving
  it privileged state is standard (asymmetric actor-critic, Pinto et al.
  2018) and keeps PPO's value estimates usable while the actor learns to see.
* Network: brax's vision PPO (CNN → spatial softmax → MLP). Spatial softmax
  outputs *where* each feature fires in the image, which is what steering
  toward a ball needs.
* Rendering: MuJoCo Warp's batched GPU renderer, one frame per control step
  for every parallel world (render context sized to the number of envs).

Reward, heading gate, target resampling, and the autoreset handling are
inherited unchanged from the directed env.

Run from the repo root with the mujoco_playground venv:

``experiment/mujoco_playground/.venv/bin/python \
    experiment/rl/vision_locomotion_ppo_jax.py --num-timesteps 100000000``
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import functools
import json
import sys
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
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground import wrapper
from mujoco_playground._src import locomotion, mjx_env

sys.path.insert(0, str(Path(__file__).resolve().parent))
import directed_locomotion_ppo_jax as base  # noqa: E402

ENV_NAME = "ArielInsectVisionLocomotion"
EYE_CAMERA_NAME = "eye"
TARGET_BALL_RGBA = (0.9, 0.1, 0.1, 1.0)
# Distractor colours, used in order. Chosen far from red in RGB.
DISTRACTOR_RGBAS = (
    (0.1, 0.3, 0.9, 1.0),   # blue
    (0.1, 0.8, 0.2, 1.0),   # green
    (0.9, 0.9, 0.1, 1.0),   # yellow
    (0.8, 0.2, 0.9, 1.0),   # purple
)
BALL_RADIUS = 0.12  # ~5 px wide at 2.5 m in a 64-px-wide image

# Vision is far heavier per env than proprio-only.
NUM_ENVS = 1024
# Rollout chunks collected per PPO update (batch_size * num_minibatches =
# VISION_BATCH_RATIO * num_envs). The proprio scripts use 6; with images
# that stores ~6x more frames and brax runs the CNN over the whole batch at
# once — at 512 envs that was a single 7.7 GiB allocation (OOM on 12 GiB).
VISION_BATCH_RATIO = 1


# ============================================================================ #
#                                   Config                                     #
# ============================================================================ #
def default_config() -> config_dict.ConfigDict:
    cfg = base.default_config()
    # The actor never sees the target coordinates in this env, so the
    # proprio FOV mask is irrelevant; the eye camera defines what is seen.
    cfg.fov_deg = 360.0
    cfg.naconmax = NUM_ENVS * base.NACONMAX_PER_ENV
    cfg.naccdmax = NUM_ENVS * base.NACCDMAX_PER_ENV
    cfg.n_distractors = 2
    cfg.distractor_min_sep_deg = 60.0   # min bearing gap from the target (seen from the robot)
    cfg.frame_stack = 2
    cfg.vision = config_dict.create(
        nworld=NUM_ENVS,                 # must equal the number of parallel envs
        cam_res=(64, 48),                # (width, height)
        cam_fovy=75.0,                   # vertical FOV, deg → ~91° horizontal at 4:3
        cam_tilt_deg=15.0,               # look slightly down
        use_textures=False,
        use_shadows=False,
    )
    return cfg


def _add_eye_and_balls(cfg: config_dict.ConfigDict):
    """Spec hook: eye camera on the head face + target ball + distractors."""
    if cfg.head_axis != "-y":
        raise ValueError("vision env currently supports head_axis='-y' only")

    def fn(spec: mujoco.MjSpec, core_body: Any) -> None:
        # Camera looks along its local -z with +y up. We want it to look
        # along the core's -y (the head axis), tilted down by `a`:
        #   x_cam = -x_core,  y_cam = (0, -sin a, cos a),  z_cam = (0, cos a, sin a)
        a = np.deg2rad(cfg.vision.cam_tilt_deg)
        core_body.add_camera(
            name=EYE_CAMERA_NAME,
            pos=[0.0, -0.105, 0.03],     # just outside the core's -y face
            xyaxes=[-1.0, 0.0, 0.0, 0.0, -np.sin(a), np.cos(a)],
            fovy=cfg.vision.cam_fovy,
        )
        # Red ball on the (already mocap) target body.
        # spec.body(name) returns None on ARIEL's composed world spec, so
        # search the worldbody's direct children instead.
        target = next(b for b in spec.worldbody.bodies if b.name == base.TARGET_BODY_NAME)
        target.add_geom(
            name="target_ball",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[BALL_RADIUS, 0.0, 0.0],
            pos=[0.0, 0.0, BALL_RADIUS],
            rgba=list(TARGET_BALL_RGBA),
            contype=0,
            conaffinity=0,
        )
        for k in range(cfg.n_distractors):
            b = spec.worldbody.add_body(
                name=f"distractor_{k}", mocap=True, pos=[0.0, 0.0, 0.0]
            )
            b.add_geom(
                name=f"distractor_{k}_ball",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[BALL_RADIUS, 0.0, 0.0],
                pos=[0.0, 0.0, BALL_RADIUS],
                rgba=list(DISTRACTOR_RGBAS[k % len(DISTRACTOR_RGBAS)]),
                contype=0,
                conaffinity=0,
            )

    return fn


# ============================================================================ #
#                                     Env                                      #
# ============================================================================ #
class VisionLocomotionInsect(base.DirectedLocomotionInsect):
    """Directed locomotion where the target must be found in a camera image."""

    def __init__(
        self,
        config: Optional[config_dict.ConfigDict] = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        # Skip DirectedLocomotionInsect.__init__'s model build: initialise the
        # MjxEnv base with our config, then build the model with the hook.
        mjx_env.MjxEnv.__init__(self, config or default_config(), config_overrides)
        cfg = self._config
        self._mj_model = base._build_insect_mj_model(
            cfg.sim_dt,
            morphology=cfg.morphology,
            n_pairs=cfg.n_pairs,
            friction=cfg.friction,
            core_mass=cfg.core_mass,
            extra_spec_fn=_add_eye_and_balls(cfg),
        )
        self._mjx_model = mjx_env.put_model(self._mj_model, impl=cfg.impl)
        self._init_common()

        m = self._mj_model
        self._distractor_mocap_ids = jp.array(
            [int(m.body_mocapid[m.body(f"distractor_{k}").id]) for k in range(cfg.n_distractors)],
            dtype=jp.int32,
        )
        self._eye_cam_id = m.camera(EYE_CAMERA_NAME).id
        cam_active = tuple(i == self._eye_cam_id for i in range(m.ncam))
        self._rc = mjx.create_render_context(
            mjm=m,
            nworld=cfg.vision.nworld,
            cam_res=tuple(cfg.vision.cam_res),
            use_textures=cfg.vision.use_textures,
            use_shadows=cfg.vision.use_shadows,
            render_rgb=(True,),          # per ACTIVE camera (only the eye)
            render_depth=(False,),
            enabled_geom_groups=[0, 1, 2],
            cam_active=cam_active,
        )
        self._rc_pytree = self._rc.pytree()

    # ---- Distractors ------------------------------------------------------ #
    def _sample_distractors(
        self, rng: jax.Array, origin_xy: jax.Array, target_xy: jax.Array
    ) -> jax.Array:
        """(n_distractors, 2) positions around origin, at bearings at least
        ``distractor_min_sep_deg`` away from the target's bearing, so a
        distractor never sits in front of / behind the target."""
        n = self._config.n_distractors
        if n == 0:
            return jp.zeros((0, 2))
        a_rng, d_rng = jax.random.split(rng)
        to_t = target_xy - origin_xy
        t_angle = jp.arctan2(to_t[1], to_t[0])
        sep = jp.deg2rad(self._config.distractor_min_sep_deg)
        offsets = jax.random.uniform(a_rng, (n,), minval=sep, maxval=2 * jp.pi - sep)
        angles = t_angle + offsets
        dists = jax.random.uniform(
            d_rng, (n,),
            minval=self._config.target_dist_min,
            maxval=self._config.target_dist_max,
        )
        return origin_xy + dists[:, None] * jp.stack([jp.cos(angles), jp.sin(angles)], -1)

    def _write_distractors(self, data: mjx.Data, xy: jax.Array) -> mjx.Data:
        if self._config.n_distractors == 0:
            return data
        mocap_pos = data.mocap_pos.at[self._distractor_mocap_ids, :2].set(xy)
        return data.replace(mocap_pos=mocap_pos)

    # ---- Camera ----------------------------------------------------------- #
    def _render_eye(self, data: mjx.Data) -> tuple[jax.Array, mjx.Data]:
        """Render the eye camera. Returns (H, W, 3) RGB in [-0.5, 0.5]."""
        # Mocap positions (target / distractors) were just written; recompute
        # kinematics so geom poses reflect them — otherwise the renderer draws
        # each object where it was one step earlier.
        data = mjx.kinematics(self.mjx_model, data)
        data = mjx.refit_bvh(self.mjx_model, data, self._rc_pytree)
        out = mjx.render(self.mjx_model, data, self._rc_pytree)
        data = out[2]
        rgb = mjx.get_rgb(self._rc_pytree, 0, out[0])  # index among active cams
        return rgb[..., :3] - 0.5, data

    # ---- Observations ----------------------------------------------------- #
    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> Dict[str, jax.Array]:
        """Proprio-only actor state + privileged critic state (no pixels;
        those are added in reset/step after rendering)."""
        obs = super()._get_obs(data, info)
        # Base `state` ends with [masked cos, sin, dist, visible] (4 values):
        # strip them, the actor must find the target by sight.
        proprio = obs["state"][:-4]
        cos_b, sin_b, dist = self._head_bearing(data, info["target"])
        privileged = jp.concatenate([
            proprio,
            jp.array([cos_b, sin_b, jp.minimum(dist, self._config.obs_dist_clip)]),
        ])
        privileged = jp.clip(jp.nan_to_num(privileged), -10.0, 10.0)
        return {"state": proprio, "privileged_state": privileged}

    def _with_pixels(self, obs: Dict[str, jax.Array], stack: jax.Array) -> Dict[str, jax.Array]:
        return {**obs, "pixels/view_0": stack}

    # ---- Rollout ---------------------------------------------------------- #
    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, d_rng = jax.random.split(rng)
        state = super().reset(rng)
        data = state.data
        core_xy = data.xpos[self._torso_body_id, :2]
        distractors = self._sample_distractors(d_rng, core_xy, state.info["target"])
        data = self._write_distractors(data, distractors)

        frame, data = self._render_eye(data)
        stack = jp.tile(frame, (1, 1, self._config.frame_stack))
        info = dict(state.info)
        info["distractors"] = distractors
        info["frame_stack"] = stack
        return state.replace(data=data, obs=self._with_pixels(state.obs, stack), info=info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        is_first = state.data.time < 0.5 * self._config.sim_dt
        old_target = state.info["target"]
        # Physics, rewards, target reach/resample, autoreset re-init.
        state = super().step(state, action)
        info = dict(state.info)
        data = state.data

        # Resample distractors whenever the target changed (reached, or a
        # new episode after autoreset). On a genuine reset the target is
        # unchanged and the distractors from reset() are kept.
        target_changed = jp.any(info["target"] != old_target)
        rng, d_rng = jax.random.split(info["rng"])
        info["rng"] = rng
        core_xy = data.xpos[self._torso_body_id, :2]
        new_d = self._sample_distractors(d_rng, core_xy, info["target"])
        distractors = jp.where(target_changed, new_d, info["distractors"])
        data = self._write_distractors(data, distractors)
        info["distractors"] = distractors

        frame, data = self._render_eye(data)
        rolled = jp.concatenate([info["frame_stack"][..., 3:], frame], axis=-1)
        fresh = jp.tile(frame, (1, 1, self._config.frame_stack))
        stack = jp.where(is_first, fresh, rolled)
        info["frame_stack"] = stack

        return state.replace(data=data, obs=self._with_pixels(state.obs, stack), info=info)


locomotion.register_environment(ENV_NAME, VisionLocomotionInsect, default_config)


# ============================================================================ #
#                                PPO config                                    #
# ============================================================================ #
def ppo_config(num_timesteps: int, episode_length: int, num_envs: int) -> config_dict.ConfigDict:
    cfg = base.ppo_config(num_timesteps, episode_length)
    cfg.num_envs = num_envs
    cfg.batch_size = VISION_BATCH_RATIO * num_envs // base.NUM_MINIBATCHES
    cfg.num_eval_envs = 64
    cfg.num_evals = 20
    cfg.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(128, 128),
        value_hidden_layer_sizes=(256, 256, 256),
        policy_obs_key="state",
        value_obs_key="privileged_state",
        cnn_output_channels=(16, 32, 32),
        cnn_kernel_size=(5, 3, 3),
        cnn_stride=(2, 2, 1),
        cnn_padding="zeros",
        cnn_activation="elu",
        cnn_global_pool="spatial_softmax",
    )
    return cfg


# ============================================================================ #
#                                  Training                                    #
# ============================================================================ #
def _env_config(args: argparse.Namespace, n_worlds: int) -> config_dict.ConfigDict:
    cfg = default_config()
    cfg.morphology = args.morphology
    cfg.n_pairs = args.n_pairs
    cfg.episode_length = args.episode_length
    cfg.gate_progress_by_heading = not args.no_heading_gate
    cfg.target_dist_min = args.target_dist_min
    cfg.target_dist_max = args.target_dist_max
    cfg.n_distractors = args.n_distractors
    cfg.frame_stack = args.frame_stack
    cfg.vision.cam_res = (args.cam_width, args.cam_height)
    cfg.vision.nworld = n_worlds
    cfg.naconmax = n_worlds * base.NACONMAX_PER_ENV
    cfg.naccdmax = n_worlds * base.NACCDMAX_PER_ENV
    return cfg


def train(args: argparse.Namespace) -> tuple[Any, Any, Path]:
    print(f"jax backend: {jax.default_backend()}  devices: {jax.devices()}")
    rl_cfg = ppo_config(args.num_timesteps, args.episode_length, args.num_envs)
    rl_cfg.num_evals = args.num_evals

    env_cfg = _env_config(args, args.num_envs)
    env = VisionLocomotionInsect(env_cfg)
    eval_cfg = _env_config(args, int(rl_cfg.num_eval_envs))
    eval_env = VisionLocomotionInsect(eval_cfg)
    print(f"obs shapes: { {k: v for k, v in env.observation_size.items()} }")

    tag_bits = [args.morphology]
    if args.morphology in ("centipede", "insect_n", "insect_longspine_n"):
        tag_bits.append(f"n{args.n_pairs}")
    tag_bits += [f"d{args.n_distractors}", f"seed{args.seed}"]
    out_dir = Path(args.outdir) / f"{'_'.join(tag_bits)}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "env_config.json").write_text(json.dumps(env_cfg.to_dict(), indent=2))
    (out_dir / "ppo_config.json").write_text(json.dumps(rl_cfg.to_dict(), indent=2))
    print(f"logging to {out_dir}")

    history: dict[str, list[float]] = {"steps": [], "reward": [], "reward_std": [], "targets_reached": []}
    times = [time.monotonic()]

    def progress(num_steps: int, metrics: dict[str, Any]) -> None:
        times.append(time.monotonic())
        reward = float(metrics.get("eval/episode_reward", float("nan")))
        reward_std = float(metrics.get("eval/episode_reward_std", float("nan")))
        reached = float(metrics.get("eval/episode_reached", float("nan")))
        history["steps"].append(int(num_steps))
        history["reward"].append(reward)
        history["reward_std"].append(reward_std)
        history["targets_reached"].append(reached)
        print(
            f"[{num_steps:>10}] reward={reward:+.3f} ± {reward_std:.3f}"
            f"  targets/ep={reached:.2f}  elapsed={times[-1] - times[0]:7.1f}s",
            flush=True,
        )
        (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))

    train_params = dict(rl_cfg)
    network_factory = functools.partial(
        ppo_networks_vision.make_ppo_networks_vision, **train_params.pop("network_factory")
    )
    num_eval_envs = train_params.pop("num_eval_envs")
    make_inference_fn, params, _ = ppo.train(
        **train_params,
        network_factory=network_factory,
        seed=args.seed,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_eval_envs=num_eval_envs,
        vision=True,
        environment=env,
        eval_env=eval_env,
        progress_fn=progress,
    )
    if len(times) > 1:
        print(f"time to jit: {times[1] - times[0]:.1f}s  time to train: {times[-1] - times[1]:.1f}s")
    model.save_params(str(out_dir / "params.pkl"), params)

    if history["steps"]:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        steps = np.array(history["steps"])
        r = np.array(history["reward"]); rs = np.array(history["reward_std"])
        ax.plot(steps, r); ax.fill_between(steps, r - rs, r + rs, alpha=0.25)
        ax.set_xlabel("environment steps"); ax.set_ylabel("eval episode reward")
        ax.set_title("Vision directed locomotion (PPO)"); ax.grid(True, alpha=0.3)
        ax2.plot(steps, history["targets_reached"], color="tab:red")
        ax2.set_xlabel("environment steps"); ax2.set_ylabel("targets reached / episode"); ax2.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / "reward_curve.png", dpi=140); plt.close(fig)
    return make_inference_fn, params, out_dir


# ============================================================================ #
#                              Rollout and render                              #
# ============================================================================ #
def render_rollout(make_inference_fn: Any, params: Any, out_dir: Path, args: argparse.Namespace) -> None:
    """Side-by-side video: tracking camera | what the robot's eye sees."""
    env = VisionLocomotionInsect(_env_config(args, 1))
    policy = jax.jit(make_inference_fn(params, deterministic=True))
    # The warp renderer expects a leading world axis (nworld=1): vmap a batch of 1.
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    rng = jax.random.PRNGKey(args.seed + 100)
    state = reset(rng[None])
    rollout, n_reached = [state], 0
    for _ in range(args.episode_length):
        rng, a_rng = jax.random.split(rng)
        act, _ = policy(state.obs, a_rng[None])
        state = step(state, act)
        rollout.append(state)
        n_reached += int(state.metrics["reached"][0])
        if bool(state.done[0]):
            break
    print(f"rollout length: {len(rollout)}  targets reached: {n_reached}")

    # Drop the size-1 world axis. Some warp-only data fields are empty
    # (shape (0, ...)) and were never batched; leave those as they are.
    unbatch = lambda s: jax.tree.map(  # noqa: E731
        lambda x: x[0] if (getattr(x, "ndim", 0) > 0 and x.shape[0] == 1) else x, s)
    traj = [unbatch(s) for s in rollout[::2]]
    h, w = 480, 640
    track = env.render(traj, camera="track", height=h, width=w)
    eye = env.render(traj, camera=EYE_CAMERA_NAME, height=h, width=w)
    # Also show the policy's actual low-res input, upscaled (nearest).
    frames = []
    for i, s in enumerate(traj):
        low = np.asarray(s.obs["pixels/view_0"][..., -3:]) + 0.5
        low = (np.clip(low, 0, 1) * 255).astype(np.uint8)
        ry, rx = h // low.shape[0] + 1, w // low.shape[1] + 1
        up = np.repeat(np.repeat(low, ry, 0), rx, 1)[:h, :w]
        frames.append(np.concatenate([track[i], eye[i], up], axis=1))
    path = out_dir / "rollout_vision.mp4"
    media.write_video(str(path), frames, fps=1.0 / env.dt / 2)
    print(f"saved video → {path}  (left: tracking cam | middle: eye cam | right: policy input)")


# ============================================================================ #
#                                   Main                                       #
# ============================================================================ #
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-timesteps", type=int, default=100_000_000)
    p.add_argument("--num-envs", type=int, default=NUM_ENVS)
    p.add_argument("--num-evals", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str,
                   default=str(Path(__file__).parent.parent / "__data__" / "vision_locomotion_ppo_jax"))
    p.add_argument("--episode-length", type=int, default=500)
    p.add_argument("--morphology", choices=["insect", "centipede", "insect_n", "insect_longspine_n"], default="insect")
    p.add_argument("--n-pairs", type=int, default=2)
    p.add_argument("--n-distractors", type=int, default=2, help="Non-target balls of other colours (0-4).")
    p.add_argument("--frame-stack", type=int, default=2)
    p.add_argument("--cam-width", type=int, default=64)
    p.add_argument("--cam-height", type=int, default=48)
    p.add_argument("--target-dist-min", type=float, default=1.0)
    p.add_argument("--target-dist-max", type=float, default=2.5)
    p.add_argument("--no-heading-gate", action="store_true")
    p.add_argument("--render-only", type=str, default=None,
                   help="Skip training; render the video for an existing run dir.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.render_only:
        run = Path(args.render_only)
        p = json.loads((run / "ppo_config.json").read_text())
        env = VisionLocomotionInsect(_env_config(args, 1))
        net = ppo_networks_vision.make_ppo_networks_vision(
            env.observation_size, env.action_size,
            preprocess_observations_fn=__import__(
                "brax.training.acme.running_statistics", fromlist=["normalize"]).normalize,
            **p["network_factory"],
        )
        make_inference_fn = ppo_networks_vision.make_inference_fn(net) if hasattr(
            ppo_networks_vision, "make_inference_fn") else __import__(
            "brax.training.agents.ppo.networks", fromlist=["make_inference_fn"]).make_inference_fn(net)
        render_rollout(make_inference_fn, model.load_params(str(run / "params.pkl")), run, args)
        return
    make_inference_fn, params, out_dir = train(args)
    import gc
    jax.clear_caches(); gc.collect()
    try:
        render_rollout(make_inference_fn, params, out_dir, args)
    except Exception as e:  # training results are already saved
        print(f"render_rollout skipped: {type(e).__name__}: {e}")
        print(f"re-render with: --render-only {out_dir}")


if __name__ == "__main__":
    main()
