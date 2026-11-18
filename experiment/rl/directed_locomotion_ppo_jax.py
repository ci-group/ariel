"""Directed (target-seeking) locomotion for ARIEL morphologies.

Trains a PPO policy with mujoco-playground + brax on GPU. Unlike
``undirected_locomotion_ppo_jax.py``, each episode has a goal: a target
point on the floor sampled at a random bearing and distance from the robot.
The core module is treated as a *head with eyes*:

* The head points along a fixed axis of the core body frame (``head_axis``).
  For every ARIEL morphology here the spine grows out of the core's FRONT
  face, which is the core's local +y. The core therefore sits at one end
  of the body, and by default the head looks *away* from the body
  (local -y), the way an animal's head leads its trunk.
* The policy sees the target only egocentrically: the bearing of the target
  relative to where the head points (cos, sin) and its distance, plus a
  ``visible`` flag. The eyes have a limited field of view (``--fov-deg``,
  default 120°): outside it the target is hidden and the policy has to turn
  to find it. The critic always sees the true target (``privileged_state``).
* Everything else in the observation is also expressed in the body / yaw
  frame (gravity vector, linear and angular velocity), so the policy does not
  depend on the robot's absolute world heading.

Reward = progress toward the target (Δdistance, paid only in proportion to
how well the head faces the target, so crab-walking / walking backwards
earns nothing; see ``gate_progress_by_heading``) + a heading term (cos of
the bearing) + a bonus each time the target is reached. On reaching it, a new target is sampled relative
to the robot's current position, so a good policy chains several targets per
episode.

Run from the mujoco_playground venv:

``experiment/mujoco_playground/.venv/bin/python \
    experiment/rl/directed_locomotion_ppo_jax.py --num-timesteps 50000000``
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
from matplotlib.patches import Circle
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground import wrapper
from mujoco_playground._src import locomotion, mjx_env

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.insect import (
    insect_small,
    insect_n,
    insect_longspine_n,
)
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.centipede import body_centipede_n

from ariel.simulation.environments import SimpleFlatWorld


ENV_NAME = "ArielInsectDirectedLocomotion"
TORSO_BODY_NAME = "robot1_core"
TARGET_BODY_NAME = "target"

# Head axis in the core's local frame. The spine always extends along the
# core's +y (FRONT face), so "-y" = head looks away from the body (default).
HEAD_AXES = {
    "-y": (0.0, -1.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
}


def _build_body(morphology: str, n_pairs: int):
    """Return the ariel body factory output for the requested morphology.

    For scaling variants (centipede, insect_n, insect_longspine_n) ``n_pairs``
    is reused as the single scaling parameter (n_pairs / n_segments).
    """
    if morphology == "insect":
        return insect_small()
    if morphology == "centipede":
        return body_centipede_n(n_pairs)
    if morphology == "insect_n":
        return insect_n(n_pairs)
    if morphology == "insect_longspine_n":
        return insect_longspine_n(n_pairs)
    raise ValueError(
        f"unknown morphology={morphology!r}; use 'insect', 'centipede', "
        "'insect_n' or 'insect_longspine_n'"
    )


def _discover_foot_body_names(mj_model: mujoco.MjModel) -> tuple[str, ...]:
    """Enumerate foot bodies for either morphology.

    - insect_small: six leaf bricks named ``robot1_brick_{2..7}brick``.
    - body_centipede_n / insect_n / insect_longspine_n: the terminal foot
      brick of each leg has a name ending in ``-FB-brick``.
    """
    names: list[str] = []
    for i in range(mj_model.nbody):
        n = mj_model.body(i).name
        if n.endswith("-FB-brick"):
            names.append(n)
            continue
        if n.startswith("robot1_brick_") and n.endswith("brick"):
            tail = n[len("robot1_brick_"):-len("brick")]
            if tail.isdigit() and int(tail) >= 2:
                names.append(n)
    if not names:
        raise ValueError(
            "Could not auto-discover foot bodies; check morphology naming."
        )
    return tuple(names)

# ---- Batch-size and solver-buffer budgets ---------------------------------- #
# Same budgets as the undirected script (see the notes there).
NUM_ENVS = 6144
NJMAX_PER_ENV = 800
NACONMAX_PER_ENV = 320
NACCDMAX_PER_ENV = 100

# ---- PPO batching, derived from NUM_ENVS ----------------------------------- #
NUM_MINIBATCHES = 16
BATCH_RATIO = 6
BATCH_SIZE = BATCH_RATIO * NUM_ENVS // NUM_MINIBATCHES


# ============================================================================ #
#                              MuJoCo model build                              #
# ============================================================================ #
def _build_insect_mj_model(
    sim_dt: float, morphology: str = "insect", n_pairs: int = 2,
    friction: float = 1.0,
    core_mass: float | None = None,
    extra_spec_fn: Any = None,
) -> mujoco.MjModel:
    """Compile the ariel body on a flat world, plus a visual-only target marker.

    ``extra_spec_fn(world_spec, core_body_spec)``, if given, is called just
    before compilation so derived envs can add cameras / objects. Mocap
    bodies added there are kept non-colliding automatically.
    """
    world = SimpleFlatWorld(load_precompiled=False)
    body = _build_body(morphology, n_pairs)
    world.spawn(body.spec, position=[0, 0, 0.1])

    def _find_body(parent, name):
        for b in getattr(parent, "bodies", []):
            if b.name == name:
                return b
            found = _find_body(b, name)
            if found is not None:
                return found
        return None

    # Tracking camera that follows the robot COM. Pulled back further than
    # in the undirected script so the target stays in frame more often.
    core_body = _find_body(world.spec.worldbody, TORSO_BODY_NAME)
    cam_parent = core_body if core_body is not None else world.spec.worldbody
    cam_parent.add_camera(
        name="track",
        mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM,
        pos=[0.0, -2.2, 1.8],
        xyaxes=[1, 0, 0, 0, 0.63, 0.78],
    )

    # Target marker: a mocap body whose position is written every step from
    # info["target"]. Its geom has contype=conaffinity=0 so it never
    # collides; it exists only so rendered videos show the goal.
    target = world.spec.worldbody.add_body(
        name=TARGET_BODY_NAME, mocap=True, pos=[0.0, 0.0, 0.0]
    )
    target.add_geom(
        name="target_marker",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.15, 0.005, 0.0],
        pos=[0.0, 0.0, 0.005],
        rgba=[1.0, 0.2, 0.2, 0.8],
        contype=0,
        conaffinity=0,
    )

    if extra_spec_fn is not None:
        extra_spec_fn(world.spec, core_body)

    mj_model = world.spec.compile()
    mj_model.opt.timestep = sim_dt
    mj_model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    mj_model.opt.iterations = 6
    mj_model.opt.ls_iterations = 12
    mj_model.vis.global_.offwidth = 1280
    mj_model.vis.global_.offheight = 720

    # Enable self-collisions and floor contacts for every robot/floor geom
    # (parent-child pairs are still auto-excluded by MuJoCo). Mocap bodies
    # (target marker, any visual-only objects) are left non-colliding.
    for i in range(mj_model.ngeom):
        if mj_model.body_mocapid[mj_model.geom_bodyid[i]] >= 0:
            continue
        mj_model.geom_contype[i] = 1
        mj_model.geom_conaffinity[i] = 1
        # Sliding friction for every geom (floor AND robot). MuJoCo combines
        # a contact pair's friction with max(), so lowering only the floor
        # would be overridden by the robot geoms' default 1.0. Default 1.0 =
        # MuJoCo's default; ~0.4 approximates PLA on linoleum (non-grippy).
        mj_model.geom_friction[i, 0] = friction

    # Optional core mass override (e.g. heavier battery). Inertia is scaled
    # by the same factor: same cube geometry, uniformly denser.
    if core_mass is not None:
        cid = mj_model.body(TORSO_BODY_NAME).id
        scale = core_mass / mj_model.body_mass[cid]
        mj_model.body_mass[cid] = core_mass
        mj_model.body_inertia[cid] *= scale
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
        # ---- Target / head ------------------------------------------------ #
        head_axis="-y",                 # key into HEAD_AXES (core local frame)
        fov_deg=120.0,                  # eyes' field of view; 360 = always visible
        gate_progress_by_heading=True,  # progress pays ∝ max(cos bearing, 0)
        target_dist_min=1.0,            # m; good undirected policies cover ~3 m / 10 s
        target_dist_max=2.5,
        target_reach_radius=0.3,        # m; reaching resamples a new target
        obs_dist_clip=5.0,              # distance observation is clipped to this
        reward_config=config_dict.create(
            stall_speed=0.05,
            air_time_target=0.1,
            # One-off bonus per target reached (NOT multiplied by dt).
            reach_bonus=5.0,
            scales=config_dict.create(
                # Δ(distance to target)/dt; telescopes to total distance
                # closed, so it can't be farmed by oscillating.
                target_progress=10.0,
                # cos(bearing): +1 head faces target, -1 facing away.
                # Per-episode max = 0.5 * 10 s = 5, i.e. worth 0.5 m of
                # progress, so it shapes orientation without dominating.
                heading=0.5,
                stall=-3.0,
                upright=0.01,
                action_rate=-0.1,
                torques=-1e-4,
                joint_vel=-5e-4,
                lin_vel_z=-1.0,
                ang_vel_xy=-0.2,
                # Lowered 10x vs undirected (-1.0): the robot now has to
                # turn to face the target, which this term directly taxes.
                head_yaw_rate=-0.1,
                feet_air_time=1.0,
            ),
        ),
        impl="warp",
        njmax=NJMAX_PER_ENV,
        naconmax=NUM_ENVS * NACONMAX_PER_ENV,
        naccdmax=NUM_ENVS * NACCDMAX_PER_ENV,
        morphology="insect",
        n_pairs=2,
        friction=1.0,                                  # sliding μ (MuJoCo default); ~0.4 ≈ PLA on linoleum
        core_mass=None,                                # kg; None = ARIEL default (1 kg)
    )


class DirectedLocomotionInsect(mjx_env.MjxEnv):
    """Walk the head (core) toward a randomly placed target."""

    def __init__(
        self,
        config: Optional[config_dict.ConfigDict] = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(config or default_config(), config_overrides)

        self._mj_model = _build_insect_mj_model(
            self._config.sim_dt,
            morphology=self._config.morphology,
            n_pairs=self._config.n_pairs,
            friction=self._config.friction,
            core_mass=self._config.core_mass,
        )
        self._mjx_model = mjx_env.put_model(self._mj_model, impl=self._config.impl)
        self._init_common()

    def _init_common(self) -> None:
        """Everything after the model is built (shared with derived envs that
        build their own model, e.g. the vision env)."""
        self._xml_path = ""
        self._init_qpos = jp.array(self._mj_model.qpos0)
        self._torso_body_id = self._mj_model.body(TORSO_BODY_NAME).id
        self._target_mocap_id = int(
            self._mj_model.body_mocapid[self._mj_model.body(TARGET_BODY_NAME).id]
        )
        self._nu = self._mj_model.nu
        self._nv = self._mj_model.nv
        # Sanity: the free joint must be the first joint so qpos[:7]/qvel[:6]
        # are the root. (It lives on the massless "robot1_world" body, which
        # shares the core's frame — checked when this script was written.)
        assert self._mj_model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
        assert self._mj_model.nq - 7 == self._nu

        if self._config.head_axis not in HEAD_AXES:
            raise ValueError(f"head_axis must be one of {list(HEAD_AXES)}")
        self._head_local = jp.array(HEAD_AXES[self._config.head_axis])
        self._half_fov = jp.deg2rad(self._config.fov_deg) / 2.0

        foot_names = _discover_foot_body_names(self._mj_model)
        self._foot_body_ids = jp.array(
            [self._mj_model.body(n).id for n in foot_names]
        )
        self._n_feet = int(self._foot_body_ids.shape[0])
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

    @property
    def torso_body_id(self) -> int:
        return self._torso_body_id

    # ---- Target helpers --------------------------------------------------- #
    def _sample_target(self, rng: jax.Array, origin_xy: jax.Array) -> jax.Array:
        """Uniform bearing, uniform distance in [min, max] around origin_xy."""
        a_rng, d_rng = jax.random.split(rng)
        angle = jax.random.uniform(a_rng, (), minval=-jp.pi, maxval=jp.pi)
        dist = jax.random.uniform(
            d_rng, (),
            minval=self._config.target_dist_min,
            maxval=self._config.target_dist_max,
        )
        return origin_xy + dist * jp.array([jp.cos(angle), jp.sin(angle)])

    def _head_bearing(self, data: mjx.Data, target: jax.Array):
        """Egocentric view of the target from the head.

        Returns (cos_bearing, sin_bearing, distance). The bearing is the
        signed yaw angle from the head direction to the target, measured in
        the horizontal plane (positive = target to the head's left).
        """
        rot = data.xmat[self._torso_body_id]               # body -> world
        head_world = rot @ self._head_local
        head_xy = head_world[:2]
        head_xy = head_xy / (jp.linalg.norm(head_xy) + 1e-6)

        to_target = target - data.xpos[self._torso_body_id, :2]
        dist = jp.linalg.norm(to_target)
        to_target_dir = to_target / (dist + 1e-6)

        cos_b = jp.dot(head_xy, to_target_dir)
        sin_b = head_xy[0] * to_target_dir[1] - head_xy[1] * to_target_dir[0]
        return cos_b, sin_b, dist

    def _write_target_marker(self, data: mjx.Data, target: jax.Array) -> mjx.Data:
        mocap_pos = data.mocap_pos.at[self._target_mocap_id, :2].set(target)
        return data.replace(mocap_pos=mocap_pos)

    def _fresh_episode_info(
        self, rng: jax.Array, data: mjx.Data
    ) -> dict[str, Any]:
        # Deterministic in `rng` (which is stored, not advanced): step() calls
        # this again on the first step with info["rng"], and for a genuine
        # reset that reproduces exactly the target reset() chose.
        _, target_rng = jax.random.split(rng)
        target = self._sample_target(target_rng, data.xpos[self._torso_body_id, :2])
        _, _, dist = self._head_bearing(data, target)
        return {
            "rng": rng,
            "last_act": jp.zeros(self._nu),
            "feet_air_time": jp.zeros(self._n_feet),
            "last_contact": jp.zeros(self._n_feet, dtype=bool),
            "target": target,
            # Distance to target at the previous step for the potential-based
            # progress reward.
            "prev_dist": dist,
        }

    # ---- Rollout ---------------------------------------------------------- #
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
            naccdmax=self._config.naccdmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        info = self._fresh_episode_info(rng, data)
        data = self._write_target_marker(data, info["target"])

        metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward_config.scales.keys()}
        metrics["reward/reached"] = jp.zeros(())
        metrics["distance_to_target"] = jp.zeros(())
        metrics["reached"] = jp.zeros(())

        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        info = dict(state.info)

        # ---- Episode-start re-initialisation ----------------------------- #
        # The brax auto-reset wrapper (full_reset=False) swaps in a cached
        # first `data`/`obs` on done but leaves `info` untouched. Without
        # this block the target would never be resampled and prev_dist would
        # carry over from the previous episode (spurious progress reward on
        # the first step). A freshly reset `data` has time == 0 exactly.
        # After an auto-reset, info["rng"] has been advanced by every step of
        # the previous episode, so the new target differs. Caveat: the cached
        # first obs still shows the env's very first target, so the policy's
        # first action of a re-used episode sees a stale target (1/500 steps).
        is_first = state.data.time < 0.5 * self._config.sim_dt
        fresh = self._fresh_episode_info(info["rng"], state.data)
        for k, v in fresh.items():
            if k == "rng":
                continue
            info[k] = jp.where(is_first, v, info[k])

        motor_targets = jp.clip(
            action * self._config.action_scale, -self._config.action_scale, self._config.action_scale
        )
        data = mjx_env.step(
            self.mjx_model, state.data, motor_targets, self.n_substeps
        )

        contact = self._feet_contact(data)
        air_time_prev = info["feet_air_time"]
        air_time = (air_time_prev + self.dt) * (~contact)
        first_contact = contact & (air_time_prev > 0.0)

        cos_b, _, dist = self._head_bearing(data, info["target"])
        rewards = self._compute_rewards(
            data, action, info, air_time_prev, first_contact, cos_b, dist
        )
        reward = jp.sum(
            jp.array([
                v * self._config.reward_config.scales[k]
                for k, v in rewards.items()
            ])
        ) * self.dt

        # Target reached -> bonus, then resample a new target around the
        # current position and reset the progress baseline to it.
        reached = dist < self._config.target_reach_radius
        reach_reward = reached * self._config.reward_config.reach_bonus
        reward = reward + reach_reward
        rng, tgt_rng = jax.random.split(info["rng"])
        info["rng"] = rng
        new_target = self._sample_target(tgt_rng, data.xpos[self._torso_body_id, :2])
        target = jp.where(reached, new_target, info["target"])
        _, _, new_dist = self._head_bearing(data, target)

        up_proj = data.xmat[self._torso_body_id, 2, 2]
        done = up_proj < self._config.upright_termination_z
        nan_state = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = done | nan_state

        reward = jp.where(jp.isnan(reward) | nan_state, 0.0, reward)
        reward = jp.clip(reward, -10.0, 10.0)

        # Bookkeeping — update info BEFORE building obs so the observation
        # carries the action just applied and the (possibly new) target.
        info["last_act"] = action
        info["feet_air_time"] = air_time
        info["last_contact"] = contact
        info["target"] = target
        info["prev_dist"] = new_dist
        data = self._write_target_marker(data, target)
        obs = self._get_obs(data, info)

        metrics = dict(state.metrics)
        for k, v in rewards.items():
            metrics[f"reward/{k}"] = v
        metrics["reward/reached"] = reach_reward
        metrics["distance_to_target"] = dist
        metrics["reached"] = reached.astype(jp.float32)

        return state.replace(
            data=data,
            obs=obs,
            reward=reward,
            done=done.astype(jp.float32),
            metrics=metrics,
            info=info,
        )

    def _feet_contact(self, data: mjx.Data) -> jax.Array:
        """Boolean mask of shape (n_feet,): True where foot-i is near floor."""
        foot_z = data.xpos[self._foot_body_ids, 2]
        return foot_z < self._foot_contact_z

    # ---- Observations and rewards ---------------------------------------- #
    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> Dict[str, jax.Array]:
        rot = data.xmat[self._torso_body_id]                  # body -> world
        joint_angles = data.qpos[7:]
        joint_vel = data.qvel[6:]

        # All proprioception in the body frame: no absolute yaw leaks in.
        gravity_body = rot.T @ jp.array([0.0, 0.0, -1.0])    # (3,)
        linvel_body = rot.T @ data.qvel[0:3]                  # free-joint lin vel is world frame
        angvel_body = data.qvel[3:6]                          # free-joint ang vel is already body frame

        phase = jp.array([
            jp.sin(2.0 * jp.pi * data.time),
            jp.cos(2.0 * jp.pi * data.time),
        ])

        # "Eyes": egocentric target, masked outside the field of view.
        cos_b, sin_b, dist = self._head_bearing(data, info["target"])
        bearing = jp.arctan2(sin_b, cos_b)
        visible = (jp.abs(bearing) <= self._half_fov).astype(jp.float32)
        target_obs = visible * jp.array([
            cos_b,
            sin_b,
            jp.minimum(dist, self._config.obs_dist_clip),
        ])

        state = jp.concatenate([
            jp.array([data.qpos[2]]),  # torso height
            gravity_body,
            joint_angles,
            joint_vel,
            linvel_body,
            angvel_body,
            info["last_act"],
            phase,
            target_obs,
            jp.array([visible]),
        ])
        # Critic-only ("privileged") observation: the policy's state plus the
        # UNMASKED target. With a limited FOV the actor must search for the
        # target, but the value function still knows where it is, so its
        # estimates don't turn to noise whenever the target leaves view.
        privileged = jp.concatenate([
            state,
            jp.array([cos_b, sin_b, jp.minimum(dist, self._config.obs_dist_clip)]),
        ])

        def _clean(x):
            x = jp.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
            return jp.clip(x, -10.0, 10.0)

        return {"state": _clean(state), "privileged_state": _clean(privileged)}

    def _compute_rewards(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        air_time_prev: jax.Array,
        first_contact: jax.Array,
        cos_bearing: jax.Array,
        dist: jax.Array,
    ) -> dict[str, jax.Array]:
        xy_speed = jp.linalg.norm(data.qvel[0:2])
        up_proj = data.xmat[self._torso_body_id, 2, 2]
        stall_speed = self._config.reward_config.stall_speed
        air_time_target = self._config.reward_config.air_time_target

        # Progress toward the target (Δdistance/dt). With the heading gate,
        # approaching only pays in proportion to how well the head faces the
        # target (cos bearing, clipped at 0): head-first walking earns full
        # credit, sideways ~0, backwards 0. Moving AWAY is always penalised
        # in full, so no approach/retreat cycle can net positive reward.
        progress = (info["prev_dist"] - dist) / self.dt
        if self._config.gate_progress_by_heading:
            facing = jp.clip(cos_bearing, 0.0, 1.0)
            progress = jp.where(progress > 0.0, progress * facing, progress)

        moving = (xy_speed > stall_speed).astype(jp.float32)
        feet_air_time_reward = (
            jp.sum((air_time_prev - air_time_target) * first_contact) * moving
        )

        return {
            "target_progress": progress,
            "heading": cos_bearing,
            "stall": jp.clip(stall_speed - xy_speed, 0.0, stall_speed),
            "upright": jp.clip(up_proj, 0.0, 1.0),
            "action_rate": jp.sum(jp.square(action - info["last_act"])),
            "torques": jp.sum(jp.square(data.actuator_force)),
            "joint_vel": jp.sum(jp.square(data.qvel[6:])),
            "lin_vel_z": jp.square(data.qvel[2]),
            "ang_vel_xy": jp.sum(jp.square(data.qvel[3:5])),
            "head_yaw_rate": jp.square(data.qvel[5]),
            "feet_air_time": feet_air_time_reward,
        }


locomotion.register_environment(
    ENV_NAME, DirectedLocomotionInsect, default_config
)


# ============================================================================ #
#                                PPO config                                    #
# ============================================================================ #
def ppo_config(num_timesteps: int, episode_length: int) -> config_dict.ConfigDict:
    return config_dict.create(
        num_timesteps=num_timesteps,
        num_evals=10,
        reward_scaling=1.0,
        episode_length=episode_length,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=8,
        num_minibatches=NUM_MINIBATCHES,
        num_updates_per_batch=4,
        discounting=0.99,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        num_envs=NUM_ENVS,
        num_eval_envs=128,
        batch_size=BATCH_SIZE,
        max_grad_norm=1.0,
        num_resets_per_eval=10,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(32, 32),
            value_hidden_layer_sizes=(64, 64),
            policy_obs_key="state",
            value_obs_key="privileged_state",   # asymmetric critic: sees target outside FOV
        ),
    )


# ============================================================================ #
#                                  Training                                    #
# ============================================================================ #
def _env_config_from_args(args: argparse.Namespace, n_worlds: int, nacn: int, nacc: int):
    cfg = default_config()
    cfg.morphology = args.morphology
    cfg.n_pairs = args.n_pairs
    cfg.episode_length = args.episode_length
    cfg.head_axis = args.head_axis
    cfg.fov_deg = args.fov_deg
    cfg.gate_progress_by_heading = not args.no_heading_gate
    cfg.target_dist_min = args.target_dist_min
    cfg.target_dist_max = args.target_dist_max
    cfg.friction = args.friction
    cfg.core_mass = args.core_mass
    cfg.naconmax = n_worlds * nacn
    cfg.naccdmax = n_worlds * nacc
    return cfg


def train(args: argparse.Namespace) -> tuple[Any, Any, Path]:
    print(f"jax backend: {jax.default_backend()}  devices: {jax.devices()}")

    num_envs = args.num_envs if args.num_envs is not None else NUM_ENVS
    nacc = args.naccdmax_per_env if args.naccdmax_per_env is not None else NACCDMAX_PER_ENV
    nacn = args.naconmax_per_env if args.naconmax_per_env is not None else NACONMAX_PER_ENV
    print(f"NUM_ENVS={num_envs}  NACONMAX/env={nacn}  NACCDMAX/env={nacc}")

    rl_cfg = ppo_config(args.num_timesteps, args.episode_length)
    rl_cfg.num_envs = num_envs
    rl_cfg.batch_size = BATCH_RATIO * num_envs // NUM_MINIBATCHES
    rl_cfg.num_evals = args.num_evals

    if (args.hidden_width is None) != (args.n_hidden_layers is None):
        raise SystemExit(
            "--hidden-width and --n-hidden-layers must be set together"
        )
    if args.hidden_width is not None:
        if args.n_hidden_layers < 0:
            raise SystemExit("--n-hidden-layers must be >= 0")
        if args.hidden_width <= 0:
            raise SystemExit("--hidden-width must be > 0")
        layer_sizes = tuple([args.hidden_width] * args.n_hidden_layers)
        rl_cfg.network_factory.policy_hidden_layer_sizes = layer_sizes
        if args.value_hidden_layer_sizes is None:
            rl_cfg.network_factory.value_hidden_layer_sizes = layer_sizes
            print(f"arch override: tied policy/value hidden_layer_sizes={layer_sizes}")
        else:
            print(f"arch override: policy hidden_layer_sizes={layer_sizes}")
    # Fixed critic, independent of the policy architecture (capacity sweeps:
    # keeps a tiny policy from also getting a tiny, failing value function).
    if args.value_hidden_layer_sizes is not None:
        value_sizes = tuple(
            int(x) for x in args.value_hidden_layer_sizes.split(",") if x.strip()
        )
        if any(v <= 0 for v in value_sizes):
            raise SystemExit("--value-hidden-layer-sizes entries must be > 0")
        rl_cfg.network_factory.value_hidden_layer_sizes = value_sizes
        print(f"critic override: value hidden_layer_sizes={value_sizes}")

    env_cfg = _env_config_from_args(args, num_envs, nacn, nacc)
    env = DirectedLocomotionInsect(env_cfg)
    # Eval env sized for num_eval_envs worlds so it doesn't hold a
    # training-sized CCD workspace.
    eval_cfg = _env_config_from_args(args, int(rl_cfg.num_eval_envs), nacn, nacc)
    eval_env = DirectedLocomotionInsect(eval_cfg)

    tag_bits = [args.morphology]
    if args.morphology in ("centipede", "insect_n", "insect_longspine_n"):
        tag_bits.append(f"n{args.n_pairs}")
    if args.hidden_width is not None:
        tag_bits.append(f"w{args.hidden_width}")
        tag_bits.append(f"d{args.n_hidden_layers}")
    if args.fov_deg < 360.0:
        tag_bits.append(f"fov{int(args.fov_deg)}")
    if args.no_heading_gate:
        tag_bits.append("nogate")
    if args.core_mass is not None:
        tag_bits.append(f"core{args.core_mass:g}kg")
    tag_bits.append(f"seed{args.seed}")
    run_tag = "_".join(tag_bits)
    out_dir = (
        Path(args.outdir)
        / f"{run_tag}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "env_config.json").write_text(json.dumps(env_cfg.to_dict(), indent=2))
    (out_dir / "eval_env_config.json").write_text(json.dumps(eval_cfg.to_dict(), indent=2))
    (out_dir / "ppo_config.json").write_text(json.dumps(rl_cfg.to_dict(), indent=2))
    print(f"logging to {out_dir}")

    history: dict[str, list[float]] = {
        "steps": [], "reward": [], "reward_std": [], "targets_reached": [],
    }
    times = [time.monotonic()]
    snapshots: list[tuple[int, Any]] = []

    def progress(num_steps: int, metrics: dict[str, Any]) -> None:
        times.append(time.monotonic())
        reward = float(metrics.get("eval/episode_reward", float("nan")))
        reward_std = float(metrics.get("eval/episode_reward_std", float("nan")))
        # EpisodeWrapper sums per-step metrics -> targets reached per episode.
        reached = float(metrics.get("eval/episode_reached", float("nan")))
        history["steps"].append(int(num_steps))
        history["reward"].append(reward)
        history["reward_std"].append(reward_std)
        history["targets_reached"].append(reached)
        print(
            f"[{num_steps:>10}] reward={reward:+.3f} ± {reward_std:.3f}"
            f"  targets/ep={reached:.2f}"
            f"  elapsed={times[-1] - times[0]:7.1f}s",
            flush=True,
        )

    def snapshot_fn(current_step: int, _make_policy: Any, params_tuple: Any) -> None:
        snapshots.append(
            (int(current_step), jax.tree_util.tree_map(np.asarray, params_tuple))
        )

    train_params = dict(rl_cfg)
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **train_params.pop("network_factory")
    )
    num_eval_envs = train_params.pop("num_eval_envs")

    # Continue training from a previous run's final params (normalizer, policy,
    # value). The optimizer state is not saved, so Adam restarts; step counts
    # in training_history.json restart at 0 (offset recorded in restored_from.json).
    restore_kwargs = {}
    if args.restore_params is not None:
        restore_path = Path(args.restore_params)
        restore_kwargs["restore_params"] = model.load_params(str(restore_path))
        parent_hist = restore_path.parent / "training_history.json"
        offset = (json.loads(parent_hist.read_text())["steps"][-1]
                  if parent_hist.exists() else None)
        (out_dir / "restored_from.json").write_text(json.dumps(
            {"params": str(restore_path), "step_offset": offset}, indent=2))
        print(f"restoring params from {restore_path} (parent ended at step {offset})")

    train_fn = functools.partial(
        ppo.train,
        **train_params,
        network_factory=network_factory,
        seed=args.seed,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_eval_envs=num_eval_envs,
        policy_params_fn=snapshot_fn,
        **restore_kwargs,
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
    model.save_params(str(out_dir / "params.pkl"), params)
    model.save_params(str(out_dir / "params_snapshots.pkl"), snapshots)

    if history["steps"]:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        steps = np.array(history["steps"])
        reward = np.array(history["reward"])
        reward_std = np.array(history["reward_std"])
        ax.plot(steps, reward, color="tab:blue")
        ax.fill_between(steps, reward - reward_std, reward + reward_std, color="tab:blue", alpha=0.25)
        ax.set_xlabel("environment steps")
        ax.set_ylabel("eval episode reward")
        ax.set_title("Directed locomotion (PPO)")
        ax.grid(True, alpha=0.3)
        ax2.plot(steps, history["targets_reached"], color="tab:red")
        ax2.set_xlabel("environment steps")
        ax2.set_ylabel("targets reached / episode")
        ax2.grid(True, alpha=0.3)
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
    args: argparse.Namespace,
) -> None:
    rollout_cfg = _env_config_from_args(args, 1, NACONMAX_PER_ENV, NACCDMAX_PER_ENV)
    env = DirectedLocomotionInsect(rollout_cfg)
    inference_fn = make_inference_fn(params, deterministic=True)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(inference_fn)
    head_local = np.array(HEAD_AXES[args.head_axis])

    rng = jax.random.PRNGKey(args.seed + 100)
    state = jit_reset(rng)
    rollout = [state]

    def _record(s):
        core = np.asarray(s.data.xpos[env.torso_body_id])
        rot = np.asarray(s.data.xmat[env.torso_body_id])
        return core[:2], (rot @ head_local)[:2], np.asarray(s.info["target"])

    records = [_record(state)]
    n_reached = 0
    for _ in range(args.episode_length):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state)
        records.append(_record(state))
        n_reached += int(state.metrics["reached"])
        if bool(state.done):
            break

    final_dist = float(state.metrics["distance_to_target"])
    print(
        f"rollout length: {len(rollout)}  targets reached: {n_reached}"
        f"  final distance to current target: {final_dist:.3f} m"
    )

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

    # Trajectory plot: path, head-direction arrows, and every target visited.
    xy = np.array([r[0] for r in records])
    heads = np.array([r[1] for r in records])
    targets = np.array([r[2] for r in records])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xy[:, 0], xy[:, 1], "b-", linewidth=2, label="core trajectory")
    every = max(1, len(xy) // 25)
    ax.quiver(
        xy[::every, 0], xy[::every, 1], heads[::every, 0], heads[::every, 1],
        color="tab:orange", scale=15, width=0.004, label="head direction",
    )
    uniq_targets = np.unique(targets.round(4), axis=0)
    for i, t in enumerate(uniq_targets):
        ax.add_patch(Circle(t, float(rollout_cfg.target_reach_radius), color="r", alpha=0.25))
        ax.plot(t[0], t[1], "rx", markersize=10, label="target" if i == 0 else None)
    ax.plot(xy[0, 0], xy[0, 1], "go", markersize=10, label="start")
    ax.plot(xy[-1, 0], xy[-1, 1], "k*", markersize=14, label="end")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"directed locomotion — {n_reached} target(s) reached")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
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
        default=str(Path(__file__).parent.parent / "__data__" / "directed_locomotion_ppo_jax"),
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=500,
        help="Episode length (control steps) for training AND the rollout.",
    )
    parser.add_argument(
        "--morphology",
        choices=["insect", "centipede", "insect_n", "insect_longspine_n"],
        default="insect",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=2,
        help="Scaling parameter: n_pairs for centipede, n_segments for "
        "insect_n / insect_longspine_n. Ignored for fixed-size 'insect'.",
    )
    parser.add_argument(
        "--head-axis",
        choices=list(HEAD_AXES),
        default="-y",
        help="Core-local axis the head/eyes point along. '-y' = away from "
        "the spine (head leads the body); '+y' = along the spine.",
    )
    parser.add_argument(
        "--fov-deg",
        type=float,
        default=120.0,
        help="Eyes' field of view in degrees. Target is hidden from the "
        "policy when outside it. 360 = always visible.",
    )
    parser.add_argument(
        "--no-heading-gate",
        action="store_true",
        help="Pay target progress regardless of head direction (old behaviour).",
    )
    parser.add_argument(
        "--friction",
        type=float,
        default=1.0,
        help="Sliding friction for all geoms (floor and robot). Default 1.0 "
        "(MuJoCo default, grippy); ~0.4 ≈ PLA on linoleum.",
    )
    parser.add_argument(
        "--core-mass",
        type=float,
        default=None,
        help="Override the core module's mass in kg (ARIEL default 1.0). "
        "Inertia scales with it.",
    )
    parser.add_argument("--target-dist-min", type=float, default=1.0)
    parser.add_argument("--target-dist-max", type=float, default=2.5)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--naccdmax-per-env", type=int, default=None)
    parser.add_argument("--naconmax-per-env", type=int, default=None)
    parser.add_argument(
        "--hidden-width", type=int, default=None,
        help="Policy hidden width (with --n-hidden-layers). Also sets the "
        "critic unless --value-hidden-layer-sizes is given.",
    )
    parser.add_argument(
        "--n-hidden-layers", type=int, default=None,
        help="Policy hidden layers; 0 = linear policy.",
    )
    parser.add_argument(
        "--value-hidden-layer-sizes", type=str, default=None,
        help="Fix the critic independently of the policy, e.g. '256,256,256'.",
    )
    parser.add_argument(
        "--restore-params", type=str, default=None,
        help="params.pkl of a previous run to continue training from "
        "(must match the policy/critic architecture).",
    )
    parser.add_argument(
        "--num-evals", type=int, default=10,
        help="Number of evaluation points over training (incl. step 0).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    make_inference_fn, params, out_dir = train(args)
    import gc
    jax.clear_caches()
    gc.collect()
    try:
        render_rollout(make_inference_fn, params, out_dir, args)
    except Exception as e:
        print(f"render_rollout skipped: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
