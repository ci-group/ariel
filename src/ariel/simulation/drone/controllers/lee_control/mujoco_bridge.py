"""Drive a MuJoCo-spawned drone with airevolve's Lee geometric controller.

MuJoCo is the physics integrator (Z-up / ENU); Lee runs purely as a control
*policy* in NED. Each step the bridge:

1. Reads drone state from MuJoCo (``data.qpos`` / ``data.qvel``).
2. Converts ENU → NED (flip ``z`` of position/velocity, flip ``y`` of
   quaternion — same convention as example 15's NED→ENU return trip).
3. Forcibly writes the converted state into the underlying
   :class:`~ariel.simulation.drone.drone_interface.DroneInterface` and
   refreshes its cached state.
4. Calls :meth:`LeeGeometricControl.controller`, updating ``ctrl.w_cmd``
   (per-motor steady-state speed in rad/s).
5. Converts each ``w_cmd[i]`` to a normalised MuJoCo actuator command
   ``ctrl_i = (kf * w_cmd[i]²) / max_thrust`` (clipped to ``[0, 1]``),
   and writes ``data.ctrl[:]``.

The caller drives the simulation with :func:`mujoco.mj_step`. The Lee
controller never integrates the Python NED simulator state — the state
is overwritten from MuJoCo every step, so the two never diverge.

A helper :func:`spawn_blueprint_in_world` builds a mass-matched MuJoCo
model from a :class:`DroneBlueprint` (same mass-matching strategy as
``examples/d_drones/15_cppn_neat_circle_to_mujoco.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    import mujoco

    from ariel.body_phenotypes.drone.blueprint import DroneBlueprint
    from ariel.simulation.environments import SimpleFlatWorld


# ---------------------------------------------------------------------------
# Frame conversion helpers (ENU ↔ NED)
# ---------------------------------------------------------------------------

def enu_to_ned_pos(pos_enu: np.ndarray) -> np.ndarray:
    """Convert a Cartesian position vector from ENU (Z-up) to NED (Z-down)."""
    return np.array(
        [float(pos_enu[0]), float(pos_enu[1]), -float(pos_enu[2])],
        dtype=np.float64,
    )


def enu_to_ned_vel(vel_enu: np.ndarray) -> np.ndarray:
    """Convert a velocity vector from ENU to NED (negate the vertical)."""
    return enu_to_ned_pos(vel_enu)


def enu_to_ned_quat(quat_enu_wxyz: np.ndarray) -> np.ndarray:
    """Flip the ``y`` component of a MuJoCo (w, x, y, z) quaternion.

    Mirrors the inverse mapping used in example 15
    (``quat_enu[:, 2] = -quat_enu[:, 2]``) — the same axis flip is its
    own inverse, so MuJoCo's quaternion converts to NED by negating ``q.y``.
    """
    q = np.asarray(quat_enu_wxyz, dtype=np.float64).copy()
    q[2] = -q[2]
    return q


def quat_wxyz_to_euler_zyx(q: np.ndarray) -> tuple[float, float, float]:
    """Convert a (w, x, y, z) quaternion to (roll, pitch, yaw) ZYX intrinsic."""
    w, x, y, z = (float(v) for v in q)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = float(np.arctan2(sinr_cosp, cosr_cosp))

    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = float(np.arcsin(sinp))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = float(np.arctan2(siny_cosp, cosy_cosp))
    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# Mass-matched spawn helper
# ---------------------------------------------------------------------------

@dataclass
class SpawnedDrone:
    """Container for the MuJoCo model/data plus mass-matching diagnostics."""
    world: "SimpleFlatWorld"
    model: "mujoco.MjModel"
    data: "mujoco.MjData"
    body_name: str
    core_mass: float
    motor_mass_each: float
    arm_mass_each: float
    n_motors: int
    max_thrust_per_motor: float


def spawn_blueprint_in_world(
    bp: "DroneBlueprint",
    *,
    propellers: list[dict],
    target_mass: float,
    spawn_position: tuple[float, float, float] = (0.0, 0.0, 1.0),
    body_name: str = "drone",
    arm_mass_per_length: float = 0.034,
    safety_margin: float = 1.10,
) -> SpawnedDrone:
    """Spawn a drone in a ``SimpleFlatWorld`` with masses matched to Lee's params.

    The MuJoCo body's lumped mass equals ``target_mass`` (Lee's ``mB``):
    motors come from the propeller config, arms scale with their length,
    and the core absorbs the remainder. Per-motor max thrust is sized to
    ``safety_margin * kf * wmax²`` so the bridge's saturation matches Lee's.

    Parameters
    ----------
    bp
        Blueprint to compile to MJCF.
    propellers
        The list returned by ``blueprint_to_propellers(bp, convention='ned')``.
        Used here only to fetch the prop ``mass``, ``constants`` and
        ``wmax`` for actuator sizing.
    target_mass
        Drone total mass (Lee's ``quad.params['mB']``).
    spawn_position
        ENU spawn position in metres.
    body_name
        Root body name in the MJCF.
    arm_mass_per_length
        Linear-density coefficient for the arm visual mass.
    safety_margin
        ``max_thrust`` per actuator is scaled by this factor over
        ``kf * wmax²`` to give MuJoCo a small extra ceiling.

    Returns
    -------
    SpawnedDrone
    """
    import mujoco

    from ariel.body_phenotypes.drone.backends import blueprint_to_mjspec
    from ariel.body_phenotypes.drone.blueprint import ArmNode
    from ariel.simulation.environments import SimpleFlatWorld

    if not propellers:
        msg = "spawn_blueprint_in_world requires a non-empty propellers list."
        raise ValueError(msg)

    arm_ids = [a for a in bp.children(bp.root_id) if isinstance(bp.payload(a), ArmNode)]
    arm_lengths = [bp.payload(a).length for a in arm_ids]
    mean_arm_len = float(np.mean(arm_lengths)) if arm_lengths else 0.06
    n_motors = len(propellers)
    motor_mass_each = float(propellers[0]["mass"])
    arm_mass_each = arm_mass_per_length * mean_arm_len
    core_mass = max(1e-4, target_mass - n_motors * (motor_mass_each + arm_mass_each))

    kf = float(propellers[0]["constants"][0])
    wmax = float(propellers[0]["wmax"])
    max_thrust_per_motor = safety_margin * kf * wmax * wmax

    spec = blueprint_to_mjspec(
        bp,
        motor_mass=motor_mass_each,
        arm_mass=arm_mass_each,
        core_mass_override=core_mass,
        max_thrust=max_thrust_per_motor,
        body_name=body_name,
    )

    world = SimpleFlatWorld()
    world.spawn(spec, position=spawn_position, correct_collision_with_floor=False)
    model = world.spec.compile()
    data = mujoco.MjData(model)

    return SpawnedDrone(
        world=world,
        model=model,
        data=data,
        body_name=body_name,
        core_mass=core_mass,
        motor_mass_each=motor_mass_each,
        arm_mass_each=arm_mass_each,
        n_motors=n_motors,
        max_thrust_per_motor=max_thrust_per_motor,
    )


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

class LeeMujocoHoverBridge:
    """Per-step Lee → MuJoCo control bridge for a single drone.

    Build one of these once per individual, then call :meth:`step` inside
    your MuJoCo simulation loop. ``data.ctrl[:]`` is updated in place; the
    caller is responsible for invoking :func:`mujoco.mj_step` afterwards.
    """

    def __init__(
        self,
        *,
        quad,                         # DroneInterface
        lee_ctrl,                     # LeeGeometricControl (orient="NED")
        model: "mujoco.MjModel",
        data: "mujoco.MjData",
        max_thrust_per_motor: float,
        ctrl_type: str = "xyz_pos",
        target_position_enu: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 1.0),
        target_yaw: float = 0.0,
        timestep: Optional[float] = None,
    ) -> None:
        self.quad = quad
        self.lee_ctrl = lee_ctrl
        self.model = model
        self.data = data
        self.max_thrust_per_motor = float(max_thrust_per_motor)
        self.ctrl_type = ctrl_type
        self.target_position_enu = np.asarray(target_position_enu, dtype=np.float64)
        self.target_yaw = float(target_yaw)
        self.timestep = float(timestep if timestep is not None else model.opt.timestep)

        # Pre-build the desired-state vector (Lee expects 19 elements):
        #   [pos(3), vel(3), acc(3), thrust(3), eul(3), pqr(3), yawRate]
        target_ned = enu_to_ned_pos(self.target_position_enu)
        self._sDes = np.zeros(19, dtype=np.float64)
        self._sDes[0:3] = target_ned
        self._sDes[12:15] = (0.0, 0.0, self.target_yaw)

        # Cache thrust normalisation constants from the propeller config.
        first_prop = self.quad.drone_sim.config.propellers[0]
        self._kf = float(first_prop["constants"][0])

    # ------------------------------------------------------------------
    # State conversion
    # ------------------------------------------------------------------

    def _push_mujoco_state_into_quad(self) -> None:
        """Copy MuJoCo state into ``quad`` (NED frame) without round-tripping
        through Euler angles.

        Lee reads ``quad.pos``, ``quad.vel``, ``quad.quat``, ``quad.omega``
        and ``quad.euler``. We populate all five from the MuJoCo state
        directly so we never pass through ``DroneSimulator.set_state``'s
        Euler representation (which loses precision near gimbal lock and
        was the source of a slow hover divergence).
        """
        pos_enu = np.asarray(self.data.qpos[0:3], dtype=np.float64).copy()
        quat_enu_wxyz = np.asarray(self.data.qpos[3:7], dtype=np.float64).copy()
        vel_enu = np.asarray(self.data.qvel[0:3], dtype=np.float64).copy()
        # For a free joint, qvel[3:6] holds the body-frame angular velocity.
        angvel_body = np.asarray(self.data.qvel[3:6], dtype=np.float64).copy()

        # Renormalise the quaternion before applying — MuJoCo integration
        # can drift its norm by a small epsilon over many steps.
        n = float(np.linalg.norm(quat_enu_wxyz))
        if n > 1e-9:
            quat_enu_wxyz /= n

        pos_ned = enu_to_ned_pos(pos_enu)
        vel_ned = enu_to_ned_vel(vel_enu)
        quat_ned = enu_to_ned_quat(quat_enu_wxyz)
        roll, pitch, yaw = quat_wxyz_to_euler_zyx(quat_ned)
        euler = np.array([roll, pitch, yaw], dtype=np.float64)

        # Bypass DroneSimulator.set_state() (which stores Euler and rebuilds
        # quat from it) — write the canonical Lee-readable fields directly.
        self.quad.pos = pos_ned
        self.quad.vel = vel_ned
        self.quad.quat = quat_ned
        self.quad.omega = angvel_body
        self.quad.euler = euler
        self.quad.phi, self.quad.theta, self.quad.psi = roll, pitch, yaw

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self) -> np.ndarray:
        """Update ``data.ctrl[:]`` for one step. Returns the issued ctrl vector.

        Does NOT advance the simulation — call :func:`mujoco.mj_step` yourself.
        """
        self._push_mujoco_state_into_quad()

        self.lee_ctrl.controller(self._sDes, self.quad, self.ctrl_type, self.timestep)

        w_cmd = np.asarray(self.lee_ctrl.w_cmd, dtype=np.float64)
        thrust_per_motor = self._kf * w_cmd * w_cmd
        ctrl = np.clip(thrust_per_motor / self.max_thrust_per_motor, 0.0, 1.0)

        n = min(ctrl.size, self.data.ctrl.size)
        self.data.ctrl[:n] = ctrl[:n]
        if self.data.ctrl.size > n:
            self.data.ctrl[n:] = 0.0
        return self.data.ctrl.copy()

    # ------------------------------------------------------------------
    # Convenience: full hover rollout
    # ------------------------------------------------------------------

    def reset_pose(
        self,
        *,
        position_enu: np.ndarray | tuple[float, float, float] | None = None,
        quat_wxyz: np.ndarray | tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    ) -> None:
        """Reset the MuJoCo drone to an identity attitude (or supplied pose).

        Always zeroes ``qvel``. Useful before a hover rollout so the drone
        doesn't carry over any residual velocity from the previous run.
        """
        import mujoco

        mujoco.mj_resetData(self.model, self.data)
        if position_enu is None:
            position_enu = self.target_position_enu
        self.data.qpos[0:3] = [float(v) for v in position_enu]
        self.data.qpos[3:7] = [float(v) for v in quat_wxyz]
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def run_hover(
        self,
        *,
        duration: float,
        warm_up: float = 0.0,
        log_pose: bool = False,
        reset: bool = True,
    ) -> dict[str, np.ndarray]:
        """Step the MuJoCo simulation for ``duration`` seconds.

        Lee is active throughout (including any ``warm_up`` window) so the
        drone never free-falls. The warm-up only governs *logging*: poses
        within it are discarded, so initial controller transients don't
        contaminate the fitness signal. When ``reset=True`` (default) the
        drone is placed at the target pose with zero velocity before stepping.

        Returns a dict with logged arrays (always 'pos', 'tilt_cos',
        'ctrl_norm'; 'quat' added when ``log_pose=True``).
        """
        import mujoco

        if reset:
            self.reset_pose()

        steps_total = int(round(duration / self.timestep))
        steps_warm = int(round(max(0.0, warm_up) / self.timestep))

        n_logged = max(0, steps_total - steps_warm)
        pos_log = np.zeros((n_logged, 3), dtype=np.float64)
        tilt_log = np.zeros(n_logged, dtype=np.float64)
        ctrl_norm_log = np.zeros(n_logged, dtype=np.float64)
        quat_log = np.zeros((n_logged, 4), dtype=np.float64) if log_pose else None

        log_idx = 0
        for step_i in range(steps_total):
            self.step()
            mujoco.mj_step(self.model, self.data)

            if step_i >= steps_warm:
                pos_log[log_idx] = self.data.qpos[0:3]
                w, x, y, z = self.data.qpos[3:7]
                # Tilt cosine: world-+Z projected onto body-+Z. For (w,x,y,z),
                # the (3,3) entry of the rotation matrix is 1 - 2(x² + y²).
                tilt_log[log_idx] = 1.0 - 2.0 * (x * x + y * y)
                ctrl_norm_log[log_idx] = float(np.mean(self.data.ctrl ** 2))
                if quat_log is not None:
                    quat_log[log_idx] = (w, x, y, z)
                log_idx += 1

        out: dict[str, np.ndarray] = {
            "pos": pos_log[:log_idx],
            "tilt_cos": tilt_log[:log_idx],
            "ctrl_norm": ctrl_norm_log[:log_idx],
        }
        if quat_log is not None:
            out["quat"] = quat_log[:log_idx]
        return out


# ---------------------------------------------------------------------------
# High-level fitness computation
# ---------------------------------------------------------------------------

def hover_fitness_from_log(
    log: dict[str, np.ndarray],
    *,
    target_position_enu: np.ndarray | tuple[float, float, float],
    drift_weight: float = 1.0,
    tilt_weight: float = 1.0,
    ctrl_weight: float = 0.05,
) -> float:
    """Combined hover-stability fitness (lower is better).

    ::

        fit = mean(|z - z_target|)
              + drift_weight * mean(sqrt((x-xt)² + (y-yt)²))
              + tilt_weight  * mean(1 - cos θ_tilt)
              + ctrl_weight  * mean(ctrl²)

    Spawn failures (empty log) return ``+inf``.
    """
    pos = log["pos"]
    if pos.size == 0:
        return float("inf")
    target = np.asarray(target_position_enu, dtype=np.float64)
    dz = np.abs(pos[:, 2] - target[2])
    dxy = np.sqrt((pos[:, 0] - target[0]) ** 2 + (pos[:, 1] - target[1]) ** 2)
    tilt_dev = 1.0 - np.clip(log["tilt_cos"], -1.0, 1.0)
    return float(
        np.mean(dz)
        + drift_weight * np.mean(dxy)
        + tilt_weight * np.mean(tilt_dev)
        + ctrl_weight * float(np.mean(log["ctrl_norm"])),
    )
