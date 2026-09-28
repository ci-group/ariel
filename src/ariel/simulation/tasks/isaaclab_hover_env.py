"""Isaac Lab hover-to-goal task driven by a DroneBlueprint.

Phase 2 of the pluggable-simulator effort. Adapts Isaac Lab's reference
``QuadcopterEnv`` (``isaaclab_tasks/direct/quadcopter/quadcopter_env.py``)
with one substitution: the articulation USD is generated at runtime
from a :class:`DroneBlueprint` (via ``blueprint_to_urdf`` +
``UrdfConverter``) instead of being a hard-coded Crazyflie asset.

Architecture (see DRONE_BLUEPRINT_PLAN.md §6 entry 17):

* Trained with ``rl_games`` PPO (Isaac Lab's most-stable native RL
  library on the current install), not stable-baselines3. The
  two-Protocols-one-trainer-per-backend decision means each backend
  brings its own RL library; we use rl_games here because (a) it's
  native to Isaac Lab's DirectRLEnv shape, (b) it avoids the numpy-2
  ABI issues that stable-baselines3 has in the unified isaaclab
  conda env, and (c) it's stable on the actual installed library
  version while ``isaaclab_rl.rsl_rl`` was caught between rsl-rl-lib
  3.x and 5.x API drift in our smoke tests.
* Action space, chosen by ``cfg.action_mode``:

  - ``"mixer"`` (default): collective thrust + 3 body torques (4-D for
    any rotor count; a = 0 is hover, torque ±1 = fixed N·m values in the
    cfg). The blueprint's allocation matrix (about the CoM, read from the
    spawned articulation) turns them into per-rotor thrusts, clipped to
    each rotor's physical range, then the ``"rotor"`` physics below
    applies them — as a flight controller's mixer would. On 2026-09-28
    PPO learned it reliably (default quad, 200 epochs × 64 envs: 45.0 /
    48.6 / 53.8 over seeds 42-44), unlike direct ``"rotor"`` control
    (4 of 9 runs took off at the same budget).
  - ``"rotor"``: one command per motor. Each motor link
    gets thrust ``k_f·ω²`` along its local +Z and reaction torque
    ``∓k_m·ω²`` about it, with the NumPy backend's rotor model
    (sqrt-polynomial command→speed map, first-order lag τ) and physical
    constants from ``propeller_data`` by propsize. Action 0 is the
    drone's own hover throttle (piecewise-linear map, full range kept). Rotor positions,
    thrust directions, spins, propeller size and mass therefore all
    shape what the policy can do.
  - ``"wrench"``: Isaac Lab ``QuadcopterEnv``'s total thrust + 3 body
    moments at ``base_link``, thrust scaled to the drone's weight. Rotor
    layout never enters the control: on 2026-09-28, 0.10 m vs 0.30 m
    arms differed by +1.6 mean fitness against PPO seed noise of
    8.6-20.0, so this mode cannot rank morphologies. Kept for comparison.
* Reward: Isaac Lab ``QuadcopterEnv``'s, per step
  ``(15·(1 − tanh(d/0.8)) − 0.05·|v|² − 0.01·|ω|²) × step_dt``. The
  distance term is bounded and smooth at d = 0 (numerically stable for
  all distances, per the project lead's design note) and non-negative,
  so surviving longer pays. Superseded 2026-09-28: the reward used to
  be ``-distance × step_dt``, which made crashing early pay (a
  free-falling drone scored -0.59 vs -6.09 for one holding hover
  thrust).

Note: this module imports ``isaaclab.*`` at the top of the file; it
should only be imported after ``AppLauncher`` has launched Isaac Sim.
The tutorial's ``train.py`` does this dispatch correctly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from ariel.body_phenotypes.drone.backends import blueprint_to_urdf
from ariel.body_phenotypes.drone.blueprint import DroneBlueprint


# ---------- Blueprint → USD helper ---------------------------------------------

def make_blueprint_usd(
    blueprint: DroneBlueprint,
    *,
    output_dir: str | Path,
    robot_name: str = "drone",
) -> str:
    """Convert a ``DroneBlueprint`` to USD via the URDF intermediate.

    Requires Isaac Sim to already be running (so ``UrdfConverter`` can
    import). The tutorial's ``train.py`` launches the app before
    importing this module.

    Returns the absolute path to the produced ``.usd`` file.
    """
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: PLC0415

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    urdf_path = output_dir / f"{robot_name}.urdf"
    blueprint_to_urdf(blueprint, str(urdf_path), robot_name=robot_name)

    cfg = UrdfConverterCfg(
        asset_path=str(urdf_path),
        usd_dir=str(output_dir),
        usd_file_name=f"{robot_name}.usd",
        force_usd_conversion=True,
        merge_fixed_joints=False,
        fix_base=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="none",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0, damping=0.0,
            ),
        ),
    )
    return UrdfConverter(cfg).usd_path


# ---------- env config ----------------------------------------------------------

@configclass
class IsaacLabBlueprintHoverEnvCfg(DirectRLEnvCfg):
    """Config for the Blueprint-driven hover-to-goal task.

    Use :meth:`from_blueprint` to build a fully-populated config from a
    ``DroneBlueprint`` (generates the USD as a side effect).
    """

    # env
    episode_length_s: float = 5.0
    decimation: int = 2
    action_space: int = 4   # "rotor": number of motors (set by from_blueprint); "wrench": 4
    observation_space: int = 12
    state_space: int = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 100.0, render_interval=decimation)

    # ground plane
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64, env_spacing=3.0, replicate_physics=True,
    )

    # robot — usd_path filled in by `from_blueprint()`; left empty in default
    # so constructing the bare cfg without a blueprint raises an obvious
    # error from UsdFileCfg rather than silently spawning nothing.
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
            copy_from_source=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            # Blueprint-generated drones have all-fixed joints, so the
            # articulation has zero movable joints. Override the default
            # `{".*": 0.0}` joint_pos/vel regexes that otherwise raise
            # "Not all regular expressions are matched!" against an empty
            # joint list.
            joint_pos={},
            joint_vel={},
        ),
        # Rigid drone: all blueprint joints are fixed, so no actuated joints.
        actuators={},
    )

    # actuation
    #   "rotor"  — one action per motor. Each motor link gets its thrust along
    #              its local +Z and its reaction torque about that axis, using
    #              the NumPy backend's rotor model (see IsaacLabBlueprintHoverEnv).
    #              Rotor layout, propeller size and mass all shape what the
    #              policy can do.
    #   "mixer"  — policy outputs collective thrust + 3 body torques; a mixer
    #              turns them into per-rotor thrusts with the blueprint's
    #              allocation matrix, clipped to each rotor's physical range,
    #              then the "rotor" physics applies them. Morphology enters
    #              through allocation, saturation, mass and inertia.
    #   "wrench" — Isaac Lab QuadcopterEnv's abstraction: total thrust + 3 body
    #              moments at base_link, thrust scaled to the drone's weight.
    #              Rotor layout never enters the control.
    action_mode: str = "mixer"

    # "mixer" mode: torque (N·m) commanded by an action of ±1. Fixed physical
    # values, the same for every morphology, so a layout's limited authority
    # shows up as saturation. Chosen 2026-09-28 against the hover authority of
    # the tutorial's "+" quad, 2·L·(T_hover − T_min) ≈ 0.23 / 0.42 / 0.70 N·m
    # for 0.10 / 0.18 / 0.30 m arms (roll/pitch), and ≈ 0.053 N·m yaw from
    # rotor drag for any arm length.
    mixer_roll_pitch_torque: float = 0.3
    mixer_yaw_torque: float = 0.05

    # "wrench" mode only
    thrust_to_weight: float = 2.0
    moment_scale: float = 0.05

    # "rotor" mode: one entry per motor, filled in by from_blueprint() in the
    # order the actions use (sorted blueprint MotorNode ids).
    motor_names: list[str] = []       # URDF link names, "motor_<id>"
    motor_propsizes: list[int] = []   # inches; selects constants in propeller_data
    motor_spins: list[str] = []       # "cw" | "ccw"

    # reward shaping — same terms and scales as Isaac Lab's QuadcopterEnv
    lin_vel_reward_scale: float = -0.05
    ang_vel_reward_scale: float = -0.01
    distance_to_goal_reward_scale: float = 15.0

    # termination
    z_lower: float = 0.1
    z_upper: float = 3.0

    @classmethod
    def from_blueprint(
        cls,
        blueprint: DroneBlueprint,
        *,
        num_envs: int = 64,
        usd_output_dir: Optional[str | Path] = None,
        **overrides,
    ) -> "IsaacLabBlueprintHoverEnvCfg":
        """Build a config from a ``DroneBlueprint`` by generating a USD asset.

        Generates the USD into ``usd_output_dir`` (a fresh temp dir if
        not provided), then slots the path into ``robot.spawn.usd_path``.
        """
        import tempfile  # noqa: PLC0415
        if usd_output_dir is None:
            usd_output_dir = tempfile.mkdtemp(prefix="ariel_blueprint_usd_")
        usd_path = make_blueprint_usd(blueprint, output_dir=usd_output_dir)
        cfg = cls(**overrides)
        cfg.scene.num_envs = num_envs
        cfg.robot.spawn.usd_path = usd_path

        # blueprint_to_urdf names each motor link "motor_<node id>".
        motor_ids = sorted(blueprint.nodes_of_type("Motor"))
        cfg.motor_names = [f"motor_{m}" for m in motor_ids]
        cfg.motor_propsizes = [int(blueprint.payload(m).propsize) for m in motor_ids]
        cfg.motor_spins = [blueprint.payload(m).spin for m in motor_ids]
        cfg.action_space = len(motor_ids) if cfg.action_mode == "rotor" else 4
        # NOTE (open, 2026-09-28): mass and inertia in this backend come from
        # the blueprint (URDF: 0.4 kg core plate + arms + motors), while the
        # NumPy backend's DroneConfiguration builds its own mass model
        # (controller + battery + propellers + beams). The same blueprint
        # therefore has a different mass in the two backends, and fitness is
        # not comparable across them until one mass model is shared.
        return cfg


# ---------- env class -----------------------------------------------------------

class IsaacLabBlueprintHoverEnv(DirectRLEnv):
    """Direct RL env: a drone (built from a ``DroneBlueprint`` USD)
    learns to hover at a randomly-sampled goal in body-frame coords.

    Modeled on Isaac Lab's reference ``QuadcopterEnv``.
    """

    cfg: IsaacLabBlueprintHoverEnvCfg

    def __init__(
        self,
        cfg: IsaacLabBlueprintHoverEnvCfg,
        render_mode: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Resolve the root body index. `blueprint_to_urdf` emits a "base_link"
        # link for the CorePlate; that's the body we apply the wrench to.
        body_ids, _ = self._robot.find_bodies("base_link")
        if not body_ids:
            raise RuntimeError(
                "Could not find 'base_link' on the blueprint-generated articulation. "
                "Was the URDF emitted by blueprint_to_urdf?"
            )
        self._body_id = body_ids

        # Mass / weight for thrust scaling.
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = float((self._robot_mass * self._gravity_magnitude).item())

        if self.cfg.action_mode in ("rotor", "mixer"):
            self._init_rotors()
            if self.cfg.action_mode == "mixer":
                self._init_mixer()
        elif self.cfg.action_mode != "wrench":
            raise ValueError(
                f"action_mode must be 'rotor', 'mixer' or 'wrench', got {self.cfg.action_mode!r}"
            )

    def _init_rotors(self) -> None:
        """Per-motor constants and state for ``action_mode="rotor"``.

        Same rotor model as the NumPy backend's DroneSimulator, per motor:
        throttle U ∈ [0, 1] → target speed
        ω_c = (ω_max − ω_min)·sqrt(k·U² + (1 − k)·U) + ω_min, first-order lag
        dω/dt = (ω_c − ω)/τ, thrust k_f·ω² along the motor's +Z and reaction
        torque k_m·ω² about it (−1 for ccw, +1 for cw — DroneConfiguration's
        sign convention). Constants come from propeller_data by propsize.

        Action → throttle is piecewise linear around this drone's hover
        throttle U_h: a ∈ [-1, 0] → U ∈ [0, U_h], a ∈ [0, 1] → U ∈ [U_h, 1].
        So a = 0 is hover for every morphology while the full physical range
        stays reachable. (With the plain U = (a + 1)/2, a = 0 commanded
        2.61x the weight of a 0.5 kg quad, so a fresh PPO policy launched
        itself into the ceiling and learning to hover was a seed lottery.)
        """
        from ariel.simulation.drone.propeller_data import get_propeller_specs  # noqa: PLC0415

        names = list(self.cfg.motor_names)
        if not names:
            raise RuntimeError("action_mode='rotor' needs cfg.motor_names; build the cfg with from_blueprint().")
        motor_ids, found = self._robot.find_bodies(names, preserve_order=True)
        if list(found) != names:
            raise RuntimeError(f"Motor links {names} not all found on the articulation (found {found}).")
        self._motor_ids = motor_ids
        n = len(names)

        specs = [get_propeller_specs(p) for p in self.cfg.motor_propsizes]

        def row(values: list[float]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.float32, device=self.device).unsqueeze(0)

        self._k_f = row([s["constants"][0] for s in specs])
        self._k_m = row([s["constants"][1] for s in specs])
        self._w_min = row([s["w_min"] for s in specs])
        self._w_max = row([s["wmax"] for s in specs])
        self._k_shape = row([s["k"] for s in specs])
        # Exact discretisation of the first-order lag over one physics step.
        self._lag_alpha = 1.0 - torch.exp(-self.physics_dt / row([s["tau"] for s in specs]))
        self._spin_sign = row([-1.0 if s == "ccw" else 1.0 for s in self.cfg.motor_spins])

        # Motor speed at which equal-sharing rotors hold the drone's weight.
        # Episodes start mid-air, so motors are reset to this speed.
        self._w_hover = torch.sqrt(self._robot_weight / self._k_f.sum()).clamp(
            self._w_min.max(), self._w_max.min()
        )
        # Hover throttle per motor: invert the speed map at ω_hover, i.e.
        # solve k·U² + (1 − k)·U = f² with f = (ω_hover − ω_min)/(ω_max − ω_min).
        f = (self._w_hover - self._w_min) / (self._w_max - self._w_min)
        k = self._k_shape
        self._u_hover = (-(1.0 - k) + torch.sqrt((1.0 - k) ** 2 + 4.0 * k * f**2)) / (2.0 * k)

        self._motor_speed = torch.full((self.num_envs, n), float(self._w_hover), device=self.device)
        self._motor_speed_cmd = self._motor_speed.clone()
        self._rotor_forces = torch.zeros(self.num_envs, n, 3, device=self.device)
        self._rotor_torques = torch.zeros(self.num_envs, n, 3, device=self.device)

    def _init_mixer(self) -> None:
        """Allocation matrix and its pseudo-inverse for ``action_mode="mixer"``.

        Rows of the 4×n matrix A map rotor thrusts to [F_z, τ_x, τ_y, τ_z] in
        the body (base_link) frame about the drone's centre of mass: column i
        is [a_z, r_i × a_i + s_i·(k_m/k_f)·a_i] with a_i the rotor's thrust
        axis, r_i its position relative to the CoM and s_i its spin sign.
        Geometry and masses are read from the spawned articulation (env 0,
        all envs are identical), so the mixer matches what physics simulates.
        """
        d = self._robot.data
        R_root = matrix_from_quat(d.root_link_quat_w[0])                    # body → world
        masses = self._robot.root_physx_view.get_masses()[0].to(self.device)  # (bodies,)
        com_w = (masses[:, None] * d.body_com_pos_w[0]).sum(0) / masses.sum()
        com_b = R_root.T @ (com_w - d.root_link_pos_w[0])
        pos_b = (R_root.T @ (d.body_link_pos_w[0, self._motor_ids] - d.root_link_pos_w[0]).T).T
        axis_b = (R_root.T @ matrix_from_quat(d.body_link_quat_w[0, self._motor_ids])[..., :, 2].T).T
        drag_arm = (self._k_m / self._k_f)[0]                                # (n,) metres
        moments = torch.cross(pos_b - com_b, axis_b, dim=-1) + (self._spin_sign[0] * drag_arm)[:, None] * axis_b
        self._alloc = torch.cat([axis_b[:, 2:3], moments], dim=1).T          # (4, n)
        self._alloc_pinv = torch.linalg.pinv(self._alloc)                    # (n, 4)
        self._thrust_min = self._k_f * self._w_min**2                        # (1, n)
        self._thrust_max = self._k_f * self._w_max**2
        # Collective thrust available along body +Z at full throttle.
        self._collective_max = float((self._thrust_max[0] * axis_b[:, 2]).sum())
        self._torque_scale = torch.tensor(
            [self.cfg.mixer_roll_pitch_torque, self.cfg.mixer_roll_pitch_torque, self.cfg.mixer_yaw_torque],
            device=self.device,
        )

    # -- scene -----------------------------------------------------------

    def _setup_scene(self) -> None:
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # -- per-step dynamics -----------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone().clamp(-1.0, 1.0)
        if self.cfg.action_mode == "rotor":
            # action ∈ [-1, 1] → throttle U ∈ [0, 1], with a = 0 at hover
            # → target motor speed (rad/s).
            a = self._actions
            u = torch.where(a <= 0.0, self._u_hover * (1.0 + a), self._u_hover + (1.0 - self._u_hover) * a)
            self._motor_speed_cmd = (self._w_max - self._w_min) * torch.sqrt(
                self._k_shape * u**2 + (1.0 - self._k_shape) * u
            ) + self._w_min
            return
        if self.cfg.action_mode == "mixer":
            # a[0] → collective thrust, piecewise linear with a = 0 at hover:
            # [-1, 0] → [0, weight], [0, 1] → [weight, all rotors at full thrust].
            a0 = self._actions[:, 0]
            w = self._robot_weight
            collective = torch.where(a0 <= 0.0, w * (1.0 + a0), w + (self._collective_max - w) * a0)
            torque = self._actions[:, 1:] * self._torque_scale
            wrench = torch.cat([collective[:, None], torque], dim=1)          # (N, 4)
            thrust = (wrench @ self._alloc_pinv.T).clamp(self._thrust_min, self._thrust_max)
            self._motor_speed_cmd = torch.sqrt(thrust / self._k_f)
            return
        # action[0] ∈ [-1, 1] → thrust along body +Z in [0, thrust_to_weight * weight].
        self._thrust[:, 0, 2] = (
            self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        )
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self) -> None:
        if self.cfg.action_mode in ("rotor", "mixer"):
            # Called once per physics step: advance the motor lag, then apply
            # each rotor's thrust and reaction torque in its motor link's frame.
            self._motor_speed += (self._motor_speed_cmd - self._motor_speed) * self._lag_alpha
            w2 = self._motor_speed**2
            self._rotor_forces[..., 2] = self._k_f * w2
            self._rotor_torques[..., 2] = self._spin_sign * self._k_m * w2
            self._robot.permanent_wrench_composer.set_forces_and_torques(
                body_ids=self._motor_ids,
                forces=self._rotor_forces,
                torques=self._rotor_torques,
            )
            return
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id,
            forces=self._thrust,
            torques=self._moment,
        )

    # -- obs / reward / done ---------------------------------------------

    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._desired_pos_w,
        )
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # Isaac Lab QuadcopterEnv's reward. The distance term
        # 1 - tanh(d / 0.8) is bounded in (0, 1] and smooth at d = 0, and it
        # is non-negative, so every step survived adds reward. (A pure
        # -distance reward made early termination pay: a free-falling drone
        # outscored one holding hover thrust about 10x.)
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(
            self._desired_pos_w - self._robot.data.root_pos_w, dim=1
        )
        distance_to_goal_mapped = 1.0 - torch.tanh(distance_to_goal / 0.8)
        return (
            lin_vel * self.cfg.lin_vel_reward_scale
            + ang_vel * self.cfg.ang_vel_reward_scale
            + distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale
        ) * self.step_dt

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        z = self._robot.data.root_pos_w[:, 2]
        died = (z < self.cfg.z_lower) | (z > self.cfg.z_upper)
        return died, time_out

    # -- reset -----------------------------------------------------------

    def _reset_idx(self, env_ids: Optional[torch.Tensor]) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self._actions[env_ids] = 0.0
        if self.cfg.action_mode in ("rotor", "mixer"):
            self._motor_speed[env_ids] = self._w_hover
            self._motor_speed_cmd[env_ids] = self._w_hover

        # Random goal in [-1.5, 1.5]² XY, [0.6, 1.5] Z, around the env origin.
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(
            self._desired_pos_w[env_ids, :2]
        ).uniform_(-1.5, 1.5)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(
            self._desired_pos_w[env_ids, 2]
        ).uniform_(0.6, 1.5)

        # Reset robot pose/velocity from defaults.
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)


# ---------- rl_games PPO config helper -----------------------------------------

def make_rl_games_agent_cfg(
    *,
    max_epochs: int = 200,
    horizon_length: int = 24,
    minibatch_size: int = 24 * 64,  # horizon_length × default num_envs
    device: str = "cuda:0",
    experiment_name: str = "ariel_blueprint_hover",
    seed: int = 42,
) -> dict:
    """Return an rl_games agent-config dict matched to Isaac Lab's
    reference quadcopter PPO config (``rl_games_ppo_cfg.yaml``).

    rl_games is configured by nested dicts (the standard YAML pattern),
    so this helper returns a plain ``dict`` rather than a dataclass.
    Mutate the returned dict if you need to override anything.

    We use rl_games (rather than rsl_rl) because Isaac Lab's
    ``isaaclab_rl.rl_games`` adapter is stable on the actual installed
    rl_games version, while ``isaaclab_rl.rsl_rl`` was caught between
    rsl-rl-lib 3.x and 5.x API drift when we tried it.
    """
    return {
        "params": {
            # rl_games' Runner.load() reseeds torch / numpy from this value,
            # overriding the env's cfg.seed — so this is the effective seed.
            "seed": seed,
            "env": {
                "clip_actions": 1.0,
            },
            "algo": {"name": "a2c_continuous"},
            "model": {"name": "continuous_a2c_logstd"},
            "network": {
                "name": "actor_critic",
                "separate": False,
                "space": {
                    "continuous": {
                        "mu_activation": "None",
                        "sigma_activation": "None",
                        "mu_init": {"name": "default"},
                        "sigma_init": {"name": "const_initializer", "val": 0},
                        "fixed_sigma": True,
                    },
                },
                "mlp": {
                    "units": [64, 64],
                    "activation": "elu",
                    "d2rl": False,
                    "initializer": {"name": "default"},
                    "regularizer": {"name": "None"},
                },
            },
            "load_checkpoint": False,
            "load_path": "",
            "config": {
                "name": experiment_name,
                "env_name": "rlgpu",
                "device": device,
                "device_name": device,
                "multi_gpu": False,
                "ppo": True,
                "mixed_precision": False,
                "normalize_input": True,
                "normalize_value": True,
                "value_bootstrap": True,
                "num_actors": -1,  # filled in by the train script
                "reward_shaper": {"scale_value": 0.01},
                "normalize_advantage": True,
                "gamma": 0.99,
                "tau": 0.95,
                "learning_rate": 5e-4,
                "lr_schedule": "adaptive",
                "schedule_type": "legacy",
                "kl_threshold": 0.016,
                "score_to_win": 20000,
                "max_epochs": max_epochs,
                "save_best_after": 100,
                "save_frequency": 25,
                "grad_norm": 1.0,
                "entropy_coef": 0.0,
                "truncate_grads": True,
                "e_clip": 0.2,
                "horizon_length": horizon_length,
                "minibatch_size": minibatch_size,
                "mini_epochs": 5,
                "critic_coef": 2,
                "clip_value": True,
                "seq_length": 4,
                "bounds_loss_coef": 0.0001,
            },
        },
    }


__all__ = [
    "make_blueprint_usd",
    "IsaacLabBlueprintHoverEnvCfg",
    "IsaacLabBlueprintHoverEnv",
    "make_rl_games_agent_cfg",
]
