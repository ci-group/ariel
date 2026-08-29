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
* Action space (4D): total thrust + 3 moment components. Morphology-
  independent. Per-rotor thrust modeling is deferred to a follow-up;
  the wrench-at-root abstraction works for any roughly-balanced
  drone.
* Reward: ``-distance_to_goal`` per step (cumulates to
  ``-Σ distance × step_dt`` over the episode). Numerically stable
  for all distance values including very near zero, per the project
  lead's design note.

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
from isaaclab.utils.math import subtract_frame_transforms

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
    action_space: int = 4   # thrust + 3 moments
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

    # control allocation
    thrust_to_weight: float = 2.0
    moment_scale: float = 0.05

    # reward shaping
    distance_reward_scale: float = 1.0

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
        # action[0] ∈ [-1, 1] → thrust along body +Z in [0, thrust_to_weight * weight].
        self._thrust[:, 0, 2] = (
            self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        )
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self) -> None:
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
        # Negative distance, summed across the episode → minimizes total
        # excursion from the goal. Numerically stable for all distance
        # values; never blows up near zero.
        distance_to_goal = torch.linalg.norm(
            self._desired_pos_w - self._robot.data.root_pos_w, dim=1
        )
        return -distance_to_goal * self.cfg.distance_reward_scale * self.step_dt

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
            "seed": 42,
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
