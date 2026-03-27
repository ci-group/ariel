import pathlib
import re
import time
from itertools import starmap
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
from omegaconf import OmegaConf
from ompl import base as ob
from ompl import geometric as og

from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.scenes.table import (
    table_terrain,
)

# Import required modules for building the Mujoco model
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.build_file import (
    build_mjcf,
)
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.mj_default_sim_setup import (
    mujoco_setup_sim,
)

# Import modular components
from ariel.body_phenotypes.Lynx_Arm.lynx.robots.lynx_manipulator.constructor import (
    construct,
)

_TUBE_SEG_RE = re.compile(r"^(lynx_)?(?P<tube>tube\d+)_seg_\d+$")


def _geom_name(model, geom_id: int) -> str:
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    return n or ""


def _is_same_joint_internal(n1: str, n2: str) -> bool:
    """Filter: collisions between geoms belonging to the SAME joint module."""
    j1 = re.findall(r"joint\d+", n1)
    j2 = re.findall(r"joint\d+", n2)
    if not j1 or not j2:
        return False
    return any(a == b for a in j1 for b in j2)


def _tube_id_from_geom(name: str) -> str | None:
    """Return tube id ("tube1"/"tube2") if this geom is a tube segment."""
    m = _TUBE_SEG_RE.match(name)
    if not m:
        return None
    return m.group("tube")


def _is_same_tube_segments(n1: str, n2: str) -> bool:
    """Filter: collisions between segments of the SAME tube."""
    t1 = _tube_id_from_geom(n1)
    t2 = _tube_id_from_geom(n2)
    return (t1 is not None) and (t1 == t2)


def _is_robot_geom(name: str) -> bool:
    return name.startswith("lynx")


def trapezoidal_velocity_profile(q_start, q_end, v_max, a_max, t):
    """
    Generates a trapezoidal velocity profile for a single joint.
    Returns position, velocity, and acceleration at time t.
    """
    delta_q = q_end - q_start

    # Calculate time to accelerate to v_max (Ta)
    Ta = v_max / a_max

    # Calculate distance covered during acceleration (Sa)
    Sa = 0.5 * a_max * Ta**2

    # Check if the profile is triangular (doesn't reach v_max)
    if Sa * 2 > abs(delta_q):
        # Triangular profile
        Ta = np.sqrt(abs(delta_q) / a_max)
        v_max = a_max * Ta

        if t < Ta:
            # Acceleration phase
            a = a_max * np.sign(delta_q)
            v = a * t
            q = q_start + 0.5 * a * t**2
        elif t < 2 * Ta:
            # Deceleration phase
            a = -a_max * np.sign(delta_q)
            v = v_max * np.sign(delta_q) + a * (t - Ta)
            # Position at Ta
            q_at_Ta = q_start + 0.5 * a_max * np.sign(delta_q) * Ta**2
            # Position during deceleration
            q = q_at_Ta + v_max * \
                np.sign(delta_q) * (t - Ta) + 0.5 * a * (t - Ta)**2
        else:
            # End of motion
            a = 0.0
            v = 0.0
            q = q_end
    else:
        # Trapezoidal profile
        # Time at constant velocity (Tv)
        Tv = (abs(delta_q) - 2 * Sa) / v_max

        if t < Ta:
            # Acceleration phase
            a = a_max * np.sign(delta_q)
            v = a * t
            q = q_start + 0.5 * a * t**2
        elif t < Ta + Tv:
            # Constant velocity phase
            a = 0.0
            v = v_max * np.sign(delta_q)
            q = q_start + Sa * np.sign(delta_q) + v * (t - Ta)
        elif t < 2 * Ta + Tv:
            # Deceleration phase (trapezoidal profile)
            a = -a_max * np.sign(delta_q)
            v = v_max * np.sign(delta_q) + a * (t - Ta - Tv)
            # Position at Ta
            q_at_Ta = q_start + 0.5 * a_max * np.sign(delta_q) * Ta**2
            # Position at Ta + Tv
            q_at_Ta_Tv = q_at_Ta + v_max * np.sign(delta_q) * Tv
            # Position during deceleration
            q = q_at_Ta_Tv + v_max * \
                np.sign(delta_q) * (t - (Ta + Tv)) + \
                0.5 * a * (t - (Ta + Tv))**2
        else:
            # End of motion
            a = 0.0
            v = 0.0
            q = q_end

    return q, v, a


# ============================================================
# Cartesian Goal Region
# ============================================================

class CartesianGoalRegion(ob.GoalRegion):
    """Goal region defined in Cartesian end-effector space."""

    def __init__(self, si, planner, goal_ee_pos, threshold=0.01) -> None:
        super().__init__(si)
        self.planner = planner
        self.goal_ee_pos = np.array(goal_ee_pos)
        self.setThreshold(threshold)

    def distanceGoal(self, state) -> float:
        """Compute Euclidean distance between end-effector and goal."""
        for i, qpos_idx in enumerate(self.planner.joint_qpos_indices):
            self.planner.data.qpos[qpos_idx] = state[i]

        mujoco.mj_forward(self.planner.model, self.planner.data)

        ee_pos = self.planner.data.site_xpos[self.planner.ee_site_id].copy()
        # Adjust for body position if necessary
        ee_pos_rel = ee_pos - self.planner.body_pos
        return float(np.linalg.norm(ee_pos_rel - self.goal_ee_pos))


# ============================================================
# Lynx Planner
# ============================================================

class LynxPlanner:
    """
    OMPL-based motion planner for Lynx manipulator using MuJoCo
    forward kinematics for Cartesian goal evaluation and collision checking.
    """

    def __init__(
        self,
        cfg: Any,
        body_pos: list | None = None,
        body_ori: list | None = None,
        xml_string: str | None = None,
    ) -> None:
        if body_ori is None:
            body_ori: list = [0, 0, 0, 1]
        if body_pos is None:
            body_pos: list = [0, 0, 0.61801]
        self.cfg = cfg
        self.body_pos = np.array(body_pos)
        self.body_ori = np.array(body_ori)

        # Use the new constructor
        body = construct(
            robot_description_dict=cfg.MorphConfig.robot_description_dict,
        )

        if xml_string:
            self.xml_string = xml_string
        else:
            self.xml_string = build_mjcf(
                bodies=[body],
                body_poss=[body_pos],
                body_oris=[body_ori],
                terrain_builder=table_terrain,
                sim_setup=mujoco_setup_sim,
                ts=cfg.TrainConfig.time_step,
            )

        # Load assets for STL meshes
        # assets = {}
        # abs_stl_path = pathlib.Path(cfg.MorphConfig.robot_description_dict.clamp_stl).resolve()
        # mesh_name = cfg.MorphConfig.robot_description_dict.clamp_stl.split("/")[-1].replace(".stl", "")
        # with pathlib.Path(abs_stl_path).open("rb") as f:
        #     stl_data = f.read()
        #     match = re.search(rf'file="({mesh_name}-[^"]+\.stl)"', self.xml_string)
        #     if match:
        #         assets[match.group(1)] = stl_data
        #     else:
        #         assets[cfg.MorphConfig.robot_description_dict.clamp_stl] = stl_data

        self.model = mujoco.MjModel.from_xml_string(self.xml_string)#, assets=assets)
        self.data = mujoco.MjData(self.model)

        # Resolve end-effector site
        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector",
        )
        if self.ee_site_id == -1:
            msg = "Could not find 'end_effector' site."
            raise ValueError(msg)

        self.joint_qpos_indices = self._resolve_joint_indices()

        # Build OMPL state space
        self.space = ob.RealVectorStateSpace(6)
        bounds = ob.RealVectorBounds(6)

        for i, (lo, hi) in enumerate(cfg.RLConfig.JOINT_LIMITS):
            bounds.setLow(i, np.radians(lo))
            bounds.setHigh(i, np.radians(hi))

        self.space.setBounds(bounds)

        self.ss = og.SimpleSetup(self.space)
        self.ss.setStateValidityChecker(
            ob.StateValidityCheckerFn(self._is_valid),
        )

    def _resolve_joint_indices(self) -> list[int]:
        """Resolve MuJoCo qpos indices for joints 1–6."""
        indices = []
        for i in range(1, 7):
            joint_name = f"joint{i}_joint"
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name,
            )
            if joint_id == -1:
                msg = f"Joint '{joint_name}' not found."
                raise ValueError(msg)
            indices.append(self.model.jnt_qposadr[joint_id])
        return indices

    def _is_valid(self, state) -> bool:
        """State validity checker with collision detection."""
        # Set joint positions
        for i, qpos_idx in enumerate(self.joint_qpos_indices):
            self.data.qpos[qpos_idx] = state[i]

        # Forward kinematics and collision detection
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)

        # Check for collisions
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            n1 = _geom_name(self.model, c.geom1)
            n2 = _geom_name(self.model, c.geom2)

            if not n1 or not n2:
                continue

            is_env1 = (n1.startswith("table_") or n1 == "floor")
            is_env2 = (n2.startswith("table_") or n2 == "floor")
            is_robot1 = _is_robot_geom(n1)
            is_robot2 = _is_robot_geom(n2)

            # Robot-Environment collision
            if (is_robot1 and is_env2) or (is_robot2 and is_env1):
                return False

            # Robot-Robot collision
            if is_robot1 and is_robot2:
                if _is_same_joint_internal(n1, n2):
                    continue
                if _is_same_tube_segments(n1, n2):
                    continue
                return False

        return True

    def _set_config(self, config: np.ndarray) -> None:
        for i, qpos_idx in enumerate(self.joint_qpos_indices):
            self.data.qpos[qpos_idx] = config[i]
        mujoco.mj_forward(self.model, self.data)

    def get_ee_pos(self, config=None) -> np.ndarray:
        if config is not None:
            self._set_config(config)
        return self.data.site_xpos[self.ee_site_id].copy()

    def plan(self, start_config, goal_ee_pos, timeout=5.0):
        self.ss.clear()
        start = ob.State(self.space)
        for i in range(6):
            start()[i] = float(start_config[i])

        goal_region = CartesianGoalRegion(
            self.ss.getSpaceInformation(),
            self,
            goal_ee_pos,
            threshold=0.01,
        )

        self.ss.setStartState(start)
        self.ss.setGoal(goal_region)

        planner = og.RRTstar(self.ss.getSpaceInformation())
        self.ss.setPlanner(planner)

        solved = self.ss.solve(timeout)
        if not solved:
            return None

        self.ss.simplifySolution()
        return self._extract_path()

    def _extract_path(self):
        path = self.ss.getSolutionPath()
        configs = []
        for i in range(path.getStateCount()):
            state = path.getState(i)
            configs.append(np.array([state[j] for j in range(6)]))
        return configs

    def run(self, goal, start_config=None, visualize=False) -> tuple[list[float], float]:
        if start_config is None:
            start_config = np.zeros(6)  # Default home

        path = self.plan(start_config, goal, timeout=5.0)
        if path is None:
            return None, np.inf

        final_ee = self.get_ee_pos(path[-1])
        final_ee_rel = final_ee - self.body_pos
        final_error = np.linalg.norm(np.array(goal) - final_ee_rel)

        if visualize:
            self.visualize_path_smooth(path, goal)

        return path, float(final_error)

    def run_multiple(self, goal, start_arr=None, num_points=10):
        if start_arr is None:
            start_arr = np.array([
                list(starmap(np.random.uniform, self.cfg.RLConfig.JOINT_LIMITS))
                for _ in range(num_points)
            ])
            start_arr = np.deg2rad(start_arr)

        errors = []
        paths = []

        for start in start_arr:
            path, error = self.run(goal, start_config=start, visualize=False)
            if path is not None:
                errors.append(error)
                paths.append(path)

        if len(errors) == 0:
            return [], np.inf

        mean_error = float(np.mean(errors))
        return paths, mean_error

    def visualize_path_smooth(self, path, goal, dt=None) -> None:
        if path is None or len(path) < 2:
            return

        if dt is None:
            dt = self.model.opt.timestep

        start_ee = self.get_ee_pos(path[0])
        final_ee = self.get_ee_pos(path[-1])
        goal_world = np.array(goal) + self.body_pos

        v_max = np.radians(20.0)
        a_max = np.radians(100.0)

        with mujoco.viewer.launch(self.model, self.data) as viewer:
            def add_sphere(scene, pos, rgba, size=0.02) -> None:
                if scene.ngeom >= scene.maxgeom:
                    return
                g = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    g,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([size, 0, 0]),
                    np.array(pos),
                    np.eye(3).flatten(),
                    np.array(rgba),
                )
                scene.ngeom += 1

            for seg_idx in range(len(path) - 1):
                q_start = path[seg_idx]
                q_end = path[seg_idx + 1]

                duration = 0.0
                for j in range(6):
                    delta = abs(q_end[j] - q_start[j])
                    Ta = v_max / a_max
                    Sa = 0.5 * a_max * Ta**2
                    if 2 * Sa > delta:
                        Ta = np.sqrt(delta / a_max)
                        seg_dur = 2 * Ta
                    else:
                        Tv = (delta - 2 * Sa) / v_max
                        seg_dur = 2 * Ta + Tv
                    duration = max(duration, seg_dur)

                t = 0.0
                while t <= duration:
                    config = np.zeros(6)
                    for j in range(6):
                        config[j], _, _ = trapezoidal_velocity_profile(
                            q_start[j], q_end[j], v_max, a_max, t,
                        )
                    self._set_config(config)

                    add_sphere(viewer.user_scn, start_ee, [1, 0, 0, 1])
                    add_sphere(viewer.user_scn, final_ee, [0, 0, 1, 1])
                    add_sphere(viewer.user_scn, goal_world, [0, 1, 0, 1])

                    viewer.sync()
                    time.sleep(dt)
                    t += dt

            self._set_config(path[-1])
            while viewer.is_running():
                add_sphere(viewer.user_scn, start_ee, [1, 0, 0, 1])
                add_sphere(viewer.user_scn, final_ee, [0, 0, 1, 1])
                add_sphere(viewer.user_scn, goal_world, [0, 1, 0, 1])
                viewer.sync()
                time.sleep(0.01)

    def generate_and_execute_path(self, goal_ee_pos, env: "LynxSimEnv", start_config=None, waypoint_threshold=0.05):
        """
        Generates a path to the goal and executes it in the provided environment.
        Moves to the next waypoint in the path once the current one is reached within waypoint_threshold.
        Collects intermediate states, actions, rewards, etc.
        """
        # 1. Reset env with the goal and get initial state
        if start_config is not None:
            obs, info = env.reset(options={
                "joint_position": "custom",
                "custom_joint_pos_deg": np.degrees(start_config),
                "target_position": np.array(goal_ee_pos),
            })
        else:
            obs, info = env.reset(options={
                "joint_position": "home",
                "target_position": np.array(goal_ee_pos),
            })
            current_joint_pos_deg = np.array([pv[0] for pv in env.robot.get_PosVel()])
            start_config = np.radians(current_joint_pos_deg)

        # 2. Plan the path (list of 6D joint vectors)
        path = self.plan(start_config, goal_ee_pos)
        if path is None:
            return None

        # 3. Execute path by following waypoints
        action_range_rad = np.radians(10.0)
        dataset = []
        total_steps = env.max_episode_steps
        executed_steps = 0

        # Skip the first point as it's the start configuration
        waypoints = path[1:]
        current_waypoint_idx = 0

        while executed_steps < total_steps:
            if current_waypoint_idx < len(waypoints):
                target_q = waypoints[current_waypoint_idx]

                # Get current joint positions
                current_joint_pos_deg = np.array([pv[0] for pv in env.robot.get_PosVel()])
                current_q = np.radians(current_joint_pos_deg)

                # Check if we reached the current waypoint
                q_error = np.linalg.norm(target_q - current_q)

                is_last_waypoint = (current_waypoint_idx == len(waypoints) - 1)
                threshold = env.distance_threshold if is_last_waypoint else waypoint_threshold

                # If reached waypoint, move to next
                if q_error < threshold:
                    current_waypoint_idx += 1
                    if current_waypoint_idx < len(waypoints):
                        target_q = waypoints[current_waypoint_idx]

                # Simple P-control or direct delta for action
                action = target_q - current_q
                action = np.clip(action, -action_range_rad, action_range_rad)
            else:
                # Final approach or stay at goal if already reached
                current_ee_pos = info["end_effector_position"]
                dist_to_goal = np.linalg.norm(current_ee_pos - np.array(goal_ee_pos))

                if dist_to_goal > env.distance_threshold:
                    target_q = path[-1]
                    current_joint_pos_deg = np.array([pv[0] for pv in env.robot.get_PosVel()])
                    current_q = np.radians(current_joint_pos_deg)

                    action = target_q - current_q
                    action = np.clip(action, -action_range_rad, action_range_rad)
                else:
                    action = np.zeros(6)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)

            dataset.append({
                "observations": obs,
                "actions": action,
                "rewards": reward,
                "next_observations": next_obs,
                "terminated": terminated,
                "truncated": truncated,
                "info": info,
            })

            obs = next_obs
            executed_steps += 1

            if terminated or truncated:
                break

        return dataset


if __name__ == "__main__":
    cfg = OmegaConf.load("configs/sim.yaml")
    planner = LynxPlanner(cfg)

    goal = [0.3, 0.3, 0.5]  # Relative to body_pos
    # paths, mean_error = planner.run_multiple(goal, num_points=5)
    path, mean_error = planner.run(goal, visualize=True)
