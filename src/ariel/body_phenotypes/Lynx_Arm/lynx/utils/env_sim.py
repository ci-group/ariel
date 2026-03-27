import gymnasium as gym
import numpy as np
import mujoco
import re, os
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple

import time
import yaml
from scipy.spatial.transform import Rotation as R

# Import required modules for building the Mujoco model
from lynx.morphlib.tools.build_file import build_mjcf
from lynx.morphlib.tools.mj_default_sim_setup import mujoco_setup_sim
from lynx.morphlib.scenes.table import table_terrain

# Import modular components
# from lynx.robots.lynx_manipulator.legacy.constructor import construct
from lynx.robots.lynx_manipulator.constructor import construct

from lynx.robots.sim_mujoco_robot import MujocoSimRobot
from lynx.sensors.sim_mujoco_sensor import MujocoSimSensor
from lynx.rl.reach.utils import (
    load_config,
    sample_point_in_sphere, 
    sample_point_in_cylinder, 
    sample_point_in_cube, 
    sample_point_in_table,
    sample_kinematic_reachable_points, 
    quat_mul_np, quat_conjugate_np, axis_angle_from_quat_np
    )


_TUBE_SEG_RE = re.compile(r"^(lynx_)?(?P<tube>tube\d+)_seg_\d+$")

def _geom_name(model, geom_id: int) -> str:
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    return n or ""

def _is_same_joint_internal(n1: str, n2: str) -> bool:
    """
    Filter: collisions between geoms belonging to the SAME joint module.
    You have names like:
    - "lynxg1_joint4", "lynxg2_joint4"
    - maybe also "lynx_joint4_..." depending on your builder
    Rule here: if both names contain the exact joint token "jointK" (K = digits), ignore.
    """
    # find joint tokens like joint1, joint2, ... in each name
    j1 = re.findall(r"joint\d+", n1)
    j2 = re.findall(r"joint\d+", n2)
    if not j1 or not j2:
        return False
    # if any joint token matches, treat as same-joint internal
    return any(a == b for a in j1 for b in j2)

def _tube_id_from_geom(name: str) -> str | None:
    """
    Return tube id ("tube1"/"tube2") if this geom is a tube segment.
    Your naming in BSplineTube: name=f"lynx_{self.name}_seg_{i}" -> "lynx_tube1_seg_0"
    """
    m = _TUBE_SEG_RE.match(name)
    if not m:
        return None
    return m.group("tube")

def _is_same_tube_segments(n1: str, n2: str) -> bool:
    """
    Filter: collisions between segments of the SAME tube (tube1 with tube1, etc).
    """
    t1 = _tube_id_from_geom(n1)
    t2 = _tube_id_from_geom(n2)
    return (t1 is not None) and (t1 == t2)

def _is_robot_geom(name: str) -> bool:
    # your robot geoms are mostly prefixed with "lynx"
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
            # Deceleration phase (triangular profile)
            a = -a_max * np.sign(delta_q)
            v = v_max * np.sign(delta_q) + a * (t - Ta)
            # Position at Ta
            q_at_Ta = q_start + 0.5 * a_max * np.sign(delta_q) * Ta**2
            # Position during deceleration
            q = q_at_Ta + v_max * np.sign(delta_q) * (t - Ta) + 0.5 * a * (t - Ta)**2
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
            q = q_at_Ta_Tv + v_max * np.sign(delta_q) * (t - (Ta + Tv)) + 0.5 * a * (t - (Ta + Tv))**2
        else:
            # End of motion
            a = 0.0
            v = 0.0
            q = q_end
            
    return q, v, a


def trapezoidal_velocity_profile_non_zero_start(q_start, q_end, v_start, v_max, a_max, t):
    """
    Generates a trapezoidal velocity profile for a single joint with a non-zero starting velocity.
    Returns position, velocity, and acceleration at time t.
    """
    delta_q = q_end - q_start
    direction = np.sign(delta_q) if delta_q != 0 else 1.0
    # print(f"[trapezoidal_velocity_profile_non_zero_start] delta_q: {delta_q}, direction: {direction}, v_start: {v_start}")

    # Adjust v_start based on direction of motion
    if delta_q != 0 and np.sign(v_start) != direction:
        v_start = 0.0
    else:
        v_start = v_start

    # Calculate times for acceleration and deceleration phases
    # Time to accelerate from v_start to v_max
    Ta1 = (v_max - abs(v_start)) / a_max if v_max > abs(v_start) else 0.0
    # Distance covered during Ta1
    Sa1 = abs(v_start) * Ta1 + 0.5 * a_max * Ta1**2

    # Time to decelerate from v_max to 0
    Ta2 = v_max / a_max
    # Distance covered during Ta2
    Sa2 = 0.5 * a_max * Ta2**2

    # Total distance required for full acceleration to v_max and deceleration from v_max
    S_accel_decel = Sa1 + Sa2

    # Determine if the profile is triangular or trapezoidal
    if abs(delta_q) < (Sa1 + Sa2): # Triangular profile (doesn't reach v_max)

        v_peak = np.sqrt((2 * a_max * abs(delta_q) + v_start**2) / 2)

        T_accel = (v_peak - abs(v_start)) / a_max
        T_decel = v_peak / a_max
        T_total = T_accel + T_decel

        if t < T_accel:
            # Acceleration phase
            a = a_max * direction
            v = v_start + a * t
            q = q_start + v_start * t + 0.5 * a * t**2
        elif t < T_total:
            # Deceleration phase
            a = -a_max * direction
            v = v_peak * direction + a * (t - T_accel) # v_peak still needs direction
            q_at_T_accel = q_start + v_start * T_accel + 0.5 * a_max * direction * T_accel**2
            q = q_at_T_accel + v_peak * direction * (t - T_accel) + 0.5 * a * (t - T_accel)**2 # v_peak still needs direction
        else:
            # End of motion
            a = 0.0
            v = 0.0
            q = q_end
    else: # Trapezoidal profile (reaches v_max)
        T_const_vel = (abs(delta_q) - S_accel_decel) / v_max
        T_total = Ta1 + T_const_vel + Ta2

        if t < Ta1:
            # Acceleration phase 1 (from v_start to v_max)
            a = a_max * direction
            v = v_start + a * t
            q = q_start + v_start * t + 0.5 * a * t**2
        elif t < Ta1 + T_const_vel:
            # Constant velocity phase
            a = 0.0
            v = v_max * direction
            q_at_Ta1 = q_start + v_start * Ta1 + 0.5 * a_max * direction * Ta1**2
            q = q_at_Ta1 + v_max * direction * (t - Ta1)
        elif t < T_total:
            # Deceleration phase (from v_max to 0)
            a = -a_max * direction
            v = v_max * direction + a * (t - Ta1 - T_const_vel)
            q_at_Ta1 = q_start + v_start * Ta1 + 0.5 * a_max * direction * Ta1**2
            q_at_Ta1_T_const_vel = q_at_Ta1 + v_max * direction * T_const_vel
            q = q_at_Ta1_T_const_vel + v_max * direction * (t - (Ta1 + T_const_vel)) + 0.5 * a * (t - (Ta1 + T_const_vel))**2
        else:
            # End of motion
            a = 0.0
            v = 0.0
            q = q_end

    # print(f"vel: {v}") # Remove debug print

    return q, v, a


class LynxSimEnv(gym.Env):
    """
    A gymnasium environment for a reconfigurable manipulator learning to reach target positions (pos+ori)
    in a Mujoco simulation, integrated with MujocoSimRobot, MujocoSimSensor, and SafetyWatchdogSim.
    TODO: Add domain randomization
    """
    def __init__(
        self,
        cfg: Any, # Hydra config object
        body_pos: list = [0, 0, 0.61801],
        body_ori: list = [0, 0, 0, 1],
        xml_string: Optional[str] = None, # New argument for pre-defined XML
        robot_description_dict: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        # body = lynx_manipulator_bspline(
        #     robot_description_dict=cfg.MorphConfig.robot_description_dict if robot_description_dict is None else robot_description_dict,
        # )
        body = construct(
            robot_description_dict=cfg.MorphConfig.robot_description_dict
        )
        terrain_builder = table_terrain
        sim_setup = mujoco_setup_sim

        self.current_step = 0
        
        self.body = body
        self.body_pos = body_pos
        self.body_ori = body_ori
        self.terrain_builder = terrain_builder
        self.sim_setup = sim_setup
        self.timestep = cfg.TrainConfig.time_step
        self.render_mode = cfg.TrainConfig.render_mode
        self.max_episode_steps = cfg.TrainConfig.max_steps_per_episode
        self.reward_type = cfg.TrainConfig.reward_type
        self.sphere_target_range = cfg.TrainConfig.sphere_target_range
        self.distance_threshold = cfg.TrainConfig.distance_threshold
        self.target_pos_sampling_method = cfg.TrainConfig.target_pos_sampling_method
        print(f"[RL Env] Target position sampling method: {self.target_pos_sampling_method}")
        self.action_mode = cfg.TrainConfig.action_mode
        self.fixed_target_orientation = cfg.TrainConfig.fixed_target_orientation
        self.fixed_target_position = cfg.RLConfig.FIXED_TARGET_POSITION

        self.real_time_speed = cfg.TrainConfig.real_time_speed
        self.collision_detection = cfg.TrainConfig.collision_detection
        self.vis_target_sphere = cfg.TrainConfig.vis_target_sphere
        self.static_viewer = cfg.TrainConfig.static_viewer

        self.collect_imu = cfg.TrainConfig.collect_imu
        if self.collect_imu:
            self.imu_data_partial_raw = np.zeros(8)  # Placeholder for IMU data (accel + gyro)

        self._episode_end_ee_distance = 0.0
        self._current_distance = 0.0
        self._current_action_difference = 0.0
        self._current_orientation_difference = 0.0

        self.cfg = cfg # Store the Hydra config

        self.kinematic_reachable_points = None # Initialize to None
        self.real_lynx = False  # Simulation environment

        # Set reward weights
        reward_weights = {
            'alpha': -0.2,   # Distance
            'beta': 0.1,     # Fine-grained distance
            'gamma': -0.0001,    # Action smoothness penalty weight (discourage jitter)
            'delta': -0.1,   # Orientation
            'epsilon': 0.1,  # Fine-grained orientation
            'zeta': -5.0,   # Self-collision penalty
            'eta': -10.0,     # Table collision penalty
            'success': 10.0,  # Success bonus
        }
        self.reward_weights = reward_weights
        
        # Build MuJoCo XML and create model
        if xml_string:
            self.xml_string = xml_string
            print("[RL Env] Using provided XML string to initialize Mujoco model.")
        else:
            self.xml_string = build_mjcf(
                bodies=[body],
                body_poss=[body_pos],
                body_oris=[body_ori],
                terrain_builder=terrain_builder,
                sim_setup=sim_setup,
                ts=self.timestep
            )
            print("[RL Env] Generated XML string to initialize Mujoco model.")
            
            # Inject target_orientation site into the XML string
            # Find the closing </worldbody> tag and insert the site before it
            worldbody_end_tag = "</worldbody>"
            if worldbody_end_tag in self.xml_string:
                insert_index = self.xml_string.rfind(worldbody_end_tag)
                if insert_index != -1:
                    orientation_site_xml = """
        <site name="target_orientation_x" pos="0 0 0" size="0.005 0.05" rgba="1 0 0 1" type="capsule" quat="0.7071068 0 -0.7071068 0"/> <!-- X-axis (Red) -->
        <site name="target_orientation_y" pos="0 0 0" size="0.005 0.05" rgba="0 1 0 1" type="capsule" quat="0.7071068 0.7071068 0 0"/> <!-- Y-axis (Green) -->
        <site name="target_orientation_z" pos="0 0 0" size="0.005 0.05" rgba="0 0 1 1" type="capsule" quat="0 0 0 1"/> <!-- Z-axis (Blue) -->
    """
                    self.xml_string = (
                        self.xml_string[:insert_index]
                        + orientation_site_xml
                        + self.xml_string[insert_index:]
                    )
                    print("[RL Env] Injected 'target_orientation' site into XML.")
                else:
                    print("[RL Env] Warning: Could not find </worldbody> tag to inject 'target_orientation' site.")
            else:
                print("[RL Env] Warning: Could not find </worldbody> tag in XML to inject 'target_orientation' site.")

        # Load model with assets
        # We use a dictionary to map the filename to its content
        assets = {}
        abs_stl_path = os.path.abspath(cfg.MorphConfig.robot_description_dict.clamp_stl)
        mesh_name = cfg.MorphConfig.robot_description_dict.clamp_stl.split('/')[-1].replace('.stl', '')
        with open(abs_stl_path, 'rb') as f:
            stl_data = f.read()
            
            # Find the actual filename used in the XML (dm_control adds a hash)
            import re
            match = re.search(f'file="({mesh_name}-[^"]+\.stl)"', self.xml_string)
            if match:
                assets[match.group(1)] = stl_data
            else:
                assets[cfg.MorphConfig.robot_description_dict.clamp_stl] = stl_data

        # Load MuJoCo model from XML string
        self.model = mujoco.MjModel.from_xml_string(self.xml_string, assets=assets)
        self.data = mujoco.MjData(self.model)

        # setup the viewer angle:
        self.azimuth = -180        # azimuth angle (rotation around z-axis)
        self.elevation = -10      # angle from horizontal plane
        self.distance = 2.0       # distance from the target
        self.lookat = [0.0, 0.0, 0.8]  # point the camera is looking at
        
        # Adjust camera properties to zoom out
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "isometric_view")
        if camera_id != -1:
            # Set camera parameters to match the requested view:
            # azimuth = -180, elevation = -10, distance = 2.0, lookat = [0, 0, 0.8]
            
            # Convert spherical to cartesian for cam_pos
            dist = self.distance
            elev_rad = np.radians(self.elevation)
            azim_rad = np.radians(self.azimuth)
            lookat = np.array(self.lookat)
            
            cam_pos = lookat + np.array([
                dist * np.cos(elev_rad) * np.cos(azim_rad),
                dist * np.cos(elev_rad) * np.sin(azim_rad),
                -dist * np.sin(elev_rad) # MuJoCo elevation is negative downwards
            ])
            
            # For the quaternion, we want the camera to look at the 'lookat' point.
            def _cam_quat_from_lookat(eye, target, up=(0,0,1)):
                eye    = np.asarray(eye, dtype=float)
                target = np.asarray(target, dtype=float)
                up     = np.asarray(up, dtype=float)
                f = target - eye                     
                f = f / np.linalg.norm(f)
                r = np.cross(f, up); r /= np.linalg.norm(r)
                u = np.cross(r, f)
                R_mat = np.column_stack([r, u, -f])
                m00,m01,m02 = R_mat[0]; m10,m11,m12 = R_mat[1]; m20,m21,m22 = R_mat[2]
                t = m00 + m11 + m22
                if t > 0:
                    S = np.sqrt(t + 1.0) * 2
                    w = 0.25 * S
                    x = (m21 - m12) / S
                    y = (m02 - m20) / S
                    z = (m10 - m01) / S
                elif (m00 > m11) and (m00 > m22):
                    S = np.sqrt(1.0 + m00 - m11 - m22) * 2
                    w = (m21 - m12) / S
                    x = 0.25 * S
                    y = (m01 + m10) / S
                    z = (m02 + m20) / S
                elif m11 > m22:
                    S = np.sqrt(1.0 + m11 - m00 - m22) * 2
                    w = (m02 - m20) / S
                    x = (m01 + m10) / S
                    y = 0.25 * S
                    z = (m12 + m21) / S
                else:
                    S = np.sqrt(1.0 + m22 - m00 - m11) * 2
                    w = (m10 - m01) / S
                    x = (m02 + m20) / S
                    y = (m12 + m21) / S
                    z = 0.25 * S
                return np.array([w, x, y, z])

            self.model.cam_pos[camera_id] = cam_pos
            self.model.cam_quat[camera_id] = _cam_quat_from_lookat(cam_pos, lookat)
            self.model.cam_fovy[camera_id] = 45.0

        # Make the target sphere transparent
        target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target")
        if target_site_id != -1:
            # Set rgba to red with 0.5 alpha (transparency)
            if self.vis_target_sphere:
                self.model.site_rgba[target_site_id] = [1.0, 0.0, 0.0, 0.5]
            else:
                self.model.site_rgba[target_site_id] = [1.0, 0.0, 0.0, 0.0]
        
        # Initialize MujocoSimRobot
        self.robot = MujocoSimRobot(model=self.model, data=self.data, render_mode=self.render_mode)
        self.robot.init_motors() # Initialize motors within the robot controller
        
        # Get info from robot for sensor and watchdog
        _model, _data, end_effector_site_id = self.robot.get_info()
        
        # Initialize MujocoSimSensor
        self.sensor = MujocoSimSensor(model=_model, data=_data, end_effector_site_id=end_effector_site_id)
        self.sensor.start()

        # Initialize viewer if render_mode is human
        if self.render_mode == "human":
            import mujoco.viewer as mjv
            self.viewer = mjv.launch(self.model, self.data)
            # Use the specialized camera defined in the model
            if self.static_viewer:
                camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "specialized_view")
                if camera_id != -1:
                    self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    self.viewer.cam.fixedcamid = camera_id
                else:
                    self.viewer.cam.azimuth = self.azimuth
                    self.viewer.cam.elevation = self.elevation
                    self.viewer.cam.distance = self.distance
                    self.viewer.cam.lookat[:] = np.array(self.lookat)
            else:
                self.viewer.cam.azimuth = self.azimuth
                self.viewer.cam.elevation = self.elevation
                self.viewer.cam.distance = self.distance
                self.viewer.cam.lookat[:] = np.array(self.lookat)
        else:
            self.viewer = None

        # Initialize SafetyWatchdogSim
        # self.watchdog = SafetyWatchdogSim(
        #     robot_controller=self.robot,
        #     natnet_data_handler=self.sensor.data_handler,
        #     joint_limits=self.cfg.RLConfig.JOINT_LIMITS, # Use joint limits from Hydra config
        #     marker_radii={} # Not used in simulation
        # )
        # self.watchdog.start()

        # If kinematic sampling is chosen, pre-generate reachable points
        if self.target_pos_sampling_method == "kinematic_sampling":
            print("[RL Env] Pre-generating kinematic reachable points...")
            model, data, end_effector_site_id = self.robot.get_info()
            self.kinematic_reachable_points = sample_kinematic_reachable_points(
                model, data, end_effector_site_id, self.cfg.RLConfig.JOINT_LIMITS, self.cfg.TrainConfig.kinematic_sampling_points, body_pos=self.body_pos,
            )
            print(f"[RL Env] Generated {len(self.kinematic_reachable_points)} kinematic reachable points.")
        
        # Define action and observation spaces
        # Actions: joint relative positions for the 6 controlled joints
        action_range_rad = np.radians(cfg.TrainConfig.action_range_deg)
        self.action_space = spaces.Box(
            low=-action_range_rad, high=action_range_rad, shape=(6,), dtype=np.float32
        )
        
        # Observations: controlled joint positions, controlled joint velocities, end effector position, target position
        num_joints = 6 # Assuming 6 controlled joints
        if self.reward_type == "pos":
            obs_dim = num_joints + 3 + 3  # 6 qpos, ee_pos, target_pos [12]
        elif self.reward_type == "pos+ori":
            obs_dim = num_joints + 3 + 4 + 3 + 4  # 6 qpos, ee_pos, ee_quat, target_pos, target_quat [20]
        elif self.reward_type == "pos+fixed_ori":
            obs_dim = num_joints + 3 + 4 + 3  # 6 qpos, ee_pos, ee_quat, target_pos [16]
        else:
            NotImplementedError(f"Unknown reward_type: {self.reward_type}")

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.target_position = np.zeros(3)
        self.target_orientation = np.array([1.0, 0.0, 0.0, 0.0]) # Quaternion (w, x, y, z)
        self.initial_ee_position = np.zeros(3)
        
        # Action scaling (adjust based on your robot's capabilities)
        self.action_scale = 1.0  # rad/s for joint velocities
        
        # Track previous action for smoothness penalty (6 controlled joints)
        self.previous_action = np.zeros(6)

        self.verbose_collisions = True  # Print collision details for debugging

        # Trapezoidal velocity profile parameters
        self.joint_target_positions = np.zeros(6) # Target position for each joint (radians)
        self.joint_start_positions = np.zeros(6)  # Start position for each joint (radians)
        self.joint_start_velocities = np.zeros(6) # Start velocity for each joint (radians/s)
        self.motion_start_time = 0.0              # Timestamp when motion started (real-world time)
        self.motion_duration = 0.0                # Total duration of the motion
        self._current_motion_sim_time = 0.0       # Accumulated simulation time for the current motion

        # Convert max velocity and acceleration to radians
        self.max_joint_velocity = np.radians(cfg.TrainConfig.max_speed / 100.0)
        self.max_joint_acceleration = np.radians(cfg.TrainConfig.motor_acceleration)

        self.simulation_steps_per_action = int(1.0 / (self.timestep * cfg.TrainConfig.control_frequency))
        self.simulation_steps_real_compensate = int(cfg.TrainConfig.real_time_latency / self.timestep) # Compensate for 0.09s real-world delay
        self.sample_k = self.simulation_steps_per_action - self.simulation_steps_real_compensate

        self._last_action = np.zeros(6)
        self.v_start_scale = cfg.TrainConfig.v_start_scale
        self.reset_random_scale = cfg.TrainConfig.reset_random_scale # Scale for random reset range
        self.clip_action_scale = cfg.TrainConfig.clip_action_scale # Scale for clipping action to avoid extreme commands

        # ===================== Domain Randomization =====================
        self.dr_cfg = getattr(cfg, "DomainRand", None)
        self.dr_enable = bool(self.dr_cfg and getattr(self.dr_cfg, "enable", False))

        # Cache base limits so we can resample each episode without drift
        self._base_max_joint_velocity = float(self.max_joint_velocity)
        self._base_max_joint_acceleration = float(self.max_joint_acceleration)

        # Action delay buffer (in controller steps, not sim sub-steps)
        self._dr_action_delay_steps = 0
        self._action_buffer = []  # list[np.ndarray], oldest first

        self._base_control_frequency = float(cfg.TrainConfig.control_frequency)
        self._base_real_time_latency = float(cfg.TrainConfig.real_time_latency)

        # Per-episode bias (optional)
        self._dr_action_bias = np.zeros(6, dtype=np.float32)
        self._dr_obs_ee_bias = np.zeros(3, dtype=np.float32)


        # Sampled DR params for current episode (for logging)
        self._dr_params = {
            "action_noise_std": 0.0,
            "action_delay_steps": 0,
            "vmax_scale": 1.0,
            "amax_scale": 1.0,
            "kp_scale": 1.0,
            "kd_scale": 1.0,
        }
        self._dr_params.update({
            "latency_sec": self._base_real_time_latency,
            "control_frequency_hz": self._base_control_frequency,
            "action_bias": self._dr_action_bias.copy(),
            "ee_pos_bias": self._dr_obs_ee_bias.copy(),
        })

        # Optional: deterministic RNG for DR
        if self.dr_enable and getattr(self.dr_cfg, "deterministic", False):
            self._dr_rng = np.random.default_rng(int(getattr(self.dr_cfg, "seed", 0)))
        else:
            self._dr_rng = np.random.default_rng()
        # ========================================================================
        print(f"[RL Env] initialized DR params: {self._dr_params}")


    def reset(self, seed: Optional[int] = None, options: Dict = {"joint_position": "random"}):
        """Reset the environment to initial state and sample new target."""
        super().reset(seed=seed)
        self.data.qvel[:] = 0

        # --- Domain randomization per episode ---
        self._sample_domain_randomization()
        
        # Move robot to home position using its own method
        if options["joint_position"] == "random":
            reset_joint_pos_deg = np.zeros(6)
            for i in range(6):
                # Use full joint limits for random reset, scaled by reset_random_scale
                reset_joint_pos_deg[i] = np.random.uniform(
                    low=self.cfg.RLConfig.JOINT_LIMITS[i][0] * self.reset_random_scale,
                    high=self.cfg.RLConfig.JOINT_LIMITS[i][1] * self.reset_random_scale,
                )
        elif options["joint_position"] == "home":
            reset_joint_pos_deg = np.zeros(6)
        elif options["joint_position"] == "freezed":
            reset_joint_pos_deg = np.zeros(6)
            fixed_angles = [17.3, 128.9, 60.2, 0, 0]
            # Joint 0 is random, others are fixed
            reset_joint_pos_deg[0] = np.random.uniform(
                low=self.cfg.RLConfig.JOINT_LIMITS[0][0] * self.reset_random_scale,
                high=self.cfg.RLConfig.JOINT_LIMITS[0][1] * self.reset_random_scale,
            )
            for i in range(1, 6):
                reset_joint_pos_deg[i] = fixed_angles[i-1]
        elif options["joint_position"] == "custom":
            reset_joint_pos_deg = np.array(options["custom_joint_pos_deg"])
        else:
            NotImplementedError(f"Unknown joint position option: {options['joint_position']}")
        
        # Set the initial joint positions in MuJoCo data directly
        self.data.qpos[:6] = np.radians(reset_joint_pos_deg)
        self.data.ctrl[:6] = np.radians(reset_joint_pos_deg)
        mujoco.mj_forward(self.model, self.data)

        # Initialize trapezoidal profile parameters
        current_joint_pos_deg = np.array([p[0] for p in self.robot.get_Position()])
        current_joint_pos_rad = np.radians(current_joint_pos_deg)
        self.joint_start_positions = current_joint_pos_rad.copy()
        self.joint_target_positions = current_joint_pos_rad.copy()
        self.joint_start_velocities = np.zeros(6) # Initialize start velocities to zero on reset
        self.motion_start_time = time.time()
        self.motion_duration = 0.0
        self._current_motion_sim_time = 0.0 # Reset simulation time for new motion
        self._last_action = np.zeros(6)
        self.previous_action = np.zeros(6)

        # Sample target position within reachable range
        if "target_position" in options:
            print(f"[RL Env] Resetting with provided target position: {options['target_position']}")
            self.target_position = options["target_position"]
        else:
            self.target_position = self._sample_target_position()
            print(f"[RL Env] Resetting with sampled target position: {self.target_position}")
        
        # Sample target orientation
        if "target_orientation" in options:
            print(f"[RL Env] Resetting with provided target orientation: {options['target_orientation']}")
            self.target_orientation = options["target_orientation"]
        else:
            if self.reward_type == "pos+fixed_ori":
                self.target_orientation = np.ndarray(self.fixed_target_orientation) # Fixed orientation (w, x, y, z)
            else:
                self.target_orientation = self._sample_target_orientation()

        # ==================================== target visulization ======================================
        try:
            target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target")
            self.model.site_pos[target_site_id] = self.target_position + self.body_pos  # Adjust for body position
        except:
            pass  # Target site doesn't exist, skip visualization

        if self.reward_type != "pos":
            # Add target orientation visualization (if target_orientation site exists in model)
            target_pos_for_sites = self.target_position + self.body_pos

            # Base quaternions for axes (cylinder points along Z by default)
            # X-axis: Rotate Z to X (90 deg around Y)
            base_rot_x = R.from_euler('y', 90, degrees=True)
            # Y-axis: Rotate Z to Y (-90 deg around X)
            base_rot_y = R.from_euler('x', -90, degrees=True)
            # Z-axis: No rotation
            base_rot_z = R.from_quat([0, 0, 0, 1]) # Identity quaternion (xyzw)

            try:
                # X-axis
                site_id_x = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_orientation_x")
                if site_id_x != -1:
                    final_rot_x = R.from_quat(self.target_orientation[[1,2,3,0]]) * base_rot_x
                    # Calculate offset in world coordinates
                    local_offset_x = np.array([0, 0, 0.05]) # half-length for X-axis
                    rotated_offset_x = final_rot_x.apply(local_offset_x)
                    self.model.site_pos[site_id_x] = target_pos_for_sites + rotated_offset_x
                    self.model.site_quat[site_id_x] = final_rot_x.as_quat()[[3,0,1,2]] # Convert xyzw to wxyz

                # Y-axis
                site_id_y = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_orientation_y")
                if site_id_y != -1:
                    final_rot_y = R.from_quat(self.target_orientation[[1,2,3,0]]) * base_rot_y
                    # Calculate offset in world coordinates
                    local_offset_y = np.array([0, 0, 0.05]) # half-length for Y-axis
                    rotated_offset_y = final_rot_y.apply(local_offset_y)
                    self.model.site_pos[site_id_y] = target_pos_for_sites + rotated_offset_y
                    self.model.site_quat[site_id_y] = final_rot_y.as_quat()[[3,0,1,2]] # Convert xyzw to wxyz

                # Z-axis
                site_id_z = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_orientation_z")
                if site_id_z != -1:
                    final_rot_z = R.from_quat(self.target_orientation[[1,2,3,0]]) * base_rot_z
                    # Calculate offset in world coordinates
                    local_offset_z = np.array([0, 0, 0.05]) # half-length for Z-axis
                    rotated_offset_z = final_rot_z.apply(local_offset_z)
                    self.model.site_pos[site_id_z] = target_pos_for_sites + rotated_offset_z
                    self.model.site_quat[site_id_z] = final_rot_z.as_quat()[[3,0,1,2]] # Convert xyzw to wxyz

                # print(f"[RL Env] Resetting with sampled target orientation: {self.target_orientation}")
            except Exception as e:
                print(f"[RL Env] Error resetting target orientation sites: {e}")
                pass # Target orientation sites don't exist, skip visualization
        # =====================================================================================================
            
        # Forward kinematics to get initial end effector position and update visualization
        # mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)
        self.initial_ee_position = self.data.site_xpos[self.robot._end_effector_site_id].copy()
        self.initial_ee_position = self.initial_ee_position.copy() - self.body_pos

        self.current_step = 0
        self.previous_action = np.zeros(6)
        observation, _, _ = self._get_observation()
        info = self._get_info(current_action=np.zeros(6), observation=observation)  # No action since just reset

        return observation, info
    
    @staticmethod
    def joints_clip(joint1, joint2, min, max):
        # suppose the two joints have the same direction.
        current_sum = joint1 + joint2
        if current_sum > max:
            diff = current_sum - max
            joint1 -= diff / 2
            joint2 -= diff / 2
        elif current_sum < min:
            diff = min - current_sum
            joint1 += diff / 2
            joint2 += diff / 2
        return joint1, joint2
    
    def step(self, action: np.ndarray):
        """Execute one step in the environment using a trapezoidal velocity profile."""
        # print(f"[DEBUG] step() called with action: {action}")

        # Check for emergency state from watchdog (copied from base class)
        if self.robot._in_emergency_state:
            reward = -100.0 # Large negative reward
            print(f"[RL Env] Emergency state triggered! Severe penalty {reward} applied.")
            terminated = True
            truncated = False
            while self.robot._in_emergency_state:
                print("[RL Env] Waiting for robot to exit emergency state before reset...")
                time.sleep(0.2)
            
            observation, _, _ = self._get_observation()
            info = self._get_info(current_action=np.zeros(6), observation=observation)
            info["emergency_state"] = True
            return observation, reward, terminated, truncated, info
        
        current_action = np.zeros(6) # Default to zero if no action provided
        if action is not None:
            current_action = np.array(action, dtype=np.float32)

            # 1) action delay: buffer in controller steps
            if self._dr_action_delay_steps > 0:
                self._action_buffer.append(current_action.copy())
                current_action = self._action_buffer.pop(0)  # delayed action
            else:
                current_action = current_action

            if self.dr_enable:
                current_action = current_action + self._dr_action_bias

            # 2) action noise (additive Gaussian), then clip to action_space
            if self.dr_enable and self._dr_params.get("action_noise_std", 0.0) > 0.0:
                std = float(self._dr_params["action_noise_std"])
                current_action = current_action + self._dr_rng.normal(0.0, std, size=current_action.shape).astype(np.float32)

            # clip to env action bounds (important after noise)
            current_action = np.clip(current_action, self.action_space.low, self.action_space.high).astype(np.float32)

            # ----- 1. Get current joint positions and velocities in degrees -----
            current_joint_pos_vel_deg = self.robot.get_PosVel()
            current_joint_pos_deg = np.array([pv[0] for pv in current_joint_pos_vel_deg])
            current_joint_vel_deg = np.array([pv[1] for pv in current_joint_pos_vel_deg])
            current_joint_pos_rad = np.radians(current_joint_pos_deg).copy()
            current_joint_vel_rad = np.radians(current_joint_vel_deg).copy()

            # ----- 2. Calculate target joint positions in degrees based on the action -----
            scaled_action = current_action * self.action_scale # Action is a relative change in radians
            # print(f"[DEBUG] current_action: {current_action}, scaled_action: {scaled_action}")
            scaled_action_deg = np.degrees(scaled_action)
            target_joint_pos_deg = current_joint_pos_deg + scaled_action_deg
            # print(f"[DEBUG] current_joint_pos_deg: {current_joint_pos_deg}, target_joint_pos_deg: {target_joint_pos_deg}")
            # Clip the joint positions (copied from base class)
            assert len(target_joint_pos_deg) == len(self.cfg.RLConfig.JOINT_LIMITS)
            for j in range(len(target_joint_pos_deg)):
                target_joint_pos_deg[j] = np.clip(target_joint_pos_deg[j], self.cfg.RLConfig.JOINT_LIMITS[j][0]*self.clip_action_scale, self.cfg.RLConfig.JOINT_LIMITS[j][1]*self.clip_action_scale)
            # Clip the combined angle of joint 2 and 3 (copied from base class)
            if self.cfg.RLConfig.COMBINED_JOINT_2_3_LIMIT > 0:
                joint1, joint2 = self.joints_clip(target_joint_pos_deg[1], -target_joint_pos_deg[2], -self.cfg.RLConfig.COMBINED_JOINT_2_3_LIMIT, self.cfg.RLConfig.COMBINED_JOINT_2_3_LIMIT)
                target_joint_pos_deg[1] = joint1
                target_joint_pos_deg[2] = -joint2
            # Convert to radians for trapezoidal profile calculation
            target_joint_pos_rad = np.radians(target_joint_pos_deg).copy()

            # ----- 3. Update motion profile parameters -----
            self.joint_start_positions = current_joint_pos_rad
            self.joint_target_positions = target_joint_pos_rad
            self.motion_start_time = time.time()
            self._current_motion_sim_time = 0.0 # Reset simulation time for new motion

            # Calculate motion duration for each joint and find the maximum
            max_duration = 0.0
            for i in range(len(self.joint_start_positions)):
                q_start = self.joint_start_positions[i]
                q_end = self.joint_target_positions[i]
                v_start_raw = self.joint_start_velocities[i] # Raw start velocity
                v_max = self.max_joint_velocity
                a_max = self.max_joint_acceleration
                
                delta_q = q_end - q_start
                direction = np.sign(delta_q) if delta_q != 0 else 1.0

                # Adjust v_start based on direction of motion
                if delta_q != 0 and np.sign(v_start_raw) != direction:
                    v_start = 0.0
                else:
                    v_start = direction * min(abs(v_start_raw), v_max)

                if abs(delta_q) < 1e-6 and abs(v_start) < 1e-6: # If no displacement and no initial velocity, duration is 0
                    duration = 0.0
                else:
                    # Calculate time to accelerate from v_start to v_max (Ta1)
                    Ta1 = (v_max - abs(v_start)) / a_max if v_max > abs(v_start) else 0.0
                    Sa1 = abs(v_start) * Ta1 + 0.5 * a_max * Ta1**2

                    # Calculate time to decelerate from v_max to 0 (Ta2)
                    Ta2 = v_max / a_max
                    Sa2 = 0.5 * a_max * Ta2**2

                    # Total distance required for acceleration to v_max and deceleration from v_max
                    S_accel_decel = Sa1 + Sa2

                    if S_accel_decel * direction > abs(delta_q) * direction: # Check if triangular profile is needed
                        # Triangular profile: calculate actual peak velocity and total time
                        # Solve for v_peak: 2*a_max*abs(delta_q) + v_start^2 = 2*v_peak^2
                        v_peak = np.sqrt((2 * a_max * abs(delta_q) + v_start**2) / 2)
                        v_peak = min(v_peak, v_max) # Ensure peak velocity doesn't exceed v_max

                        Ta_accel = (v_peak - abs(v_start)) / a_max
                        Ta_decel = v_peak / a_max
                        duration = Ta_accel + Ta_decel
                    else:
                        # Trapezoidal profile
                        Tv = (abs(delta_q) - S_accel_decel) / v_max
                        duration = Ta1 + Tv + Ta2
                max_duration = max(max_duration, duration)
            self.motion_duration = max_duration
        
        # Step the simulation with several sim steps per action, applying trapezoidal profile
        # action_timestamp_2 = time.time() # Initialize here, will be updated in loop
        for k in range(self.simulation_steps_per_action):
            # Use accumulated simulation time for the trapezoidal profile
            t_profile = min(self._current_motion_sim_time, self.motion_duration)
            
            # Calculate desired joint positions using trapezoidal profile
            desired_joint_pos_rad = np.zeros(6)
            desired_joint_vel_rad = np.zeros(6)
            for i in range(len(self.joint_start_positions)):
                q_start = self.joint_start_positions[i]
                q_end = self.joint_target_positions[i]
                v_start = self.joint_start_velocities[i] * self.v_start_scale  # Pass start velocity
                v_max = self.max_joint_velocity
                a_max = self.max_joint_acceleration
                
                pos, vel, _ = trapezoidal_velocity_profile_non_zero_start(q_start, q_end, v_start, v_max, a_max, t_profile)
                # pos, vel, _ = trapezoidal_velocity_profile(q_start, q_end, v_max, a_max, t_profile)
                desired_joint_pos_rad[i] = pos
                desired_joint_vel_rad[i] = vel
                # Debug prints for each joint
                # print(f"[DEBUG] Joint {i}: t_profile={t_profile:.4f}, q_start={np.degrees(q_start):.2f}, q_end={np.degrees(q_end):.2f}, delta_q={np.degrees(delta_q):.2f}, v_start={np.degrees(v_start):.2f}, pos={np.degrees(pos):.2f}, vel={np.degrees(vel):.2f}")

            # Convert desired joint positions back to degrees
            desired_joint_pos_deg = np.degrees(desired_joint_pos_rad)

            # Apply to robot
            self.robot.move_abs_direct(desired_joint_pos_deg)
            # print(f"[DEBUG] Applied desired_joint_pos_deg: {desired_joint_pos_deg}")
            # action_timestamp_2 = time.time() # Update timestamp after applying action
            # reset the simulation solver every 500 simulation steps:
            if self.current_step % 500 == 0:
                # mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
                mujoco.mj_forward(self.model, self.data)  # Forward kinematics to update state
                mujoco.mj_collision(self.model, self.data)

            mujoco.mj_forward(self.model, self.data)  # Forward kinematics to update state
            mujoco.mj_collision(self.model, self.data)
            # mujoco.mj_step(self.model, self.data)
            # print(f"[RL Env] Step {self.current_step}, Sim sub-step {k+1}/{self.simulation_steps_per_action}, Applied joint positions (deg): {desired_joint_pos_deg}, Joint velocities (deg/s): {np.degrees(desired_joint_vel_rad)}")
            # mujoco.mj_step(self.model, self.data)  # DO NOT use this mj_step() function! pos gap issue TODO
            if self.viewer: # Sync viewer if human render mode
                self.viewer.sync()
            
            # Increment simulation time for the current motion
            self._current_motion_sim_time += self.timestep
            if self.real_time_speed:
                time.sleep(self.timestep)  # Sleep to simulate real-time

            # if k == (self.simulation_steps_per_action - self.simulation_steps_real_compensate):
            #     # Get current state the actual action tasks about 0.09s
            #     observation, optitrack_timestamp, proprioception_timestamp = self._get_observation()
            #     # print(f"get the observation at sim step {k}, time compensate {self.simulation_steps_real_compensate}, optitrack timestamp: {optitrack_timestamp}, proprioception timestamp: {proprioception_timestamp}")

            if k == self.sample_k:
                observation, optitrack_timestamp, proprioception_timestamp = self._get_observation()


        self.joint_start_velocities = desired_joint_vel_rad

        # print(f"[RL Env] Executed action: {current_action}, New observation: {observation}")
        reward, terminated = self._compute_reward(current_action, observation) # Using isaac reward from base class

        # timestamp_diff_action_obs = action_timestamp_2 - proprioception_timestamp

        info = self._get_info(current_action, observation)
        info["domain_rand"] = self._dr_params.copy()
        info["emergency_state"] = self.robot._in_emergency_state
        # info["timestamp_diff_action_obs"] = timestamp_diff_action_obs
        
        # Update previous action for next step (copied from base class)
        self.previous_action = current_action.copy()

        # Check termination conditions (copied from base class)
        self.current_step += 1
        # terminated = False # Base class also sets this to False
        truncated = self.current_step >= self.max_episode_steps or terminated
        if truncated:
            self._episode_end_ee_distance = info["distance_to_target"]
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        proprioception_timestamp = time.time()
        qpos = np.radians(np.array([p[0] for p in self.robot.get_Position()]))
        optitrack_timestamp = time.time()
        ee_pos, ee_quat = self.sensor.get_relative_pose()
        ee_pos = ee_pos.copy() - self.body_pos  # Make ee_pos relative to body position
        target_pos = self.target_position.copy()
        target_quat = self.target_orientation.copy() # Currently not used in observation

        # -------- DR: observation noise --------
        if self.dr_enable:
            ee_pos = ee_pos + self._dr_obs_ee_bias
            obs_cfg = getattr(self.dr_cfg, "obs_noise", None)
            if obs_cfg and getattr(obs_cfg, "enable", False):
                ee_std = float(getattr(obs_cfg, "ee_pos_std", 0.0))
                q_std = float(getattr(obs_cfg, "qpos_std", 0.0))
                if ee_std > 0:
                    ee_pos = ee_pos + self._dr_rng.normal(0.0, ee_std, size=ee_pos.shape)
                if q_std > 0:
                    qpos = qpos + self._dr_rng.normal(0.0, q_std, size=qpos.shape)
        # -----------------------------------------------

        
        # Concatenate all observations
        if self.reward_type == "pos":
            obs = np.concatenate([qpos, ee_pos, target_pos])  # 6+3+3=12
        elif self.reward_type == "pos+ori":
            obs = np.concatenate([qpos, ee_pos, ee_quat, target_pos, target_quat])  # 6+7+7=20
        elif self.reward_type == "pos+fixed_ori":
            obs = np.concatenate([qpos, ee_pos, ee_quat, target_pos])  # 6+3+4+3=16
        else:
            NotImplementedError(f"Unknown reward_type: {self.reward_type}")

        return obs.astype(np.float32), optitrack_timestamp, proprioception_timestamp

    def _compute_reward(
            self, 
            action: np.ndarray, 
            observation: np.ndarray,
            ) -> float:
        reward = 0.0

        if self.reward_type == "pos":
            ee_pos = observation[6:9]
            target_pos = observation[9:12]
        elif self.reward_type == "pos+ori":
            ee_pos = observation[6:9]
            ee_quat = observation[9:13]
            target_pos = observation[13:16]
            target_quat = observation[16:20]
        elif self.reward_type == "pos+fixed_ori":
            ee_pos = observation[6:9]
            ee_quat = observation[9:13]
            target_pos = observation[13:16]
            target_quat = self.target_orientation.copy() # Fixed orientation (w, x, y, z)

        # 1. Pos dist:
        distance_to_target = np.linalg.norm(target_pos - ee_pos)
        self._current_distance = distance_to_target
        # print(f"[RL Env] EE Pos: {ee_pos}, Target Pos: {target_pos}, Distance: {distance_to_target}")

        # Reward based on distance and velocity towards target
        distance_reward = self.reward_weights['alpha'] * distance_to_target
        reward += distance_reward

        fine_grained_distance = 1 - np.tanh(distance_to_target/0.1)
        fine_grained_reward = self.reward_weights['beta'] * fine_grained_distance
        reward += fine_grained_reward

        # 2. Ori dist:
        if self.reward_type == "pos+ori" or self.reward_type == "pos+fixed_ori":
            quat_diff = quat_mul_np(ee_quat, quat_conjugate_np(target_quat))  # q1 * q2^-1
            axis_angle_diff = axis_angle_from_quat_np(quat_diff)
            orientation_difference = np.linalg.norm(axis_angle_diff)
            self._current_orientation_difference = orientation_difference
            orientation_reward = self.reward_weights['delta'] * orientation_difference
            reward += orientation_reward

            fine_grained_ori_distance = 1 - np.tanh(orientation_difference/0.2)
            fine_grained_ori_reward = self.reward_weights['epsilon'] * fine_grained_ori_distance
            reward += fine_grained_ori_reward

        # 3. Action smoothness penalty (discourage jitter)
        action_difference = np.linalg.norm(action - self.previous_action)
        self._current_action_difference = action_difference
        action_smoothness_penalty = self.reward_weights['gamma'] * action_difference

        reward += action_smoothness_penalty

        action_penalty = self.reward_weights['gamma'] * np.linalg.norm(action)  # gamma = -0.0001

        reward += action_penalty

        # 4. Self-collision penalty

        if self.collision_detection:
            reward += self.reward_weights['zeta'] * self._check_self_collision_jointwise_soft()
            reward += self.reward_weights['eta'] * self._check_self_collision_totable_soft()

        # 5. Success bonus
        terminated = False
        if distance_to_target < self.distance_threshold:
            reward += self.reward_weights['success']
            terminated = True

        return reward, terminated

    def _is_success(self) -> bool:
        """Check if the task is successfully completed."""
        ee_pos, _ = self.sensor.get_relative_pose()
        ee_pos = ee_pos.copy() - self.body_pos  # Make ee_pos relative to body position
        distance = np.linalg.norm(ee_pos - self.target_position)
        return distance < self.distance_threshold
    
    def _sample_domain_randomization(self) -> None:
        """
        Sample per-episode domain randomization parameters (minimal set).
        This focuses on control/actuation randomness: delay, action noise, vmax/amax scaling.
        """
        # Reset model parameters to base before sampling new ones
        if hasattr(self, "_base_actuator_gainprm"):
            self.model.actuator_gainprm[:] = self._base_actuator_gainprm
            self.model.actuator_biasprm[:] = self._base_actuator_biasprm
        else:
            self._base_actuator_gainprm = self.model.actuator_gainprm.copy()
            self._base_actuator_biasprm = self.model.actuator_biasprm.copy()

        if not self.dr_enable:
            # reset to base
            self.max_joint_velocity = self._base_max_joint_velocity
            self.max_joint_acceleration = self._base_max_joint_acceleration
            self._dr_action_delay_steps = 0
            self._action_buffer = []
            self._dr_params.update({
                "action_noise_std": 0.0,
                "action_delay_steps": 0,
                "vmax_scale": 1.0,
                "amax_scale": 1.0,
                "kp_scale": 1.0,
                "kd_scale": 1.0,
            })
            return

        # ---------- action noise ----------
        noise_cfg = getattr(self.dr_cfg, "action_noise", None)
        if noise_cfg and getattr(noise_cfg, "enable", False):
            action_noise_std = float(getattr(noise_cfg, "std", 0.0))
        else:
            action_noise_std = 0.0

        # ---------- action delay (controller-step buffer) ----------
        delay_cfg = getattr(self.dr_cfg, "action_delay", None)
        if delay_cfg and getattr(delay_cfg, "enable", False):
            min_steps = int(getattr(delay_cfg, "min_steps", 0))
            max_steps = int(getattr(delay_cfg, "max_steps", 0))
            if max_steps < min_steps:
                max_steps = min_steps
            delay_steps = int(self._dr_rng.integers(min_steps, max_steps + 1))
        else:
            delay_steps = 0

        # ---------- vmax/amax scaling ----------
        dyn_cfg = getattr(self.dr_cfg, "dynamics_limits", None)
        if dyn_cfg and getattr(dyn_cfg, "enable", False):
            vmax_scale_cfg = getattr(dyn_cfg, "vmax_scale", None)
            amax_scale_cfg = getattr(dyn_cfg, "amax_scale", None)
            kp_scale_cfg = getattr(dyn_cfg, "kp_scale", None)
            kd_scale_cfg = getattr(dyn_cfg, "kd_scale", None)

            vmax_low = float(getattr(vmax_scale_cfg, "low", 1.0)) if vmax_scale_cfg else 1.0
            vmax_high = float(getattr(vmax_scale_cfg, "high", 1.0)) if vmax_scale_cfg else 1.0
            amax_low = float(getattr(amax_scale_cfg, "low", 1.0)) if amax_scale_cfg else 1.0
            amax_high = float(getattr(amax_scale_cfg, "high", 1.0)) if amax_scale_cfg else 1.0
            kp_low = float(getattr(kp_scale_cfg, "low", 1.0)) if kp_scale_cfg else 1.0
            kp_high = float(getattr(kp_scale_cfg, "high", 1.0)) if kp_scale_cfg else 1.0
            kd_low = float(getattr(kd_scale_cfg, "low", 1.0)) if kd_scale_cfg else 1.0
            kd_high = float(getattr(kd_scale_cfg, "high", 1.0)) if kd_scale_cfg else 1.0

            vmax_scale = float(self._dr_rng.uniform(vmax_low, vmax_high))
            amax_scale = float(self._dr_rng.uniform(amax_low, amax_high))
            kp_scale = float(self._dr_rng.uniform(kp_low, kp_high))
            kd_scale = float(self._dr_rng.uniform(kd_low, kd_high))
        else:
            vmax_scale = 1.0
            amax_scale = 1.0
            kp_scale = 1.0
            kd_scale = 1.0

        # Apply to env
        self.max_joint_velocity = self._base_max_joint_velocity * vmax_scale
        self.max_joint_acceleration = self._base_max_joint_acceleration * amax_scale

        # Apply kp/kd randomization to Mujoco actuators
        if self.dr_enable and (kp_scale != 1.0 or kd_scale != 1.0):
            for i in range(self.model.nu):
                # Mujoco stores actuator gains in model.actuator_gainprm
                # For position actuators: gainprm[0] is kp
                # For velocity actuators: gainprm[0] is kv (kd)
                # We need to check the actuator type or just apply based on index if we know the mapping
                # In our case, actuators are added for each joint.
                self.model.actuator_gainprm[i, 0] = self._base_actuator_gainprm[i, 0] * kp_scale
                # Bias parameters for position actuators: biasprm[1] is -kp, biasprm[2] is -kv
                self.model.actuator_biasprm[i, 1] = self._base_actuator_biasprm[i, 1] * kp_scale
                self.model.actuator_biasprm[i, 2] = self._base_actuator_biasprm[i, 2] * kd_scale

        self._dr_action_delay_steps = delay_steps
        self._action_buffer = [np.zeros(6, dtype=np.float32) for _ in range(delay_steps)]

        # Store for logging
        self._dr_params.update({
            "action_noise_std": float(action_noise_std),
            "action_delay_steps": int(delay_steps),
            "vmax_scale": float(vmax_scale),
            "amax_scale": float(amax_scale),
            "kp_scale": float(kp_scale),
            "kd_scale": float(kd_scale),
        })

        # ---------- timing DR: latency + control frequency ----------
        timing_cfg = getattr(self.dr_cfg, "timing", None)
        if timing_cfg and getattr(timing_cfg, "enable", False):
            # latency
            lat_cfg = getattr(timing_cfg, "latency_sec", None)
            if lat_cfg is not None:
                lat_low = float(getattr(lat_cfg, "low", self._base_real_time_latency))
                lat_high = float(getattr(lat_cfg, "high", self._base_real_time_latency))
                latency_sec = float(self._dr_rng.uniform(lat_low, lat_high))
            else:
                latency_sec = self._base_real_time_latency

            # control frequency
            cf_cfg = getattr(timing_cfg, "control_frequency_hz", None)
            if cf_cfg is not None:
                cf_low = float(getattr(cf_cfg, "low", self._base_control_frequency))
                cf_high = float(getattr(cf_cfg, "high", self._base_control_frequency))
                control_freq = float(self._dr_rng.uniform(cf_low, cf_high))
            else:
                control_freq = self._base_control_frequency
        else:
            latency_sec = self._base_real_time_latency
            control_freq = self._base_control_frequency

        # Apply to env timing
        self.simulation_steps_per_action = max(1, int(round(1.0 / (self.timestep * control_freq))))
        self.simulation_steps_real_compensate = max(0, int(round(latency_sec / self.timestep)))

        self.sample_k = self.simulation_steps_per_action - self.simulation_steps_real_compensate
        self.sample_k = int(np.clip(self.sample_k, 0, self.simulation_steps_per_action - 1))

        # ---------- optional: action bias ----------
        ab_cfg = getattr(self.dr_cfg, "action_bias", None)
        if ab_cfg and getattr(ab_cfg, "enable", False):
            ab_std = float(getattr(ab_cfg, "std", 0.0))
            self._dr_action_bias = self._dr_rng.normal(0.0, ab_std, size=(6,)).astype(np.float32)
        else:
            self._dr_action_bias = np.zeros(6, dtype=np.float32)

        # ---------- optional: obs bias (EE pos) ----------
        ob_cfg = getattr(self.dr_cfg, "obs_bias", None)
        if ob_cfg and getattr(ob_cfg, "enable", False):
            ee_std = float(getattr(ob_cfg, "ee_pos_std", 0.0))
            self._dr_obs_ee_bias = self._dr_rng.normal(0.0, ee_std, size=(3,)).astype(np.float32)
        else:
            self._dr_obs_ee_bias = np.zeros(3, dtype=np.float32)

        # Store for logging
        self._dr_params.update({
            "latency_sec": float(latency_sec),
            "control_frequency_hz": float(control_freq),
            "action_bias": self._dr_action_bias.copy(),
            "ee_pos_bias": self._dr_obs_ee_bias.copy(),
        })
        print(f"[RL Env] Sampled DR parameters: {self._dr_params}")
    
    def _sample_target_position(self) -> np.ndarray:
        """Sample a random target position within reachable range based on the selected method."""
        if self.target_pos_sampling_method == "sphere":
            # Sample within a sphere around the initial end effector position
            target_pos = np.array(sample_point_in_sphere(self.sphere_target_range, h=1.0))
        elif self.target_pos_sampling_method == "fixed":
            target_pos = np.array(self.cfg.RLConfig.FIXED_TARGET_POSITION)
        elif self.target_pos_sampling_method == "cylinder":
            # Sample within a cylinder around the initial end effector position
            # Cylinder height is related to the robot arm's length
            target_pos = np.array(sample_point_in_cylinder(self.cfg.RLConfig.TARGET_CYLINDER_RADIUS, self.cfg.RLConfig.TARGET_CYLINDER_HEIGHT_MAX, self.cfg.RLConfig.TARGET_CYLINDER_HEIGHT_MIN))
            while np.linalg.norm(target_pos) < self.cfg.RLConfig.TARGET_CYLINDER_MINIMUM_DISTANCE:
                target_pos = np.array(sample_point_in_cylinder(self.cfg.RLConfig.TARGET_CYLINDER_RADIUS, self.cfg.RLConfig.TARGET_CYLINDER_HEIGHT_MAX, self.cfg.RLConfig.TARGET_CYLINDER_HEIGHT_MIN))
        elif self.target_pos_sampling_method == "cube":
            # Sample within a cube around the initial end effector position
            target_pos = np.array(sample_point_in_cube(self.cfg))
        elif self.target_pos_sampling_method == "kinematic_sampling":
            if self.kinematic_reachable_points is None:
                raise RuntimeError("Kinematic reachable points not generated. Check TARGET_SAMPLING_METHOD in config.")
            # Randomly select a pre-generated reachable point
            idx = np.random.randint(0, len(self.kinematic_reachable_points))
            target_pos = self.kinematic_reachable_points[idx]
        elif self.target_pos_sampling_method == "table":
            target_pos = np.array(sample_point_in_table(self.cfg))
        else:
            raise ValueError(f"Unknown target sampling method: {self.target_pos_sampling_method}")
        
        # Ensure target_pos z-coordinate is positive (above ground)
        if target_pos[2] < 0:
            target_pos[2] = -target_pos[2]

        # target = target_pos + self.body_pos  # TODO: a more decent way to set the target
        target = target_pos
        return target

    def _sample_target_orientation(self) -> np.ndarray:
        """Sample a random target orientation as a quaternion (w, x, y, z)."""
        random_rotation = R.random()
        quat_xyzw = random_rotation.as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        # print(f"Sampled quaternion (w,x,y,z): {quat_wxyz}") # Debug print
        return quat_wxyz

    def _get_info(self, current_action, observation: np.ndarray) -> Dict[str, Any]:
        """Get additional information about the current state."""

        ee_pos = observation[6:9]
        # ee_quat = observation[9:13]
        fine_grained_distance = 1 - np.tanh(self._current_distance/0.1)
        # Calculate individual reward components for debugging
        return {
            "distance_to_target": self._current_distance,
            "orientation_difference": self._current_orientation_difference,
            # "is_success": self._is_success(),
            "end_effector_position": ee_pos.copy(),
            # "end_effector_orientation": ee_quat.copy(),
            "target_position": self.target_position.copy(),
            "target_orientation": self.target_orientation.copy(),
            "action_smoothness": self._current_action_difference,
            "reward_components": {
                "distance_to_target_reward": self.reward_weights['alpha'] * self._current_distance,
                "fine_grained_distance_to_target_reward": self.reward_weights['beta'] * fine_grained_distance,
                "action_smoothness_penalty": self.reward_weights['gamma'] * self._current_action_difference,
                "orientation_reward": self.reward_weights['delta'] * self._current_orientation_difference
            }
        }
    
    def _check_collision(self) -> bool:
        """
        Check if the robot has collided with the table or other environment objects.
        
        Returns:
            bool: True if collision detected, False otherwise
        """
        # Define table geometry names to check for collisions
        table_geom_names = [
            "table_top",
            "table_leg_1", 
            "table_leg_2",
            "table_leg_3", 
            "table_leg_4"
        ]
        
        # Get all contact points in the simulation
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            # Get the names of the two bodies in contact
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            
            # Get geometry names (handle potential errors)
            try:
                geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
                geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)
            except:
                continue  # Skip if name lookup fails
            
            # Check if either geometry is part of the table
            if geom1_name in table_geom_names or geom2_name in table_geom_names:
                # Check if the other geometry is part of the robot
                robot_geom_names = [
                    "base_link", "link1", "link2", "link3", "link4", "link5", "link6",
                    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"
                ]
                
                # More flexible robot collision detection
                robot_collision = False
                for robot_geom in robot_geom_names:
                    if (robot_geom in geom1_name or robot_geom in geom2_name or 
                        geom1_name.startswith("lynx") or geom2_name.startswith("lynx")):
                        robot_collision = True
                        break
                
                if robot_collision:
                    if self.verbose_collisions:
                        print(f"Collision detected between {geom1_name} and {geom2_name}")
                    return True
        
        return False

    def check_collisions_filtered(self):
        """
        Returns:
        self_collision: robot-robot collision after filtering internal same-joint / same-tube collisions
        env_collision: robot-table/floor collision (not filtered)
        contacts_kept: list of tuples (name1, name2) kept after filtering (for debug)
        """
        self_collision = False
        env_collision = False
        contacts_kept = []

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            n1 = _geom_name(self.model, c.geom1)
            n2 = _geom_name(self.model, c.geom2)

            if not n1 or not n2:
                continue

            # classify environment objects (customize if needed)
            is_env1 = (n1.startswith("table_") or n1 == "floor")
            is_env2 = (n2.startswith("table_") or n2 == "floor")

            is_robot1 = _is_robot_geom(n1)
            is_robot2 = _is_robot_geom(n2)

            # -------------------------
            # ENV collisions: keep them
            # -------------------------
            if (is_robot1 and is_env2) or (is_robot2 and is_env1):
                env_collision = True
                contacts_kept.append((n1, n2))
                continue

            # ---------------------------------------------------
            # SELF collisions: filter out internal same-part ones
            # ---------------------------------------------------
            if is_robot1 and is_robot2:
                # ignore same-joint internal collisions
                if _is_same_joint_internal(n1, n2):
                    continue

                # ignore same-tube segment collisions
                if _is_same_tube_segments(n1, n2):
                    continue

                # otherwise: it's a meaningful self-collision
                self_collision = True
                contacts_kept.append((n1, n2))

        return self_collision, env_collision, contacts_kept
    
    def _check_self_collision_jointwise(self) -> int:
        """
        Count self-collision occurrences using distances between joint centers.

        Returns:
            int: number of colliding joint-pairs (non-adjacent pairs only)
        """
        joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        joint_points = []  # list of (name, position)

        for name in joint_names:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            if site_id != -1:
                joint_points.append((name, self.data.site_xpos[site_id].copy()))
                continue

            # Fallback to body position if site not found
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id != -1:
                joint_points.append((name, self.data.xpos[body_id].copy()))

        # Minimum safe distance between joint centers (adjust based on robot geometry)
        min_safe_dist = 0.1  # 10cm

        collision_count = 0
        n = len(joint_points)

        # Check distances between joints that are not immediate neighbors
        for i in range(n):
            for j in range(i + 2, n):  # Skip immediate neighbors (i, i+1) [4+3+2+1=10 pairs]
                name_i, pos_i = joint_points[i]
                name_j, pos_j = joint_points[j]
                dist = np.linalg.norm(pos_i - pos_j)

                if dist < min_safe_dist:
                    collision_count += 1
                    if self.verbose_collisions:
                        print(
                            f"[RL Env] Self-collision pair: **{name_i}** vs **{name_j}** | dist={dist:.4f} (<{min_safe_dist})"
                        )

        if self.verbose_collisions and collision_count > 0:
            print(f"[RL Env] Total self-collision pairs: {collision_count}")

        return collision_count
    
    def _check_self_collision_jointwise_soft(self) -> int:
        """
        Count self-collision occurrences using distances between joint centers.

        Returns:
            int: number of colliding joint-pairs (non-adjacent pairs only)
        """
        joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        joint_points = []  # list of (name, position)

        for name in joint_names:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            if site_id != -1:
                joint_points.append((name, self.data.site_xpos[site_id].copy()))
                continue

            # Fallback to body position if site not found
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id != -1:
                joint_points.append((name, self.data.xpos[body_id].copy()))

        # Minimum safe distance between joint centers (adjust based on robot geometry)
        min_safe_dist = 0.10  # 10cm
        min_soft_dist = 0.135  # 13.5cm

        n = len(joint_points)
        cost = 0

        # Check distances between joints that are not immediate neighbors
        for i in range(n):
            for j in range(i + 2, n):  # Skip immediate neighbors (i, i+1) [4+3+2+1=10 pairs]
                name_i, pos_i = joint_points[i]
                name_j, pos_j = joint_points[j]
                dist = np.linalg.norm(pos_i - pos_j)

                if dist < min_safe_dist:
                    cost += 1
                    if self.verbose_collisions:
                        print(
                            f"[RL Env] DIRECT Self-collision pair: **{name_i}** vs **{name_j}** | dist={dist:.4f} (<{min_safe_dist})"
                        )

                if dist < min_soft_dist:
                    x = (min_soft_dist - dist) / (min_soft_dist - min_safe_dist)
                    if self.verbose_collisions:
                        print(
                            f"[RL Env] SOFT Self-collision pair: **{name_i}** vs **{name_j}** | dist={dist:.4f} (<{min_safe_dist})"
                        )
                    cost += x*x

        if self.verbose_collisions and cost > 0:
            print(f"[RL Env] Total self-collision cost: {cost:.4f}")

        return cost

    def _check_self_collision_totable(self) -> int:
        """
        Count table-collision occurrences by checking each joint point against table height.

        Returns:
            int: number of joints that are below the table surface threshold
        """
        joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        joint_points = []  # list of (name, position)

        for name in joint_names:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            if site_id != -1:
                joint_points.append((name, self.data.site_xpos[site_id].copy()))
                continue

            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id != -1:
                joint_points.append((name, self.data.xpos[body_id].copy()))

        table_height = float(self.body_pos[2])  # table top z
        buffer = -0.03  # 5cm above table surface counts as collision

        collision_count = 0
        threshold = table_height - buffer

        for name, pos in joint_points:
            if pos[2] < threshold:
                collision_count += 1
                if self.verbose_collisions:
                    print(
                        f"[RL Env] Table collision at **{name}**: z={pos[2]:.4f} < threshold={threshold:.4f} "
                        f"(table={table_height:.4f}, buffer={buffer:.4f})"
                    )

        if self.verbose_collisions and collision_count > 0:
            print(f"[RL Env] Total table-colliding joints: {collision_count}")

        return collision_count
    
    def _check_self_collision_totable_soft(self) -> int:
        """
        Count table-collision occurrences by checking each joint point against table height.

        Returns:
            int: number of joints that are below the table surface threshold
        """
        joint_names = ["joint3", "joint4", "joint5", "joint6"]
        joint_points = []  # list of (name, position)

        for name in joint_names:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            if site_id != -1:
                joint_points.append((name, self.data.site_xpos[site_id].copy()))
                continue

            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id != -1:
                joint_points.append((name, self.data.xpos[body_id].copy()))

        table_height = float(self.body_pos[2])  # table top z
        # buffer = -0.03  # 5cm above table surface counts as collision
        safe_distance = 0.15  # 15cm

        cost = 0

        for name, pos in joint_points:
            if pos[2] > table_height:
                if pos[2] - table_height < safe_distance:
                    x = (pos[2] - table_height) / safe_distance
                    cost += x ** 5
                    if self.verbose_collisions:
                        print(
                            f"[RL Env] SOFT Table collision at **{name}**: z={pos[2]:.4f} < threshold={safe_distance:.4f} "
                            f"(table={table_height:.4f})"
                        )
            else:
                cost += 1
                if self.verbose_collisions:
                    print(
                        f"[RL Env] DIRECT Table collision at **{name}**: z={pos[2]:.4f} < height={table_height:.4f} "
                    )

        if self.verbose_collisions and cost > 0:
            print(f"[RL Env] Total table-colliding cost: {cost:.4f}")

        return cost


    def render_sim(self, camera_name: Optional[str] = None) -> np.ndarray:
        """
        Get RGB frames from the simulation.
        
        Returns:
            np.ndarray: RGB frame
        """
        if not hasattr(self, 'renderer'):
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)

        if camera_name is not None:
            camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        else:
            camera_id = -1 # Use the free camera (same as viewer)

        if camera_id == -1:
            # Use the same setup as the viewer
            # Create a camera object to pass to update_scene
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.fixedcamid = -1
            cam.azimuth = self.azimuth
            cam.elevation = self.elevation
            cam.distance = self.distance
            cam.lookat[:] = np.array(self.lookat)
            
            self.renderer.update_scene(self.data, camera=cam)
        else:
            self.renderer.update_scene(self.data, camera=camera_id)
            
        return self.renderer.render()
    
    def close(self):
        """Clean up resources."""
        if self.robot:
            self.robot.shutdown()
        if self.sensor:
            self.sensor.stop()
        # if self.watchdog:
        #     self.watchdog.stop()
        
        if hasattr(self, 'renderer'):
            self.renderer.close()
            del self.renderer


class DummyEnv(gym.Env):
    """A dummy environment for testing purposes."""
    def __init__(self, action_size=6, action_range_deg=30.0, observation_size=12):
        super().__init__()
        action_range_rad = np.radians(action_range_deg)
        self.action_space = spaces.Box(
            low=-action_range_rad, high=action_range_rad, shape=(action_size,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(observation_size,), dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape), {} # SB3 v2.0+ reset returns (obs, info)

    def step(self, action):
        return np.zeros(self.observation_space.shape), 0, False, {}


class LynxSimPIDEnv(LynxSimEnv):
    """
    A subclass of LynxSimEnv that uses PID control for the Lynx robot.
    """
    def __init__(self, cfg: Any):
        super().__init__(cfg)

        # Additional initialization for PID controller can be added here
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment using MuJoCo's position control.
        """
        # Check for emergency state from watchdog (copied from base class)
        if self.robot._in_emergency_state:
            reward = -100.0 # Large negative reward
            print(f"[RL Env] Emergency state triggered! Severe penalty {reward} applied.")
            terminated = True
            truncated = False
            while self.robot._in_emergency_state:
                print("[RL Env] Waiting for robot to exit emergency state before reset...")
                time.sleep(0.2)
            
            observation, _, _ = self._get_observation()
            info = self._get_info(current_action=np.zeros(6), observation=observation)
            info["emergency_state"] = True
            return observation, reward, terminated, truncated, info
        
        current_action = np.zeros(6) # Default to zero if no action provided
        
        # Get current joint positions in degrees
        current_joint_pos_vel_deg = self.robot.get_PosVel()
        current_joint_pos_deg = np.array([pv[0] for pv in current_joint_pos_vel_deg])
        
        if action is not None:
            current_action = np.array(action, dtype=np.float32)

            # 1) action delay: buffer in controller steps
            if self._dr_action_delay_steps > 0:
                self._action_buffer.append(current_action.copy())
                current_action = self._action_buffer.pop(0)  # delayed action

            if self.dr_enable:
                current_action = current_action + self._dr_action_bias

            # 2) action noise (additive Gaussian), then clip to action_space
            if self.dr_enable and self._dr_params.get("action_noise_std", 0.0) > 0.0:
                std = float(self._dr_params["action_noise_std"])
                current_action = current_action + self._dr_rng.normal(0.0, std, size=current_action.shape).astype(np.float32)

            # clip to env action bounds (important after noise)
            current_action = np.clip(current_action, self.action_space.low, self.action_space.high).astype(np.float32)

            # ----- Calculate target joint positions in degrees based on the action -----
            scaled_action = current_action * self.action_scale # Action is a relative change in radians
            scaled_action_deg = np.degrees(scaled_action)
            target_joint_pos_deg = current_joint_pos_deg + scaled_action_deg
            
            # Clip the joint positions (copied from base class)
            assert len(target_joint_pos_deg) == len(self.cfg.RLConfig.JOINT_LIMITS)
            for j in range(len(target_joint_pos_deg)):
                target_joint_pos_deg[j] = np.clip(target_joint_pos_deg[j], self.cfg.RLConfig.JOINT_LIMITS[j][0]*self.clip_action_scale, self.cfg.RLConfig.JOINT_LIMITS[j][1]*self.clip_action_scale)
            
            # Clip the combined angle of joint 2 and 3 (copied from base class)
            if self.cfg.RLConfig.COMBINED_JOINT_2_3_LIMIT > 0:
                joint1, joint2 = self.joints_clip(target_joint_pos_deg[1], -target_joint_pos_deg[2], -self.cfg.RLConfig.COMBINED_JOINT_2_3_LIMIT, self.cfg.RLConfig.COMBINED_JOINT_2_3_LIMIT)
                target_joint_pos_deg[1] = joint1
                target_joint_pos_deg[2] = -joint2
        else:
            target_joint_pos_deg = current_joint_pos_deg

        # Step the simulation with several sim steps per action
        for k in range(self.simulation_steps_per_action):
            # Apply target position to MuJoCo actuators (position control)
            if k == 0:
                self.robot.move_abs(target_joint_pos_deg)
            
            # reset the simulation solver every 500 simulation steps (copied from base class)
            if self.current_step % 500 == 0:
                mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
                mujoco.mj_forward(self.model, self.data)
                mujoco.mj_collision(self.model, self.data)

            # Step the simulation
            mujoco.mj_step(self.model, self.data)

            if self.viewer: # Sync viewer if human render mode
                self.viewer.sync()
            
            if self.real_time_speed:
                time.sleep(self.timestep)  # Sleep to simulate real-time

            if k == self.sample_k:
                observation, optitrack_timestamp, proprioception_timestamp = self._get_observation()

        # Update joint_start_velocities for consistency
        current_joint_pos_vel_deg = self.robot.get_PosVel()
        self.joint_start_velocities = np.radians(np.array([pv[1] for pv in current_joint_pos_vel_deg]))

        # Reward and info
        reward, terminated = self._compute_reward(current_action, observation)
        info = self._get_info(current_action, observation)
        info["domain_rand"] = self._dr_params.copy()
        info["emergency_state"] = self.robot._in_emergency_state
        
        self.previous_action = current_action.copy()
        self.current_step += 1
        truncated = self.current_step >= self.max_episode_steps or terminated
        if truncated:
            self._episode_end_ee_distance = info["distance_to_target"]
        
        return observation, reward, terminated, truncated, info


class LynxSimPDInterpEnv(LynxSimEnv):
    """
    A subclass of LynxSimEnv that uses PD control with interpolated targets.
    """
    def __init__(self, cfg: Any):
        super().__init__(cfg)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment using MuJoCo's position control with interpolated targets.
        """
        # Check for emergency state from watchdog
        if self.robot._in_emergency_state:
            reward = -100.0
            terminated = True
            truncated = False
            observation, _, _ = self._get_observation()
            info = self._get_info(current_action=np.zeros(6), observation=observation)
            info["emergency_state"] = True
            return observation, reward, terminated, truncated, info
        
        current_action = np.zeros(6)
        current_joint_pos_vel_deg = self.robot.get_PosVel()
        current_joint_pos_deg = np.array([pv[0] for pv in current_joint_pos_vel_deg])
        current_joint_pos_rad = np.radians(current_joint_pos_deg).copy()
        
        if action is not None:
            current_action = np.array(action, dtype=np.float32)
            if self._dr_action_delay_steps > 0:
                self._action_buffer.append(current_action.copy())
                current_action = self._action_buffer.pop(0)
            if self.dr_enable:
                current_action = current_action + self._dr_action_bias
            if self.dr_enable and self._dr_params.get("action_noise_std", 0.0) > 0.0:
                std = float(self._dr_params["action_noise_std"])
                current_action = current_action + self._dr_rng.normal(0.0, std, size=current_action.shape).astype(np.float32)
            current_action = np.clip(current_action, self.action_space.low, self.action_space.high).astype(np.float32)

            scaled_action = current_action * self.action_scale
            scaled_action_deg = np.degrees(scaled_action)
            target_joint_pos_deg = current_joint_pos_deg + scaled_action_deg
            
            for j in range(len(target_joint_pos_deg)):
                target_joint_pos_deg[j] = np.clip(target_joint_pos_deg[j], self.cfg.RLConfig.JOINT_LIMITS[j][0]*self.clip_action_scale, self.cfg.RLConfig.JOINT_LIMITS[j][1]*self.clip_action_scale)
            
            if self.cfg.RLConfig.COMBINED_JOINT_2_3_LIMIT > 0:
                joint1, joint2 = self.joints_clip(target_joint_pos_deg[1], -target_joint_pos_deg[2], -self.cfg.RLConfig.COMBINED_JOINT_2_3_LIMIT, self.cfg.RLConfig.COMBINED_JOINT_2_3_LIMIT)
                target_joint_pos_deg[1] = joint1
                target_joint_pos_deg[2] = -joint2
            
            target_joint_pos_rad = np.radians(target_joint_pos_deg).copy()
            self.joint_start_positions = current_joint_pos_rad
            self.joint_target_positions = target_joint_pos_rad
            self._current_motion_sim_time = 0.0

            max_duration = 0.0
            for i in range(6):
                q_start = self.joint_start_positions[i]
                q_end = self.joint_target_positions[i]
                v_start_raw = self.joint_start_velocities[i]
                v_max = self.max_joint_velocity
                a_max = self.max_joint_acceleration
                delta_q = q_end - q_start
                direction = np.sign(delta_q) if delta_q != 0 else 1.0
                if delta_q != 0 and np.sign(v_start_raw) != direction:
                    v_start = 0.0
                else:
                    v_start = direction * min(abs(v_start_raw), v_max)
                if abs(delta_q) < 1e-6 and abs(v_start) < 1e-6:
                    duration = 0.0
                else:
                    Ta1 = (v_max - abs(v_start)) / a_max if v_max > abs(v_start) else 0.0
                    Sa1 = abs(v_start) * Ta1 + 0.5 * a_max * Ta1**2
                    Ta2 = v_max / a_max
                    Sa2 = 0.5 * a_max * Ta2**2
                    S_accel_decel = Sa1 + Sa2
                    if S_accel_decel * direction > abs(delta_q) * direction:
                        v_peak = np.sqrt((2 * a_max * abs(delta_q) + v_start**2) / 2)
                        v_peak = min(v_peak, v_max)
                        Ta_accel = (v_peak - abs(v_start)) / a_max
                        Ta_decel = v_peak / a_max
                        duration = Ta_accel + Ta_decel
                    else:
                        Tv = (abs(delta_q) - S_accel_decel) / v_max
                        duration = Ta1 + Tv + Ta2
                max_duration = max(max_duration, duration)
            self.motion_duration = max_duration
        else:
            target_joint_pos_deg = current_joint_pos_deg

        for k in range(self.simulation_steps_per_action):
            t_profile = min(self._current_motion_sim_time, self.motion_duration)
            desired_joint_pos_rad = np.zeros(6)
            desired_joint_vel_rad = np.zeros(6)
            for i in range(6):
                q_start = self.joint_start_positions[i]
                q_end = self.joint_target_positions[i]
                v_start = self.joint_start_velocities[i] * self.v_start_scale
                v_max = self.max_joint_velocity
                a_max = self.max_joint_acceleration
                pos, vel, _ = trapezoidal_velocity_profile_non_zero_start(q_start, q_end, v_start, v_max, a_max, t_profile)
                desired_joint_pos_rad[i] = pos
                desired_joint_vel_rad[i] = vel

            self.robot.move_abs(np.degrees(desired_joint_pos_rad))
            if self.current_step % 500 == 0:
                mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
                mujoco.mj_forward(self.model, self.data)
                mujoco.mj_collision(self.model, self.data)
            mujoco.mj_step(self.model, self.data)
            if self.viewer:
                self.viewer.sync()
            self._current_motion_sim_time += self.timestep
            if self.real_time_speed:
                time.sleep(self.timestep)
            if k == self.sample_k:
                observation, optitrack_timestamp, proprioception_timestamp = self._get_observation()

        self.joint_start_velocities = desired_joint_vel_rad
        reward, terminated = self._compute_reward(current_action, observation)
        info = self._get_info(current_action, observation)
        info["domain_rand"] = self._dr_params.copy()
        info["emergency_state"] = self.robot._in_emergency_state
        self.previous_action = current_action.copy()
        self.current_step += 1
        truncated = self.current_step >= self.max_episode_steps or terminated
        if truncated:
            self._episode_end_ee_distance = info["distance_to_target"]
        return observation, reward, terminated, truncated, info



