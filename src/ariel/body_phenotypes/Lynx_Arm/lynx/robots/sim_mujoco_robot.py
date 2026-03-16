import os
import time
import threading
import numpy as np
import mujoco
import mujoco.viewer

from .motors.motor_cmd import LSS_BroadcastID
from .motors.sim_mujoco_motor import MujocoSimMotor


# Mimic motor_cmd constants needed by Robot class.
class LSSConstants:
    LSS_BroadcastID = 254

lssc = LSSConstants()

class MujocoSimRobot:
    """Simulated Lynx robot controller using Mujoco."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, render_mode="human", emergency_time=3.0):
        self.render_mode = render_mode
        self.model = model
        self.data = data
        self.viewer = None # For human rendering

        self._bus = True
        self._bus_lock = threading.Lock()  # This lock is specifically for motor access
        self._motor_ids = [0, 1, 2, 3, 4, 5] # Assuming 6 motors for a 6-DOF arm
        self._motors = []
        self.home_position = [0, 0, 0, 0, 0, 0] # Degrees
        self._controlled_joint_qpos_ids = [] # Mujoco qpos indices for controlled joints
        self._controlled_joint_qvel_ids = [] # Mujoco qvel indices for controlled joints
        self._end_effector_site_id = -1
        self._in_emergency_state = False
        self._emergency_thread = None
        self._emergency_time = emergency_time  # Time in seconds for emergency recovery

        self._current_joint_angles_deg = [0.0] * len(self._motor_ids)
        self._last_action = np.zeros(len(self._motor_ids), dtype=np.float32)  # Last action taken

        if not self.init_bus():
            raise RuntimeError("Failed to initialize simulated robot in Mujoco. Check model configuration.")

    def init_bus(self):
        print("[MujocoSimRobot] Initializing Mujoco robot...")
        try:
            # Identify controlled joints (assuming they are named joint1_joint to joint6_joint)
            controlled_joint_names = ["joint1_joint", "joint2_joint", "joint3_joint", "joint4_joint", "joint5_joint", "joint6_joint"]
            # controlled_joint_names = ["robot_joint1_joint1_joint", 
            #                             "robot_joint2_joint2_joint", 
            #                             "robot_joint3_joint3_joint", 
            #                             "robot_joint4_joint4_joint", 
            #                             "robot_joint5_joint5_joint", 
            #                             "robot_joint6_joint6_joint",
            #                         ]
            
            for joint_name in controlled_joint_names:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id == -1:
                    raise ValueError(f"Joint '{joint_name}' not found in the model.")
                # In Mujoco, qpos and qvel indices are often the same as joint IDs for simple models
                # For more complex models, one might need to map from joint_id to qpos/qvel address
                self._controlled_joint_qpos_ids.append(self.model.jnt_qposadr[joint_id])
                self._controlled_joint_qvel_ids.append(self.model.jnt_dofadr[joint_id])

            if len(self._controlled_joint_qpos_ids) != 6:
                print(f"[MujocoSimRobot] Warning: Expected 6 controlled joints, found {len(self._controlled_joint_qpos_ids)}.")

            # Identify end-effector site
            try:
                self._end_effector_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
                if self._end_effector_site_id == -1:
                    raise ValueError("Site 'end_effector' not found.")
                print(f"[MujocoSimRobot] Identified End-Effector Site: end_effector (Mujoco Site ID: {self._end_effector_site_id})")
            except ValueError as e:
                print(f"[MujocoSimRobot] Warning: Could not find specific end-effector site: {e}. Using default fallback (last controlled joint's body).")
                # Fallback: use the body associated with the last controlled joint
                if self._controlled_joint_qpos_ids:
                    last_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, controlled_joint_names[-1])
                    self._end_effector_site_id = self.model.body_jntadr[last_joint_id] # This is not a site, but a body ID. Need to clarify usage.
                    # For now, if site not found, we'll just use the last joint's body as a proxy for EE position.
                    # This needs to be handled carefully in get_info.
                else:
                    self._end_effector_site_id = -1 # No EE found

            print(f"[MujocoSimRobot] Controlled Mujoco Joint Qpos IDs (mapped to LSS IDs 0-5): {self._controlled_joint_qpos_ids}")
            
            # Initial forward pass to update data
            mujoco.mj_forward(self.model, self.data)
            
            return True
        except Exception as e:
            print(f"[MujocoSimRobot] Error during robot initialization: {e}")
            self._bus = False
            return False

    def close_bus(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        print("[MujocoSimRobot] Disconnected from Mujoco viewer.")
        self._bus = False

    def init_motors(self):
        if not self._bus:
            print("[MujocoSimRobot] Bus not initialized, cannot initialize motors.")
            return None
        self._motors = []
        for i, qpos_id in enumerate(self._controlled_joint_qpos_ids):
            # SimMotor needs to be adapted or a new MujocoSimMotor created
            m = MujocoSimMotor(id=self._motor_ids[i], model=self.model, data=self.data,
                               joint_qpos_id=qpos_id, joint_qvel_id=self._controlled_joint_qvel_ids[i],
                               bus_lock=self._bus_lock)
            self._motors.append(m)
        print(f"[MujocoSimRobot] Initialized {len(self._motors)} simulated motors.")

    def get_Position(self):
        pos_all_raw = []
        for m in self._motors:
            raw = m.getPosition() # This will read from self.data.qpos
            if raw is None:
                pos_all_raw.append(None)
                continue
            deg = float(raw) / 100.0 # Assuming SimMotor returns centi-degrees
            pos_all_raw.append(deg)

        valid_positions = [p for p in pos_all_raw if p is not None]
        if len(valid_positions) == len(self._motor_ids):
            self._current_joint_angles_deg = valid_positions
        else:
            print(
                "[MujocoSimRobot] Warning: Not all joint positions could be read. _current_joint_angles_deg not updated this cycle.")

        return [[p] if p is not None else [None] for p in pos_all_raw]

    def get_PosVel(self):
        pos_vel_all = []
        for m in self._motors:
            pos_deg, vel_deg_s = m.getPosVel()  # in degrees
            pos_vel_all.append([pos_deg, vel_deg_s])

        valid_positions = [p[0] for p in pos_vel_all if p[0] is not None]
        if len(valid_positions) == len(self._motor_ids):
            self._current_joint_angles_deg = valid_positions

        return pos_vel_all

    def move_abs_direct(self, action_degrees):
        if self._in_emergency_state:
            print("[MujocoSimRobot] In emergency state, ignoring move_abs_direct() command.")
            return

        action_degrees = list(action_degrees)
        assert len(action_degrees) == len(self._motors)

        for idx, action_degree in enumerate(action_degrees):
            self._motors[idx].move_abs_direct(action_degree)

        self._last_action = np.array(action_degrees, dtype=np.float32) * 100

    def move_abs(self, action_degrees):
        if self._in_emergency_state:
            print("[MujocoSimRobot] In emergency state, ignoring move_abs() command.")
            return

        action_degrees = list(action_degrees)
        assert len(action_degrees) == len(self._motors)

        for idx, action_degree in enumerate(action_degrees):
            # target_centi_deg = int(target_deg * 100)
            self._motors[idx].move_abs(action_degree) # This will set self.data.ctrl

    def move_abs_admin(self, action_degrees):
        # Skip the emergency state check for admin commands
        action_degrees = list(action_degrees)
        assert len(action_degrees) == len(self._motors)

        for idx, action_degree in enumerate(action_degrees):
            # target_centi_deg = int(target_deg * 100)
            self._motors[idx].move_abs(action_degree)

    def move_abs_with_speed(self, action_degrees, speed):
        if self._in_emergency_state:
            print("[MujocoSimRobot] In emergency state, ignoring move_abs_with_speed() command.")
            return

        # DIAGNOSTIC LOG: Check type and shape of 'action_degrees' right before the error
        # print(f"[MujocoSimRobot.move_abs_with_speed] Received action_degrees type: {type(action_degrees)}, shape: {action_degrees.shape if hasattr(action_degrees, 'shape') else 'N/A'}")
        action_degrees = list(action_degrees)
        assert len(action_degrees) == len(self._motors)

        for idx, action_degree in enumerate(action_degrees):
            self._motors[idx].move_abs_with_speed(action_degree, speed)

    def move_abs_with_speed_admin(self, action_degrees, speed):
        # Skip the emergency state check for admin commands
        action_degrees = list(action_degrees)
        assert len(action_degrees) == len(self._motors)

        for idx, action_degree in enumerate(action_degrees):
            # target_centi_deg = int(target_deg * 100)
            self._motors[idx].move_abs_with_speed(action_degree, speed)

    def move_rel(self, joint_deltas_degrees, speed=None):
        """
        Controls the robot's joints by moving them relative to their current positions.

        Args:
            joint_deltas_degrees (list or np.ndarray): A list or numpy array of
                                                      delta angles (in degrees) for each joint.
            speed (float, optional): The speed at which to move the joints. If None,
                                     uses the default speed of move_abs.
        """
        if self._in_emergency_state:
            print("[MujocoSimRobot] In emergency state, ignoring control_relative_movement() command.")
            return

        current_positions = self.get_Position()
        # get_Position returns [[deg1], [deg2], ...] so flatten it
        current_positions_flat = [p[0] for p in current_positions if p[0] is not None]

        if len(current_positions_flat) != len(joint_deltas_degrees):
            print("[MujocoSimRobot] Error: Mismatch in number of joint deltas and current positions.")
            return

        target_positions_degrees = np.array(current_positions_flat) + np.array(joint_deltas_degrees)

        if speed is not None:
            self.move_abs_with_speed(target_positions_degrees, speed)
        else:
            self.move_abs(target_positions_degrees)

    def limp(self):
        for m in self._motors:
            m.limp()
        print("[MujocoSimRobot] All simulated motors in limp mode.")

    def hold(self):
        for m in self._motors:
            m.hold()
        print("[MujocoSimRobot] All simulated motors in hold mode.")

    def limp_broadcast(self):
        self.limp()

    def hold_broadcast(self):
        self.hold()

    def enter_emergency_recovery(self):
        if self._in_emergency_state:
            return

        self._in_emergency_state = True
        self._emergency_recovery_task()

    def _emergency_recovery_task(self):
        print("\n" + "=" * 40)
        print("! ! ! MUJOCO SIM ROBOT ENTERING EMERGENCY RECOVERY ! ! !")
        print("[Emergency] Holding all joints...")
        self.hold()
        time.sleep(self._emergency_time)  # no simulation steps during the hold period

        print(f"[Emergency] Moving to safe home position: {self.home_position}")
        self._move_and_wait_admin(self.home_position, sim_steps_per_check=50)

        print("[Emergency] _move_and_wait in recovery finished (may have timed out).")
        self._in_emergency_state = False
        print("[Emergency] Home position reached. Resuming normal operation.")
        print("=" * 40 + "\n")
        print("[Emergency] Recovery task completed (`_in_emergency_state` is now False).")

    def _move_and_wait(self, target_position_degrees, tolerance=2.0, sim_steps_per_check=20):
        self.move_abs(target_position_degrees)

        timeout_steps = 1000
        current_sim_steps = 0

        while not self._is_at_target(target_position_degrees, tolerance=tolerance):
            mujoco.mj_step(self.model, self.data) # Step Mujoco simulation
            if self.viewer:
                self.viewer.sync()
            current_sim_steps += 1
            if current_sim_steps >= timeout_steps:
                print(f"[MujocoSimRobot] Warning: _move_and_wait timed out after {timeout_steps} steps before reaching target.")
                break
            if current_sim_steps % sim_steps_per_check == 0:
                time.sleep(0.001)

    def _move_and_wait_admin(self, target_position_degrees, tolerance=2.0, sim_steps_per_check=20):
        self.move_abs_with_speed_admin(target_position_degrees, speed=0.1)

        timeout_steps = 1e12
        current_sim_steps = 0

        while not self._is_at_target(target_position_degrees, tolerance=tolerance):
            mujoco.mj_step(self.model, self.data) # Step Mujoco simulation
            if self.viewer:
                self.viewer.sync()
            current_sim_steps += 1
            if current_sim_steps >= timeout_steps:
                print(f"[MujocoSimRobot] Warning: _move_and_wait timed out after {timeout_steps} steps before reaching target.")
                break
            if current_sim_steps % sim_steps_per_check == 0:
                time.sleep(0.001)

    def _is_at_target(self, target_position_degrees, tolerance=2.0):
        current_pos_list = self.get_Position()
        current_pos = [p[0] for p in current_pos_list if p[0] is not None]

        if len(current_pos) != len(target_position_degrees):
            print(
                f"[MujocoSimRobot._is_at_target] Mismatch in joint count. Current: {len(current_pos)}, Target: {len(target_position_degrees)}")
            return False

        max_diff = 0.0
        for i in range(len(target_position_degrees)):
            diff = abs(current_pos[i] - target_position_degrees[i])
            max_diff = max(max_diff, diff)
            if diff > tolerance:
                return False
        return True

    def shutdown(self):
        print("[MujocoSimRobot] Shutting down simulation...")
        if self._emergency_thread and self._emergency_thread.is_alive():
            print("[MujocoSimRobot] Waiting for active emergency recovery to complete before shutting down Mujoco...")
            self._emergency_thread.join()
            self._in_emergency_state = False
        self.close_bus()
        print("[MujocoSimRobot] Shutdown complete.")

    def get_info(self):
        # Returns the Mujoco model, data, and end-effector site ID
        return self.model, self.data, self._end_effector_site_id
