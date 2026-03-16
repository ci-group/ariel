import os
import time
import threading
import mujoco
import mujoco.viewer

import jax
import jax.numpy as jp
from mujoco import mjx
from typing import Optional, Union


class MjxSimRobot:
    """
    MuJoCo Jax simulated robot class.
    """
    def __init__(self, mj_model: mujoco.MjModel, model: mjx.Model, data: mjx.Data, emergency_time=3.0):
        self._mj_model = mj_model # Store the original Mujoco model for name-to-id lookups
        self.model = model
        self.data = data # mjx.Data object. Note: MJX data is immutable, must be replaced.
        self.home_position = [0, 0, 0, 0, 0, 0] # Degrees
        self._controlled_joint_qpos_ids = [] # Mujoco qpos indices for controlled joints
        self._controlled_joint_qvel_ids = [] # Mujoco qvel indices for controlled joints
        self._end_effector_site_id = -1
        # self._in_emergency_state = False
            # self._emergency_thread = None
            # self._emergency_time = emergency_time

        self._current_joint_angles_deg = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # Current joint angles in degrees
        self._last_action = jp.zeros(len(self.home_position), dtype=float)  # Last action taken

        # Initialize the robot
        self.post_init()

    def post_init(self):
        # Identify controlled joints (assuming they are named joint1_joint to joint6_joint)
        controlled_joint_names = ["joint1_joint", "joint2_joint", "joint3_joint", "joint4_joint", "joint5_joint", "joint6_joint"]
        
        for joint_name in controlled_joint_names:
            joint_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id == -1:
                print(f"[MjxRobot] Warning: Joint '{joint_name}' not found in the model.")
                continue
            # In Mujoco, qpos and qvel indices are often the same as joint IDs for simple models
            # For more complex models, one might need to map from joint_id to qpos/qvel address
            self._controlled_joint_qpos_ids.append(self.model.jnt_qposadr[joint_id])
            self._controlled_joint_qvel_ids.append(self.model.jnt_dofadr[joint_id])
            print(f"[MjxRobot] Joint '{joint_name}' found at ID {joint_id}, qposadr {self.model.jnt_qposadr[joint_id]}")

        if len(self._controlled_joint_qpos_ids) != 6:
            print(f"[MjxRobot] Warning: Expected 6 controlled joints, found {len(self._controlled_joint_qpos_ids)}.")

        # Identify end-effector site
        self._end_effector_site_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
        if self._end_effector_site_id == -1:
            raise ValueError("Site 'end_effector' not found.")
        print(f"[MjxRobot] Identified End-Effector Site: end_effector (Mujoco Site ID: {self._end_effector_site_id})")

        # self._controlled_joint_qpos_ids = jp.array(self._controlled_joint_qpos_ids)
        ids = tuple(int(x) for x in self._controlled_joint_qpos_ids)
        self._controlled_joint_qpos_ids = jp.array(ids, dtype=jp.int32)

        ids_vel = tuple(int(x) for x in self._controlled_joint_qvel_ids)
        self._controlled_joint_qvel_ids = jp.array(ids_vel, dtype=jp.int32)

        print(f"[MjxRobot] Controlled Mujoco Joint Qpos IDs (mapped to LSS IDs 0-5): {self._controlled_joint_qpos_ids}")

        # Initial forward pass to populate data
        self.data = mjx.forward(self.model, self.data)

    # def qpos6(self, data: mjx.Data) -> jax.Array:
    #     return data.qpos[self._controlled_joint_qpos_ids]
    
    # def set_qpos6(self, data: mjx.Data, qpos6: jax.Array) -> mjx.Data:
    #     return data.replace(qpos=data.qpos.at[self._controlled_joint_qpos_ids].set(qpos6))

    def get_Position(self, data: mjx.Data):
        # Read joint positions directly from mjx.Data.qpos
        # Convert radians to degrees for consistency with previous API
        current_qpos_rad = data.qpos[self._controlled_joint_qpos_ids]
        current_qpos_deg = jp.degrees(current_qpos_rad)
        
        # Return as a JAX array of shape (N, 1) for JIT compatibility.
        return current_qpos_deg.reshape(-1, 1)

    def get_Position_rad(self, data: mjx.Data) -> jax.Array:
        return data.qpos[self._controlled_joint_qpos_ids]

    def get_motor(self, index):
        raise NotImplementedError("Direct motor access is no longer supported. Use MjxSimRobot methods.")

    def get_motor_count(self):
        return len(self._controlled_joint_qpos_ids)

    # def clear_emergency(self):
    #     print("[MjxRobot] Emergency state cleared.")
    #     self._in_emergency_state = False

    def move_abs_direct(self, data: mjx.Data, action_degrees: Union[jp.ndarray, list]):
        # JIT-compatible implementation: takes data, returns new data.
        action_degrees = jp.array(action_degrees)
        target_pos_rad = jp.radians(action_degrees)
        
        new_data = data.replace(qpos=data.qpos.at[self._controlled_joint_qpos_ids].set(target_pos_rad))
        
        return new_data

    def move_abs_rad(self, data: mjx.Data, action_rad):
        # action_rad: jax.Array shape (6,)
        return data.replace(
            qpos=data.qpos.at[self._controlled_joint_qpos_ids].set(action_rad)
        )



    # def enter_emergency_recovery(self):
    #     if self._in_emergency_state:
    #         return

    #     self._in_emergency_state = True
    #     self._emergency_recovery_task()

    # def _emergency_recovery_task(self):
    #     print("\n" + "=" * 40)
    #     print("! ! ! MUJOCO SIM ROBOT ENTERING EMERGENCY RECOVERY ! ! !")
    #     print("[Emergency] Holding all joints...")
    #     self.hold()
    #     time.sleep(self._emergency_time)  # no simulation steps during the hold period

    #     print(f"[Emergency] Moving to safe home position: {self.home_position}")
    #     self._move_and_wait(self.home_position, sim_steps_per_check=50)

    #     print("[Emergency] _move_and_wait in recovery finished (may have timed out).")
    #     self._in_emergency_state = False
    #     print("[Emergency] Home position reached. Resuming normal operation.")
    #     print("=" * 40 + "\n")
    #     print("[Emergency] Recovery task completed (`_in_emergency_state` is now False).")

    def _move_and_wait(self, target_position_degrees, tolerance=2.0, sim_steps_per_check=20):
        self.move_abs_direct(target_position_degrees)

        timeout_steps = 1000
        current_sim_steps = 0

        while not self._is_at_target(target_position_degrees, tolerance=tolerance):
            self.data = mjx.forward(self.model, self.data) # Step Mujoco simulation
            current_sim_steps += 1
            if current_sim_steps >= timeout_steps:
                print(f"[MujocoSimRobot] Warning: _move_and_wait timed out after {timeout_steps} steps before reaching target.")
                break
            if current_sim_steps % sim_steps_per_check == 0:
                time.sleep(0.001)

    def _is_at_target(self, target_position_degrees, tolerance=2.0):
        current_pos_deg = self.get_Position().flatten()
        target_position_degrees = jp.array(target_position_degrees)

        if len(current_pos_deg) != len(target_position_degrees):
            print(
                f"[MujocoSimRobot._is_at_target] Mismatch in joint count. Current: {len(current_pos_deg)}, Target: {len(target_position_degrees)}")
            return False

        max_diff = jp.max(jp.abs(current_pos_deg - target_position_degrees))
        return max_diff <= tolerance

    def hold(self):
        # In MJX, 'hold' could mean setting velocity to zero and keeping current position
        # For now, we just keep the current position.
        current_pos_deg = self.get_Position().flatten()
        self.move_abs_direct(current_pos_deg)
        print("[MjxRobot] Holding current position.")
