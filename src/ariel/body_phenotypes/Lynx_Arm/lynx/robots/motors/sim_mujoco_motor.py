import numpy as np
import mujoco

class MujocoSimMotor:
    """Motor control class for simulated motors in Mujoco."""
    
    def __init__(self, id, model: mujoco.MjModel, data: mujoco.MjData, joint_qpos_id: int, joint_qvel_id: int, bus_lock):
        self._id = id
        self.model = model
        self.data = data
        self._joint_qpos_id = joint_qpos_id # Index in qpos array
        self._joint_qvel_id = joint_qvel_id # Index in qvel array (dofadr)
        self._bus_lock = bus_lock # Not strictly necessary for Mujoco, but keeping for API consistency

        self._target_position_rad = 0.0
        self._control_mode = "position" # "position" or "velocity" or "torque"
        self._current_target_ctrl = 0.0 # What's currently being written to data.ctrl

    def getPosVel(self):
        with self._bus_lock:
            joint_angle_rad = self.data.qpos[self._joint_qpos_id]
            joint_velocity_rad_s = self.data.qvel[self._joint_qvel_id]
        return np.degrees(joint_angle_rad), np.degrees(joint_velocity_rad_s)

    def genericRead_Blocking_int(self, cmd):
        if cmd == "QD":
            pos_deg, _ = self.getPosVel()
            return int(pos_deg * 100) # Return in centi-degrees
        return None

    def genericWrite(self, cmd, param=None):
        pass

    def reset(self):
        pass

    def limp(self):
        # In Mujoco, "limp" can mean setting control to passive or applying zero torque/velocity.
        # For now, we'll set the control to zero velocity.
        with self._bus_lock:
            self._control_mode = "velocity"
            self._current_target_ctrl = 0.0
            # Assuming direct velocity control via data.ctrl
            # This assumes the model is configured for velocity control on this joint
            # If not, this might need to be torque control with 0 torque.
            self.data.ctrl[self._id] = self._current_target_ctrl # Assuming motor ID maps to ctrl index

    def hold(self):
        # In Mujoco, "hold" means maintaining the current position.
        # This can be achieved by setting the current position as the target for position control.
        with self._bus_lock:
            self._control_mode = "position"
            current_pos_rad = self.data.qpos[self._joint_qpos_id]
            self._current_target_ctrl = current_pos_rad
            self.data.ctrl[self._id] = self._current_target_ctrl # Assuming motor ID maps to ctrl index

    def move_abs(self, pos_deg):
        with self._bus_lock:
            self._control_mode = "position"
            target_pos_rad = np.radians(pos_deg) # Convert degrees to radians
            self._target_position_rad = target_pos_rad
            self._current_target_ctrl = target_pos_rad
            self.data.ctrl[self._id] = self._current_target_ctrl # Assuming motor ID maps to ctrl index

    def move_abs_direct(self, pos_deg):
        with self._bus_lock:
            self._control_mode = "position"
            target_pos_rad = np.radians(pos_deg) # Convert degrees to radians
            self._target_position_rad = target_pos_rad
            self._current_target_ctrl = target_pos_rad
            self.data.qpos[self._joint_qpos_id] = self._current_target_ctrl # Assuming motor ID maps to ctrl index

    def move_abs_with_speed(self, pos_deg, speed):
        # In Mujoco, controlling with a specific speed usually means velocity control.
        # However, the original SimMotor uses POSITION_CONTROL with maxVelocity.
        # We'll interpret this as setting a target position, and relying on Mujoco's
        # internal PID or a custom controller to reach it with a certain speed.
        # For simplicity, we'll just set the target position for now, as direct speed
        # control to a target position is more complex in Mujoco without a higher-level controller.
        # The 'speed' parameter might be used to adjust a gain if a PID is active.
        with self._bus_lock:
            self._control_mode = "position"
            target_pos_rad = np.radians(pos_deg) # Convert degrees to radians
            self._target_position_rad = target_pos_rad
            self._current_target_ctrl = target_pos_rad
            # print(f"[MujocoSimMotor] Moving motor {self._id} to {self._current_target_ctrl} rad with speed {speed}.")
            self.data.ctrl[self._id] = self._current_target_ctrl # Assuming motor ID maps to ctrl index
            # print(f"[MujocoSimMotor] Commanded motor {self._id} to position {pos_deg} deg (target {self._current_target_ctrl:.3f} rad) with speed {speed}.")
            # Note: 'speed' parameter is not directly used here.
            # If fine-grained speed control is needed, a custom velocity controller
            # would be required that sets data.ctrl based on error and speed.

    def getPosition(self):
        return self.genericRead_Blocking_int("QD")
    
    def getPositionDirect(self):
        return np.degrees(self.data.qpos[self._joint_qpos_id])
