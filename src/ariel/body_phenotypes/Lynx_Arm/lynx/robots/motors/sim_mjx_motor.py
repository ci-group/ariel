import numpy as np
import jax.numpy as jp
import mujoco
from mujoco import mjx


class MjxSimMotor:
    """
    MuJoCo Jax motor control class.
    """
    def __init__(self, id, model: mjx.Model, data: mjx.Data, joint_qpos_id: int, joint_qvel_id: int):
        self._id = id
        self.model = model
        self.data = data
        self._joint_qpos_id = joint_qpos_id # Index in qpos array
        self._joint_qvel_id = joint_qvel_id # Index in qvel array (dofadr)

        self._target_position_rad = 0.0
        self._control_mode = "position" # "position" or "velocity" or "torque"
        self._current_target_ctrl = 0.0 # What's currently being written to data.ctrl

    def getPosition(self):
        return self.data.qpos[self._joint_qpos_id]  # radians
    
    def limp(self):
        # In Mujoco, "limp" can mean setting control to passive or applying zero torque/velocity.
        # For now, we'll set the control to zero velocity.
        self._control_mode = "velocity"
        self._current_target_ctrl = 0.0
        # Assuming direct velocity control via data.ctrl
        # This assumes the model is configured for velocity control on this joint
        # If not, this might need to be torque control with 0 torque.
        self.data.ctrl = self.data.ctrl.at[self._id].set(self._current_target_ctrl) # Assuming motor ID maps to ctrl index

    def hold(self):
        # In Mujoco, "hold" means maintaining the current position.
        # This can be achieved by setting the current position as the target for position control.
        self._control_mode = "position"
        current_pos_rad = self.data.qpos[self._joint_qpos_id]
        self._current_target_ctrl = current_pos_rad
        self.data.ctrl = self.data.ctrl.at[self._id].set(self._current_target_ctrl) # Assuming motor ID maps to ctrl index

    def move_abs_direct(self, pos_deg):
        target_pos_rad = jp.radians(pos_deg) # Convert degrees to radians
        return self.data.replace(qpos=self.data.qpos.at[self._joint_qpos_id].set(target_pos_rad))


    
