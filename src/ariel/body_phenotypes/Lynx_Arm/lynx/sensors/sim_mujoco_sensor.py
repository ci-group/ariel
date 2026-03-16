import numpy as np
import threading
import time
import mujoco

class MujocoSimSensor:
    """Simulated sensor interface for Mujoco simulation."""
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, end_effector_site_id: int, verbose=False):
        self.model = model
        self.data = data
        self.end_effector_site_id = end_effector_site_id
        self.verbose = verbose
        
        # Data handler for simulated sensor data
        self.data_handler = MujocoSimNatNetDataHandler(
            model=model,
            data=data,
            end_effector_site_id=end_effector_site_id,
            verbose=verbose
        )
        
        self.is_running = False

    def start(self):
        """Start the simulated sensor."""
        if self.is_running:
            print("[MujocoSimSensor] Sensor is already running.")
            return
            
        print("[MujocoSimSensor] Starting simulated sensor...")
        self.is_running = True
        # For simulation, we just mark as running - data is updated on demand

    def stop(self):
        """Stop the simulated sensor."""
        print("[MujocoSimSensor] Stopping simulated sensor...")
        self.is_running = False

    def get_relative_pose(self):
        """Get the latest relative pose of end-effector."""
        if self.is_running:
            self.data_handler.update_data()
        with self.data_handler._data_lock:
            return self.data_handler.latest_relative_pos, self.data_handler.latest_relative_quat

    def is_connected(self):
        """Check if the sensor is connected (always True for simulation)."""
        return self.is_running

    def data_available(self):
        """Check if data is available (always True for simulation when running)."""
        return self.is_running


class MujocoSimNatNetDataHandler:
    """Data handler for simulated sensor data from Mujoco."""
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, end_effector_site_id: int, verbose=False):
        self.model = model
        self.data = data
        self._end_effector_site_id = end_effector_site_id
        self._verbose = verbose

        self.latest_relative_pos = np.zeros(3, dtype=np.float32)
        self.latest_relative_quat = np.array([0., 0., 0., 1.], dtype=np.float32)
        self._data_lock = threading.Lock()

        # Initial call to update_data
        self.update_data()

    def update_data(self):
        """Update the end-effector pose data from Mujoco."""
        if self._verbose:
            print(f"[MujocoSimSensor] Updating EEF data...")
            
        # Ensure forward kinematics are up-to-date before reading site positions
        # This might be redundant if the environment steps the simulation before calling this.
        # mujoco.mj_forward(self.model, self.data) 
        
        # Extract position from site_xpos
        eef_position = self.data.site_xpos[self._end_effector_site_id].copy()
        
        # Extract orientation from site_xmat (rotation matrix) and convert to quaternion
        eef_rotation_matrix = self.data.site_xmat[self._end_effector_site_id].reshape(3, 3)
        eef_orientation = np.zeros(4)
        mujoco.mju_mat2Quat(eef_orientation, eef_rotation_matrix.flatten()) # Convert 3x3 matrix to quaternion (w,x,y,z)

        self.latest_relative_pos = eef_position
        self.latest_relative_quat = eef_orientation

        if self._verbose:
            print(f"[MujocoSimSensor] EEF Pos: {eef_position}, EEF Quat: {eef_orientation}")


class MujocoSimNatNetClient:
    """Placeholder client class for compatibility."""
    
    def __init__(self):
        pass


def run_mujoco_sim_sensor_client(natnet_client):
    """Placeholder function for compatibility."""
    pass
