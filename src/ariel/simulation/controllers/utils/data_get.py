import mujoco as mj
import numpy as np

def get_state_from_data(data:mj.MjData):
    """
    Extracts the robot state EXCLUDING global position.
    Processes quaternion to be consistent (scaled by sign of w).

    Parameters
    ----------

    data : mujoco.MjData
        Data variable of mujoco containing global x, y, z position, x, y, z quaternion orientation, and all hinge positions

    Returns
    -------
    np.array
        Array of quaternion orientation and hinge positions 
    
    """

    # 1. Get Quaternion (w, x, y, z) - Index 3 to 7
    quat = data.qpos[3:7].copy()
    
    # 2. Scale/Normalize Quaternion
    # If w is negative, negate the whole quaternion.
    if quat[0] < 0:
        quat = -quat
        
    # 3. Use only the Imaginary parts (x, y, z)
    quat_imag = quat[1:] 
    
    # 4. Get Hinge Joints (Index 7 onwards)
    joints = data.qpos[7:]
    
    return np.concatenate([quat_imag, joints])