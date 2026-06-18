import numpy as np
import math

# ============================================================================
# COORDINATE SYSTEM CONVERSIONS
# ============================================================================

def cartesian_to_spherical(x, y=None, z=None):
    """
    Convert Cartesian coordinates to spherical coordinates.
    
    Args:
        x: Either x-coordinate (scalar) or array of coordinates
        y: y-coordinate (if x is scalar)
        z: z-coordinate (if x is scalar)
        
    Array formats:
        - 1D array: [x, y, z]
        - 2D array: [[x1,y1,z1], [x2,y2,z2], ...] shape (N, 3)
        
    Returns:
        tuple or numpy.ndarray: 
        - If scalar input: (r, theta, phi)
        - If array input: array of shape (N, 3) with columns [r, theta, phi]
    """
    # Handle array inputs
    if y is None and z is None:
        coords = np.asarray(x)
        if coords.ndim == 1:
            # 1D array case
            x, y, z = coords[0], coords[1], coords[2]
        elif coords.ndim == 2:
            # 2D array case - vectorized computation
            x_arr, y_arr, z_arr = coords[:, 0], coords[:, 1], coords[:, 2]
            r = np.sqrt(x_arr**2 + y_arr**2 + z_arr**2)
            theta = np.arctan2(y_arr, x_arr)
            # Keep theta in [-π, π] range (arctan2 already provides this)
            phi = np.where(r > 0, np.arccos(z_arr / r), 0)
            return np.column_stack([r, theta, phi])
        else:
            raise ValueError("Input array must be 1D or 2D")
    
    # Scalar computation
    r = math.sqrt(x**2 + y**2 + z**2)
    theta = math.atan2(y, x)
    # Keep theta in [-π, π] range (atan2 already provides this)
    phi = math.acos(z / r) if r > 0 else 0
    return r, theta, phi

def spherical_to_cartesian(r, theta=None, phi=None):
    """
    Convert spherical coordinates to Cartesian coordinates.
    
    Args:
        r: Either radial distance (scalar) or array of coordinates
        theta: azimuthal angle in radians (if r is scalar)
        phi: polar angle in radians (if r is scalar)
        
    Array formats:
        - 1D array: [r, theta, phi]
        - 2D array: [[r1,theta1,phi1], [r2,theta2,phi2], ...] shape (N, 3)
        
    Returns:
        tuple or numpy.ndarray:
        - If scalar input: (x, y, z)
        - If array input: array of shape (N, 3) with columns [x, y, z]
    """
    # Handle array inputs
    if theta is None and phi is None:
        coords = np.asarray(r)
        if coords.ndim == 1:
            # 1D array case
            r, theta, phi = coords[0], coords[1], coords[2]
        elif coords.ndim == 2:
            # 2D array case - vectorized computation
            r_arr, theta_arr, phi_arr = coords[:, 0], coords[:, 1], coords[:, 2]
            x = r_arr * np.sin(phi_arr) * np.cos(theta_arr)
            y = r_arr * np.sin(phi_arr) * np.sin(theta_arr)
            z = r_arr * np.cos(phi_arr)
            return np.column_stack([x, y, z])
        else:
            raise ValueError("Input array must be 1D or 2D")
    
    # Scalar computation
    x = r * math.sin(phi) * math.cos(theta)
    y = r * math.sin(phi) * math.sin(theta)
    z = r * math.cos(phi)
    return x, y, z

# ============================================================================
# ROTATION CONVERSIONS
# ============================================================================

def spherical_to_quaternion(theta, phi=None):
    """
    Convert spherical coordinates to quaternion (assuming theta is yaw, phi is pitch).
    
    Args:
        theta: Either azimuthal angle (scalar) or array of coordinates
        phi: polar angle in radians (if theta is scalar)
        
    Array formats:
        - 1D array: [theta, phi]
        - 2D array: [[theta1,phi1], [theta2,phi2], ...] shape (N, 2)
        
    Returns:
        tuple or numpy.ndarray:
        - If scalar input: (w, x, y, z)
        - If array input: array of shape (N, 4) with columns [w, x, y, z]
    """
    # Handle array inputs
    if phi is None:
        coords = np.asarray(theta)
        if coords.ndim == 1:
            # 1D array case
            theta, phi = coords[0], coords[1]
        elif coords.ndim == 2:
            # 2D array case - vectorized computation
            theta_arr, phi_arr = coords[:, 0], coords[:, 1]
            cy = np.cos(theta_arr * 0.5)
            sy = np.sin(theta_arr * 0.5)
            cp = np.cos(phi_arr * 0.5)
            sp = np.sin(phi_arr * 0.5)
            
            w = cy * cp
            x = cy * sp
            y = sy * cp
            z = sy * sp
            
            return np.column_stack([w, x, y, z])
        else:
            raise ValueError("Input array must be 1D or 2D")
    
    # Scalar computation
    cy = math.cos(theta * 0.5)
    sy = math.sin(theta * 0.5)
    cp = math.cos(phi * 0.5)
    sp = math.sin(phi * 0.5)
    
    w = cy * cp
    x = cy * sp
    y = sy * cp
    z = sy * sp
    
    return w, x, y, z

def quaternion_to_spherical(w, x=None, y=None, z=None):
    """
    Convert quaternion to spherical coordinates (assuming quaternion represents yaw and pitch).
    
    Args:
        w: Either w component (scalar) or array of quaternions
        x, y, z: quaternion components (if w is scalar)
        
    Array formats:
        - 1D array: [w, x, y, z]
        - 2D array: [[w1,x1,y1,z1], [w2,x2,y2,z2], ...] shape (N, 4)
        
    Returns:
        tuple or numpy.ndarray:
        - If scalar input: (theta, phi)
        - If array input: array of shape (N, 2) with columns [theta, phi]
    """
    # Handle array inputs
    if x is None and y is None and z is None:
        coords = np.asarray(w)
        if coords.ndim == 1:
            # 1D array case
            w, x, y, z = coords[0], coords[1], coords[2], coords[3]
        elif coords.ndim == 2:
            # 2D array case - vectorized computation
            w_arr, x_arr, y_arr, z_arr = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
            # Normalize quaternions
            norm = np.sqrt(w_arr**2 + x_arr**2 + y_arr**2 + z_arr**2)
            w_arr, x_arr, y_arr, z_arr = w_arr/norm, x_arr/norm, y_arr/norm, z_arr/norm
            
            # Calculate yaw (theta)
            theta = np.arctan2(2 * (w_arr * z_arr + x_arr * y_arr), 1 - 2 * (y_arr**2 + z_arr**2))
            
            # Calculate pitch (phi)
            phi = np.arcsin(2 * (w_arr * y_arr - z_arr * x_arr))
            
            return np.column_stack([theta, phi])
        else:
            raise ValueError("Input array must be 1D or 2D")
    
    # Scalar computation
    # Normalize quaternion
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # Calculate yaw (theta)
    theta = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    
    # Calculate pitch (phi)
    phi = math.asin(2 * (w * y - z * x))
    
    return theta, phi

def euler_to_quaternion(roll, pitch=None, yaw=None):
    """
    Convert Euler angles to quaternion (ZYX rotation order).
    
    Args:
        roll: Either roll angle (scalar) or array of Euler angles
        pitch: pitch angle (if roll is scalar)
        yaw: yaw angle (if roll is scalar)
        
    Array formats:
        - 1D array: [roll, pitch, yaw]
        - 2D array: [[roll1,pitch1,yaw1], [roll2,pitch2,yaw2], ...] shape (N, 3)
        
    Returns:
        tuple or numpy.ndarray:
        - If scalar input: (w, x, y, z)
        - If array input: array of shape (N, 4) with columns [w, x, y, z]
    """
    # Handle array inputs
    if pitch is None and yaw is None:
        coords = np.asarray(roll)
        if coords.ndim == 1:
            # 1D array case
            roll, pitch, yaw = coords[0], coords[1], coords[2]
        elif coords.ndim == 2:
            # 2D array case - vectorized computation
            roll_arr, pitch_arr, yaw_arr = coords[:, 0], coords[:, 1], coords[:, 2]
            cr = np.cos(roll_arr * 0.5)
            sr = np.sin(roll_arr * 0.5)
            cp = np.cos(pitch_arr * 0.5)
            sp = np.sin(pitch_arr * 0.5)
            cy = np.cos(yaw_arr * 0.5)
            sy = np.sin(yaw_arr * 0.5)
            
            w = cr * cp * cy + sr * sp * sy
            x = sr * cp * cy - cr * sp * sy
            y = cr * sp * cy + sr * cp * sy
            z = cr * cp * sy - sr * sp * cy
            
            return np.column_stack([w, x, y, z])
        else:
            raise ValueError("Input array must be 1D or 2D")
    
    # Scalar computation
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return w, x, y, z

def quaternion_to_euler(w, x=None, y=None, z=None):
    """
    Convert quaternion to Euler angles (ZYX rotation order).
    
    Args:
        w: Either w component (scalar) or array of quaternions
        x, y, z: quaternion components (if w is scalar)
        
    Array formats:
        - 1D array: [w, x, y, z]
        - 2D array: [[w1,x1,y1,z1], [w2,x2,y2,z2], ...] shape (N, 4)
        
    Returns:
        tuple or numpy.ndarray:
        - If scalar input: (roll, pitch, yaw)
        - If array input: array of shape (N, 3) with columns [roll, pitch, yaw]
    """
    # Handle array inputs
    if x is None and y is None and z is None:
        coords = np.asarray(w)
        if coords.ndim == 1:
            # 1D array case
            w, x, y, z = coords[0], coords[1], coords[2], coords[3]
        elif coords.ndim == 2:
            # 2D array case - vectorized computation
            w_arr, x_arr, y_arr, z_arr = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
            
            # Roll (x-axis rotation)
            sinr_cosp = 2 * (w_arr * x_arr + y_arr * z_arr)
            cosr_cosp = 1 - 2 * (x_arr**2 + y_arr**2)
            roll = np.arctan2(sinr_cosp, cosr_cosp)
            
            # Pitch (y-axis rotation)
            sinp = 2 * (w_arr * y_arr - z_arr * x_arr)
            pitch = np.where(np.abs(sinp) >= 1, np.copysign(np.pi / 2, sinp), np.arcsin(sinp))
            
            # Yaw (z-axis rotation)
            siny_cosp = 2 * (w_arr * z_arr + x_arr * y_arr)
            cosy_cosp = 1 - 2 * (y_arr**2 + z_arr**2)
            yaw = np.arctan2(siny_cosp, cosy_cosp)
            
            return np.column_stack([roll, pitch, yaw])
        else:
            raise ValueError("Input array must be 1D or 2D")
    
    # Scalar computation
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw

def quaternion_to_rotation_matrix(w, x=None, y=None, z=None):
    """
    Convert quaternion to 3x3 rotation matrix.
    
    Args:
        w: Either w component (scalar) or array of quaternions
        x, y, z: quaternion components (if w is scalar)
        
    Array formats:
        - 1D array: [w, x, y, z]
        - 2D array: [[w1,x1,y1,z1], [w2,x2,y2,z2], ...] shape (N, 4)
        
    Returns:
        numpy.ndarray:
        - If scalar input: 3x3 rotation matrix
        - If array input: array of shape (N, 3, 3) with rotation matrices
    """
    # Handle array inputs
    if x is None and y is None and z is None:
        coords = np.asarray(w)
        if coords.ndim == 1:
            # 1D array case
            w, x, y, z = coords[0], coords[1], coords[2], coords[3]
        elif coords.ndim == 2:
            # 2D array case - vectorized computation
            w_arr, x_arr, y_arr, z_arr = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
            # Normalize quaternions
            norm = np.sqrt(w_arr**2 + x_arr**2 + y_arr**2 + z_arr**2)
            w_arr, x_arr, y_arr, z_arr = w_arr/norm, x_arr/norm, y_arr/norm, z_arr/norm
            
            # Create rotation matrices
            R = np.zeros((len(coords), 3, 3))
            R[:, 0, 0] = 1 - 2*(y_arr**2 + z_arr**2)
            R[:, 0, 1] = 2*(x_arr*y_arr - w_arr*z_arr)
            R[:, 0, 2] = 2*(x_arr*z_arr + w_arr*y_arr)
            R[:, 1, 0] = 2*(x_arr*y_arr + w_arr*z_arr)
            R[:, 1, 1] = 1 - 2*(x_arr**2 + z_arr**2)
            R[:, 1, 2] = 2*(y_arr*z_arr - w_arr*x_arr)
            R[:, 2, 0] = 2*(x_arr*z_arr - w_arr*y_arr)
            R[:, 2, 1] = 2*(y_arr*z_arr + w_arr*x_arr)
            R[:, 2, 2] = 1 - 2*(x_arr**2 + y_arr**2)
            
            return R
        else:
            raise ValueError("Input array must be 1D or 2D")
    
    # Scalar computation
    # Normalize quaternion
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])
    
    return R

def euler_to_unit_vector(roll, pitch=None, yaw=None):
    """
    Convert Euler angles to unit vector pointing in the direction of rotation.
    
    Args:
        roll: Either roll angle (scalar) or array of Euler angles
        pitch: pitch angle (if roll is scalar)
        yaw: yaw angle (if roll is scalar)
        
    Array formats:
        - 1D array: [roll, pitch, yaw]
        - 2D array: [[roll1,pitch1,yaw1], [roll2,pitch2,yaw2], ...] shape (N, 3)
        
    Returns:
        tuple or numpy.ndarray:
        - If scalar input: (x, y, z)
        - If array input: array of shape (N, 3) with columns [x, y, z]
    """
    # Handle array inputs
    if pitch is None and yaw is None:
        coords = np.asarray(roll)
        if coords.ndim == 1:
            # 1D array case
            roll, pitch, yaw = coords[0], coords[1], coords[2]
        elif coords.ndim == 2:
            # 2D array case - vectorized computation
            roll_arr, pitch_arr, yaw_arr = coords[:, 0], coords[:, 1], coords[:, 2]
            # Apply rotations to the initial forward vector [1, 0, 0]
            x = np.cos(pitch_arr) * np.cos(yaw_arr)
            y = np.cos(pitch_arr) * np.sin(yaw_arr)
            z = -np.sin(pitch_arr)
            
            return np.column_stack([x, y, z])
        else:
            raise ValueError("Input array must be 1D or 2D")
    
    # Scalar computation
    # Apply rotations to the initial forward vector [1, 0, 0]
    x = math.cos(pitch) * math.cos(yaw)
    y = math.cos(pitch) * math.sin(yaw)
    z = -math.sin(pitch)
    
    return x, y, z

def unit_vector_to_euler(x, y=None, z=None):
    """
    Convert unit vector to Euler angles.
    
    Args:
        x: Either x component (scalar) or array of vectors
        y, z: vector components (if x is scalar)
        
    Array formats:
        - 1D array: [x, y, z]
        - 2D array: [[x1,y1,z1], [x2,y2,z2], ...] shape (N, 3)
        
    Returns:
        tuple or numpy.ndarray:
        - If scalar input: (roll, pitch, yaw)
        - If array input: array of shape (N, 3) with columns [roll, pitch, yaw]
    """
    # Handle array inputs
    if y is None and z is None:
        coords = np.asarray(x)
        if coords.ndim == 1:
            # 1D array case
            x, y, z = coords[0], coords[1], coords[2]
        elif coords.ndim == 2:
            # 2D array case - vectorized computation
            x_arr, y_arr, z_arr = coords[:, 0], coords[:, 1], coords[:, 2]
            # Normalize the vectors
            norm = np.sqrt(x_arr**2 + y_arr**2 + z_arr**2)
            mask = norm > 0
            x_arr = np.where(mask, x_arr/norm, 0)
            y_arr = np.where(mask, y_arr/norm, 0)
            z_arr = np.where(mask, z_arr/norm, 0)
            
            # Calculate pitch and yaw from the vector
            pitch = -np.arcsin(z_arr)
            yaw = np.arctan2(y_arr, x_arr)
            
            # Roll cannot be determined from direction vector alone
            # Set to 0 as a convention
            roll = np.zeros_like(pitch)
            
            return np.column_stack([roll, pitch, yaw])
        else:
            raise ValueError("Input array must be 1D or 2D")
    
    # Scalar computation
    # Normalize the vector
    norm = math.sqrt(x*x + y*y + z*z)
    if norm == 0:
        return 0, 0, 0
    x, y, z = x/norm, y/norm, z/norm
    
    # Calculate pitch and yaw from the vector
    pitch = -math.asin(z)
    yaw = math.atan2(y, x)
    
    # Roll cannot be determined from direction vector alone
    # Set to 0 as a convention
    roll = 0
    
    return roll, pitch, yaw

# ============================================================================
# COORDINATE FRAME CONVERSIONS (NED <-> ENU)
# ============================================================================

def ned_to_enu_coordinates(x_ned, y_ned=None, z_ned=None):
    """
    Convert coordinates from NED (North-East-Down) to ENU (East-North-Up).
    
    Args:
        x_ned: Either x coordinate (scalar) or array of coordinates
        y_ned, z_ned: coordinates (if x_ned is scalar)
        
    Array formats:
        - 1D array: [x_ned, y_ned, z_ned]
        - 2D array: [[x1,y1,z1], [x2,y2,z2], ...] shape (N, 3)
        
    Returns:
        tuple or numpy.ndarray:
        - If scalar input: (x_enu, y_enu, z_enu)
        - If array input: array of shape (N, 3) with columns [x_enu, y_enu, z_enu]
    """
    # Handle array inputs
    if y_ned is None and z_ned is None:
        coords = np.asarray(x_ned)
        if coords.ndim == 1:
            # 1D array case
            x_ned, y_ned, z_ned = coords[0], coords[1], coords[2]
        elif coords.ndim == 2:
            # 2D array case - vectorized computation
            x_ned_arr, y_ned_arr, z_ned_arr = coords[:, 0], coords[:, 1], coords[:, 2]
            x_enu = y_ned_arr   # East = North_NED
            y_enu = x_ned_arr   # North = East_NED  
            z_enu = -z_ned_arr  # Up = -Down_NED
            
            return np.column_stack([x_enu, y_enu, z_enu])
        else:
            raise ValueError("Input array must be 1D or 2D")
    
    # Scalar computation
    x_enu = y_ned   # East = North_NED
    y_enu = x_ned   # North = East_NED  
    z_enu = -z_ned  # Up = -Down_NED
    
    return x_enu, y_enu, z_enu

def enu_to_ned_coordinates(x_enu, y_enu=None, z_enu=None):
    """
    Convert coordinates from ENU (East-North-Up) to NED (North-East-Down).
    
    Args:
        x_enu: Either x coordinate (scalar) or array of coordinates
        y_enu, z_enu: coordinates (if x_enu is scalar)
        
    Array formats:
        - 1D array: [x_enu, y_enu, z_enu]
        - 2D array: [[x1,y1,z1], [x2,y2,z2], ...] shape (N, 3)
        
    Returns:
        tuple or numpy.ndarray:
        - If scalar input: (x_ned, y_ned, z_ned)
        - If array input: array of shape (N, 3) with columns [x_ned, y_ned, z_ned]
    """
    # Handle array inputs
    if y_enu is None and z_enu is None:
        coords = np.asarray(x_enu)
        if coords.ndim == 1:
            # 1D array case
            x_enu, y_enu, z_enu = coords[0], coords[1], coords[2]
        elif coords.ndim == 2:
            # 2D array case - vectorized computation
            x_enu_arr, y_enu_arr, z_enu_arr = coords[:, 0], coords[:, 1], coords[:, 2]
            x_ned = y_enu_arr   # North = North_ENU
            y_ned = x_enu_arr   # East = East_ENU
            z_ned = -z_enu_arr  # Down = -Up_ENU
            
            return np.column_stack([x_ned, y_ned, z_ned])
        else:
            raise ValueError("Input array must be 1D or 2D")
    
    # Scalar computation
    x_ned = y_enu   # North = North_ENU
    y_ned = x_enu   # East = East_ENU
    z_ned = -z_enu  # Down = -Up_ENU
    
    return x_ned, y_ned, z_ned

def ned_to_enu_euler(roll_ned, pitch_ned=None, yaw_ned=None):
    """
    Convert Euler angles from NED to ENU frame.
    
    Args:
        roll_ned: Either roll angle (scalar) or array of Euler angles
        pitch_ned, yaw_ned: Euler angles (if roll_ned is scalar)
        
    Array formats:
        - 1D array: [roll_ned, pitch_ned, yaw_ned]
        - 2D array: [[roll1,pitch1,yaw1], [roll2,pitch2,yaw2], ...] shape (N, 3)
        
    Returns:
        tuple or numpy.ndarray:
        - If scalar input: (roll_enu, pitch_enu, yaw_enu)
        - If array input: array of shape (N, 3) with columns [roll_enu, pitch_enu, yaw_enu]
    """
    # Handle array inputs
    if pitch_ned is None and yaw_ned is None:
        coords = np.asarray(roll_ned)
        if coords.ndim == 1:
            # 1D array case
            roll_ned, pitch_ned, yaw_ned = coords[0], coords[1], coords[2]
        elif coords.ndim == 2:
            # 2D array case - vectorized computation
            roll_ned_arr, pitch_ned_arr, yaw_ned_arr = coords[:, 0], coords[:, 1], coords[:, 2]
            roll_enu = pitch_ned_arr    # Roll_ENU = Pitch_NED
            pitch_enu = roll_ned_arr    # Pitch_ENU = Roll_NED
            yaw_enu = -yaw_ned_arr + math.pi/2  # Yaw_ENU = -Yaw_NED + 90°
            
            # Normalize yaw to [-π, π]
            yaw_enu = np.where(yaw_enu > math.pi, yaw_enu - 2*math.pi, yaw_enu)
            yaw_enu = np.where(yaw_enu < -math.pi, yaw_enu + 2*math.pi, yaw_enu)
            
            return np.column_stack([roll_enu, pitch_enu, yaw_enu])
        else:
            raise ValueError("Input array must be 1D or 2D")
    
    # Scalar computation
    roll_enu = pitch_ned    # Roll_ENU = Pitch_NED
    pitch_enu = roll_ned    # Pitch_ENU = Roll_NED
    yaw_enu = -yaw_ned + math.pi/2  # Yaw_ENU = -Yaw_NED + 90°
    
    # Normalize yaw to [-π, π]
    while yaw_enu > math.pi:
        yaw_enu -= 2 * math.pi
    while yaw_enu < -math.pi:
        yaw_enu += 2 * math.pi
    
    return roll_enu, pitch_enu, yaw_enu

def enu_to_ned_euler(roll_enu, pitch_enu=None, yaw_enu=None):
    """
    Convert Euler angles from ENU to NED frame.
    
    Args:
        roll_enu: Either roll angle (scalar) or array of Euler angles
        pitch_enu, yaw_enu: Euler angles (if roll_enu is scalar)
        
    Array formats:
        - 1D array: [roll_enu, pitch_enu, yaw_enu]
        - 2D array: [[roll1,pitch1,yaw1], [roll2,pitch2,yaw2], ...] shape (N, 3)
        
    Returns:
        tuple or numpy.ndarray:
        - If scalar input: (roll_ned, pitch_ned, yaw_ned)
        - If array input: array of shape (N, 3) with columns [roll_ned, pitch_ned, yaw_ned]
    """
    # Handle array inputs
    if pitch_enu is None and yaw_enu is None:
        coords = np.asarray(roll_enu)
        if coords.ndim == 1:
            # 1D array case
            roll_enu, pitch_enu, yaw_enu = coords[0], coords[1], coords[2]
        elif coords.ndim == 2:
            # 2D array case - vectorized computation
            roll_enu_arr, pitch_enu_arr, yaw_enu_arr = coords[:, 0], coords[:, 1], coords[:, 2]
            roll_ned = pitch_enu_arr    # Roll_NED = Pitch_ENU
            pitch_ned = roll_enu_arr    # Pitch_NED = Roll_ENU
            yaw_ned = -(yaw_enu_arr - math.pi/2)  # Yaw_NED = -(Yaw_ENU - 90°)
            
            # Normalize yaw to [-π, π]
            yaw_ned = np.where(yaw_ned > math.pi, yaw_ned - 2*math.pi, yaw_ned)
            yaw_ned = np.where(yaw_ned < -math.pi, yaw_ned + 2*math.pi, yaw_ned)
            
            return np.column_stack([roll_ned, pitch_ned, yaw_ned])
        else:
            raise ValueError("Input array must be 1D or 2D")
    
    # Scalar computation
    roll_ned = pitch_enu    # Roll_NED = Pitch_ENU
    pitch_ned = roll_enu    # Pitch_NED = Roll_ENU
    yaw_ned = -(yaw_enu - math.pi/2)  # Yaw_NED = -(Yaw_ENU - 90°)
    
    # Normalize yaw to [-π, π]
    while yaw_ned > math.pi:
        yaw_ned -= 2 * math.pi
    while yaw_ned < -math.pi:
        yaw_ned += 2 * math.pi
    
    return roll_ned, pitch_ned, yaw_ned

# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

if __name__ == "__main__":
    print("=== Enhanced Coordinate Conversions with Array Support ===")
    
    # Test scalar inputs (original functionality)
    print("\n--- Scalar Input Tests ---")
    x, y, z = 1, 1, 1
    r, theta, phi = cartesian_to_spherical(x, y, z)
    print(f"Cartesian ({x}, {y}, {z}) -> Spherical ({r:.3f}, {theta:.3f}, {phi:.3f})")
    
    # Test 1D array inputs
    print("\n--- 1D Array Input Tests ---")
    cartesian_1d = [1, 1, 1]
    spherical_result = cartesian_to_spherical(cartesian_1d)
    print(f"1D Array {cartesian_1d} -> Spherical {spherical_result}")
    
    # Test 2D array inputs (batch processing)
    print("\n--- 2D Array Input Tests (Batch Processing) ---")
    cartesian_batch = np.array([[1, 1, 1], [2, 0, 0], [0, 3, 4]])
    spherical_batch = cartesian_to_spherical(cartesian_batch)
    print(f"Batch Cartesian:\n{cartesian_batch}")
    print(f"Batch Spherical:\n{spherical_batch}")
    
    # Test round-trip conversion
    cartesian_roundtrip = spherical_to_cartesian(spherical_batch)
    print(f"Round-trip back to Cartesian:\n{cartesian_roundtrip}")
    
    # Test Euler to Quaternion batch conversion
    print("\n--- Euler to Quaternion Batch Test ---")
    euler_batch = np.array([[0.1, 0.2, 0.3], [0.5, 0.0, 1.0], [1.57, 0.78, 2.35]])
    quat_batch = euler_to_quaternion(euler_batch)
    print(f"Batch Euler:\n{euler_batch}")
    print(f"Batch Quaternion:\n{quat_batch}")
    
    # Test rotation matrix batch conversion
    print("\n--- Quaternion to Rotation Matrix Batch Test ---")
    rotation_matrices = quaternion_to_rotation_matrix(quat_batch)
    print(f"Batch Rotation Matrices shape: {rotation_matrices.shape}")
    print(f"First rotation matrix:\n{rotation_matrices[0]}")
    
    # Test NED/ENU frame conversions
    print("\n--- Frame Conversion Tests ---")
    ned_coords = np.array([[1, 2, 3], [4, 5, 6]])
    enu_coords = ned_to_enu_coordinates(ned_coords)
    print(f"NED coordinates:\n{ned_coords}")
    print(f"ENU coordinates:\n{enu_coords}")
    
    # Performance comparison example
    print("\n--- Performance Test Setup ---")
    print("For large datasets, vectorized operations will be significantly faster.")
    print("Example: Converting 1000 coordinate sets at once vs. one by one.")
    
    # Generate test data
    np.random.seed(42)
    large_batch = np.random.rand(1000, 3) * 10
    
    import time
    
    # Time the vectorized approach
    start_time = time.time()
    result_vectorized = cartesian_to_spherical(large_batch)
    vectorized_time = time.time() - start_time
    
    # Time the loop approach
    start_time = time.time()
    result_loop = []
    for i in range(len(large_batch)):
        x, y, z = large_batch[i]
        result_loop.append(cartesian_to_spherical(x, y, z))
    result_loop = np.array(result_loop)
    loop_time = time.time() - start_time
    
    print(f"Vectorized approach: {vectorized_time:.4f} seconds")
    print(f"Loop approach: {loop_time:.4f} seconds")
    print(f"Speedup: {loop_time/vectorized_time:.1f}x faster")
    
    # Verify results are equivalent
    print(f"Results match: {np.allclose(result_vectorized, result_loop)}")