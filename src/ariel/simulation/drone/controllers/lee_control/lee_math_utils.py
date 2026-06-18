# -*- coding: utf-8 -*-
"""
Lee Control Mathematical Utilities
NumPy-based implementations of mathematical functions for Lee geometric controller
Adapted from PyTorch tensor operations to work with NumPy arrays
"""

import numpy as np
from numpy import sin, cos, pi
from numpy.linalg import norm
import logging

# Set up simple logging
logger = logging.getLogger("lee_control")


def normalize(x, eps=1e-9):
    """Normalize a vector or array of vectors"""
    if x.ndim == 1:
        return x / max(norm(x), eps)
    else:
        norms = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / np.maximum(norms, eps)


def quat_multiply(q, p):
    """Quaternion multiplication for arrays of quaternions
    q, p: arrays of shape (..., 4) where [..., :] = [x, y, z, w]
    """
    if q.ndim == 1 and p.ndim == 1:
        # Single quaternion case
        qx, qy, qz, qw = q[0], q[1], q[2], q[3]
        px, py, pz, pw = p[0], p[1], p[2], p[3]
        
        w = qw*pw - qx*px - qy*py - qz*pz
        x = qw*px + qx*pw + qy*pz - qz*py
        y = qw*py - qx*pz + qy*pw + qz*px
        z = qw*pz + qx*py - qy*px + qz*pw
        
        return np.array([x, y, z, w])
    else:
        # Array case
        q = np.atleast_2d(q)
        p = np.atleast_2d(p)
        
        qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        px, py, pz, pw = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
        
        w = qw*pw - qx*px - qy*py - qz*pz
        x = qw*px + qx*pw + qy*pz - qz*py
        y = qw*py - qx*pz + qy*pw + qz*px
        z = qw*pz + qx*py - qy*px + qz*pw
        
        return np.stack([x, y, z, w], axis=-1)


def quat_mul(a, b):
    """Alias for quat_multiply for compatibility"""
    return quat_multiply(a, b)


def quat_conjugate(q):
    """Quaternion conjugate"""
    q_conj = q.copy()
    q_conj[..., :3] *= -1  # Negate x, y, z components
    return q_conj


def quat_inverse(q):
    """Quaternion inverse"""
    return quat_conjugate(q)


def quat_rotate(q, v):
    """Rotate vector v by quaternion q
    q: quaternion array (..., 4) [x, y, z, w]
    v: vector array (..., 3) [x, y, z]
    """
    if q.ndim == 1:
        q = q.reshape(1, -1)
        v = v.reshape(1, -1)
        single = True
    else:
        single = False
    
    q_w = q[..., 3:4]  # w component
    q_vec = q[..., :3]  # xyz components
    
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v, axis=-1) * q_w * 2.0
    c = q_vec * np.sum(q_vec * v, axis=-1, keepdims=True) * 2.0
    
    result = a + b + c
    return result[0] if single else result


def quat_rotate_inverse(q, v):
    """Rotate vector v by inverse of quaternion q"""
    return quat_rotate(quat_inverse(q), v)


def quat_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix
    q: quaternion array (..., 4) [x, y, z, w]
    Returns: rotation matrix array (..., 3, 3)
    """
    if q.ndim == 1:
        q = q.reshape(1, -1)
        single = True
    else:
        single = False
    
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    xx, xy, xz, xw = x*x, x*y, x*z, x*w
    yy, yz, yw = y*y, y*z, y*w
    zz, zw = z*z, z*w
    
    m = np.zeros((*q.shape[:-1], 3, 3))
    
    m[..., 0, 0] = 1 - 2*(yy + zz)
    m[..., 0, 1] = 2*(xy - zw)
    m[..., 0, 2] = 2*(xz + yw)
    m[..., 1, 0] = 2*(xy + zw)
    m[..., 1, 1] = 1 - 2*(xx + zz)
    m[..., 1, 2] = 2*(yz - xw)
    m[..., 2, 0] = 2*(xz - yw)
    m[..., 2, 1] = 2*(yz + xw)
    m[..., 2, 2] = 1 - 2*(xx + yy)
    
    return m[0] if single else m


def quat_from_euler_xyz(roll, pitch, yaw):
    """Convert Euler angles to quaternion
    Returns quaternion in [x, y, z, w] format
    """
    if np.isscalar(roll):
        # Single value case
        cy = cos(yaw * 0.5)
        sy = sin(yaw * 0.5)
        cr = cos(roll * 0.5)
        sr = sin(roll * 0.5)
        cp = cos(pitch * 0.5)
        sp = sin(pitch * 0.5)

        qw = cy * cr * cp + sy * sr * sp
        qx = cy * sr * cp - sy * cr * sp
        qy = cy * cr * sp + sy * sr * cp
        qz = sy * cr * cp - cy * sr * sp

        return np.array([qx, qy, qz, qw])
    else:
        # Array case
        roll = np.atleast_1d(roll)
        pitch = np.atleast_1d(pitch)
        yaw = np.atleast_1d(yaw)
        
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)

        qw = cy * cr * cp + sy * sr * sp
        qx = cy * sr * cp - sy * cr * sp
        qy = cy * cr * sp + sy * sr * cp
        qz = sy * cr * cp - cy * sr * sp

        return np.stack([qx, qy, qz, qw], axis=-1)


def quat_from_euler_xyz_tensor(euler_xyz):
    """Convert Euler angle array to quaternion array
    euler_xyz: array of shape (..., 3) [roll, pitch, yaw]
    Returns: quaternion array (..., 4) [x, y, z, w]
    """
    roll = euler_xyz[..., 0]
    pitch = euler_xyz[..., 1]
    yaw = euler_xyz[..., 2]
    return quat_from_euler_xyz(roll, pitch, yaw)


def get_euler_xyz_tensor(q):
    """Convert quaternion to Euler angles
    q: quaternion array (..., 4) [x, y, z, w]
    Returns: Euler angles (..., 3) [roll, pitch, yaw]
    """
    if q.ndim == 1:
        q = q.reshape(1, -1)
        single = True
    else:
        single = False
    
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = qw**2 - qx**2 - qy**2 + qz**2
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    pitch = np.where(np.abs(sinp) >= 1, 
                     np.sign(sinp) * pi / 2.0, 
                     np.arcsin(sinp))

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = qw**2 + qx**2 - qy**2 - qz**2
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    result = np.stack([roll % (2 * pi), pitch % (2 * pi), yaw % (2 * pi)], axis=-1)
    return result[0] if single else result


def compute_vee_map(skew_matrix):
    """Extract vector from skew-symmetric matrix
    skew_matrix: array of shape (..., 3, 3)
    Returns: vector array (..., 3)
    """
    vee_map = np.stack([
        -skew_matrix[..., 1, 2],
        skew_matrix[..., 0, 2],
        -skew_matrix[..., 0, 1]
    ], axis=-1)
    return vee_map


def matrix_to_quaternion(rotation_matrix):
    """Convert rotation matrix to quaternion
    Simple implementation to replace pytorch3d.transforms.matrix_to_quaternion
    """
    if rotation_matrix.ndim == 2:
        rotation_matrix = rotation_matrix.reshape(1, 3, 3)
        single = True
    else:
        single = False
    
    batch_size = rotation_matrix.shape[0]
    quaternions = np.zeros((batch_size, 4))
    
    for i in range(batch_size):
        R = rotation_matrix[i]
        
        # Shepperd's method
        trace = np.trace(R)
        
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
        
        # Convert from [w, x, y, z] to [x, y, z, w] format
        quaternions[i] = [qx, qy, qz, qw]
    
    return quaternions[0] if single else quaternions


def vehicle_frame_quat_from_quat(body_quat):
    """Get vehicle frame quaternion from body quaternion (only yaw component)"""
    body_euler = get_euler_xyz_tensor(body_quat) * np.array([0.0, 0.0, 1.0])
    return quat_from_euler_xyz_tensor(body_euler)


def rand_float_uniform(lower, upper, shape=None):
    """Generate random float values between lower and upper bounds"""
    if shape is None and np.isscalar(lower) and np.isscalar(upper):
        return np.random.uniform(lower, upper)
    else:
        if np.isscalar(lower):
            lower = np.full(shape, lower)
        if np.isscalar(upper):
            upper = np.full(shape, upper)
        return np.random.uniform(lower, upper, shape)


def pd_control(pos_error, vel_error, stiffness, damping):
    """PD controller implementation"""
    return stiffness * pos_error + damping * vel_error


def ssa(angle):
    """Smallest signed angle - wrap angle to [-pi, pi]"""
    return np.remainder(angle + pi, 2 * pi) - pi