import numpy as np
import math

def ensure_list(value):
    """Convert Vector3 objects or other iterables to plain Python lists"""
    if hasattr(value, 'to_list'):
        return value.to_list()
    elif hasattr(value, '__iter__') and not isinstance(value, str):
        return [float(x) if hasattr(x, 'to_list') else float(x) for x in value]
    else:
        return value

class Vector3:
    """Simple 3D vector class"""
    def __init__(self, components):
        if isinstance(components, (list, tuple, np.ndarray)):
            self.x, self.y, self.z = float(components[0]), float(components[1]), float(components[2])
        else:
            raise ValueError("Vector3 requires a list, tuple, or array of 3 components")
    
    def __getitem__(self, index):
        return [self.x, self.y, self.z][index]
    
    def __setitem__(self, index, value):
        if index == 0:
            self.x = float(value)
        elif index == 1:
            self.y = float(value)
        elif index == 2:
            self.z = float(value)
        else:
            raise IndexError("Vector3 index out of range")
    
    def __add__(self, other):
        return Vector3([self.x + other.x, self.y + other.y, self.z + other.z])
    
    def __sub__(self, other):
        return Vector3([self.x - other.x, self.y - other.y, self.z - other.z])
    
    def __mul__(self, scalar):
        return Vector3([self.x * scalar, self.y * scalar, self.z * scalar])
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    def magnitude(self):
        """Calculate the magnitude (length) of the vector"""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self):
        """Return a normalized version of this vector"""
        mag = self.magnitude()
        if mag > 1e-8:
            return Vector3([self.x / mag, self.y / mag, self.z / mag])
        else:
            return Vector3([0, 0, 0])
    
    def dot(self, other):
        """Calculate dot product with another vector"""
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other):
        """Calculate cross product with another vector"""
        return Vector3([
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        ])
    
    def to_list(self):
        return [self.x, self.y, self.z]
    
    def __repr__(self):
        return f"Vector3([{self.x}, {self.y}, {self.z}])"

class Quaternion:
    """Simple quaternion class with basic operations"""
    def __init__(self, components=None):
        if components is None:
            # Identity quaternion
            self.x, self.y, self.z, self.w = 0.0, 0.0, 0.0, 1.0
        elif isinstance(components, (list, tuple, np.ndarray)):
            if len(components) == 4:
                self.x, self.y, self.z, self.w = float(components[0]), float(components[1]), float(components[2]), float(components[3])
            else:
                raise ValueError("Quaternion requires 4 components [x, y, z, w]")
        else:
            raise ValueError("Invalid quaternion components")
        
    def __getitem__(self, index):
        return [self.x, self.y, self.z, self.w][index]
    
    def __setitem__(self, index, value):
        components = [self.x, self.y, self.z, self.w]
        components[index] = float(value)
        self.x, self.y, self.z, self.w = components
    
    @property
    def xyzw(self):
        """Return quaternion as [x, y, z, w] format"""
        return [self.x, self.y, self.z, self.w]
    
    def magnitude(self):
        """Calculate the magnitude of the quaternion"""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)
    
    def normalise(self):
        """Normalize the quaternion in-place"""
        magnitude = self.magnitude()
        if magnitude > 1e-8:
            self.x /= magnitude
            self.y /= magnitude
            self.z /= magnitude
            self.w /= magnitude
        else:
            # Default to identity quaternion if magnitude is too small
            self.x, self.y, self.z, self.w = 0.0, 0.0, 0.0, 1.0
    
    def normalized(self):
        """Return a normalized copy of this quaternion"""
        result = Quaternion([self.x, self.y, self.z, self.w])
        result.normalise()
        return result
    
    def conjugate(self):
        """Return the conjugate of this quaternion"""
        return Quaternion([-self.x, -self.y, -self.z, self.w])
    
    def inverse(self):
        """Return the inverse of this quaternion"""
        conj = self.conjugate()
        mag_sq = self.x**2 + self.y**2 + self.z**2 + self.w**2
        if mag_sq > 1e-8:
            return Quaternion([conj.x / mag_sq, conj.y / mag_sq, conj.z / mag_sq, conj.w / mag_sq])
        else:
            return Quaternion()  # Return identity if magnitude is too small
    
    def __mul__(self, other):
        """Quaternion multiplication"""
        if isinstance(other, Quaternion):
            # Quaternion * Quaternion
            w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
            x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
            y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
            z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
            return Quaternion([x, y, z, w])
        elif isinstance(other, Vector3):
            # Rotate vector by quaternion
            # Convert vector to quaternion
            vec_quat = Quaternion([other.x, other.y, other.z, 0.0])
            # Compute quaternion conjugate
            conj = self.conjugate()
            # Rotate: q * v * q*
            result = self * vec_quat * conj
            return Vector3([result.x, result.y, result.z])
        else:
            raise TypeError("Quaternion multiplication only supports Quaternion or Vector3")
    
    def to_mujoco_format(self):
        """Convert to MuJoCo quaternion format [w, x, y, z]"""
        return [self.w, self.x, self.y, self.z]
    
    def to_rotation_matrix(self):
        """Convert quaternion to 3x3 rotation matrix"""
        # Normalize first
        q = self.normalized()
        
        # Extract components
        x, y, z, w = q.x, q.y, q.z, q.w
        
        # Calculate rotation matrix elements
        xx, xy, xz = x*x, x*y, x*z
        yy, yz, zz = y*y, y*z, z*z
        wx, wy, wz = w*x, w*y, w*z
        
        return [
            [1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)],
            [2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)],
            [2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)]
        ]
    
    @classmethod
    def from_axis_angle(cls, axis, angle):
        """Create quaternion from axis-angle representation"""
        if isinstance(axis, (list, tuple)):
            axis = Vector3(axis)
        
        # Normalize the axis
        axis = axis.normalize()
        half_angle = angle / 2
        sin_half = math.sin(half_angle)
        
        return cls([
            axis.x * sin_half,
            axis.y * sin_half, 
            axis.z * sin_half,
            math.cos(half_angle)
        ])
    
    @classmethod
    def from_euler(cls, roll, pitch, yaw):
        """Create quaternion from Euler angles (in radians)"""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        return cls([
            sr * cp * cy - cr * sp * sy,  # x
            cr * sp * cy + sr * cp * sy,  # y
            cr * cp * sy - sr * sp * cy,  # z
            cr * cp * cy + sr * sp * sy   # w
        ])
    
    def to_euler(self):
        """Convert quaternion to Euler angles (roll, pitch, yaw) in radians"""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1 - 2 * (self.x * self.x + self.y * self.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (self.w * self.y - self.z * self.x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1 - 2 * (self.y * self.y + self.z * self.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def __repr__(self):
        return f"Quaternion([{self.x}, {self.y}, {self.z}, {self.w}])"

def angle_to_quaternion(angle: float, axis: tuple):
    """
    Convert a rotation angle around a given axis to a quaternion.
    
    :param angle: Rotation angle in radians.
    :param axis: Axis of rotation as a tuple (x, y, z).
    :return: Quaternion as a Quaternion object.
    """
    # Normalize the axis vector
    axis_length = math.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)
    if axis_length < 1e-8:
        # If axis is zero, return identity quaternion
        return Quaternion([0, 0, 0, 1])
    
    norm_axis = (axis[0] / axis_length, axis[1] / axis_length, axis[2] / axis_length)
    
    half_angle = angle / 2
    sin_half_angle = math.sin(half_angle)
    
    q_x = norm_axis[0] * sin_half_angle
    q_y = norm_axis[1] * sin_half_angle
    q_z = norm_axis[2] * sin_half_angle
    q_w = math.cos(half_angle)

    return Quaternion([q_x, q_y, q_z, q_w])