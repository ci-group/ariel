import numpy as np
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.module import Module, AttachmentPoint
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.unbuilt_child import UnbuiltChild
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.math_utils import Vector3, Quaternion, ensure_list, angle_to_quaternion

class BezierTube(Module):
    """
    A tube that follows a Bézier curve, built from multiple cylinder segments
    """
    
    ATTACH = 0  # Attachment point index

    def __init__(
        self,
        control_points: list,  # List of Vector3 control points
        num_segments: int = 20,
        cylinder_radius: float = 0.05,
        name: str = "bezier_tube",
    ):
        self.control_points = control_points
        self.num_segments = num_segments
        self.cylinder_radius = cylinder_radius
        self.name = name
        
        # Calculate curve points and create cylinder segments
        self.curve_points = self._sample_bezier_curve()
        self.cylinder_segments = self._create_cylinder_segments()
        
        attachment_points = { 
            self.ATTACH: AttachmentPoint(
                offset=control_points[-1],
                orientation=Quaternion(),  # Rotate around the z-axis
            ),
        }

        super().__init__(attachment_points)

    @property
    def attach(self) -> Module:
        return self._children.get(self.ATTACH)

    @attach.setter
    def attach(self, module: Module) -> None:
        self.set_child(module, self.ATTACH)
    
    def _bernstein_poly(self, n, i, t):
        """Bernstein polynomial for Bézier curves"""
        from math import comb
        return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
    
    def _evaluate_bezier(self, t):
        """Evaluate Bézier curve at parameter t (0 to 1)"""
        n = len(self.control_points) - 1
        point = Vector3([0.0, 0.0, 0.0])
        
        for i, cp in enumerate(self.control_points):
            basis = self._bernstein_poly(n, i, t)
            # Multiply each component separately since Vector3 doesn't support scalar division
            weighted_point = Vector3([cp.x * basis, cp.y * basis, cp.z * basis])
            point = point + weighted_point
            
        return point
    
    def _evaluate_bezier_derivative(self, t):
        """Evaluate first derivative of Bézier curve at parameter t"""
        n = len(self.control_points) - 1
        if n == 0:
            return Vector3([0.0, 0.0, 1.0])  # Default direction
        
        derivative = Vector3([0.0, 0.0, 0.0])
        
        for i in range(n):
            cp_diff = self.control_points[i + 1] - self.control_points[i]
            basis = self._bernstein_poly(n - 1, i, t)
            # Multiply components separately
            weighted_diff = Vector3([
                cp_diff.x * (n * basis),
                cp_diff.y * (n * basis), 
                cp_diff.z * (n * basis)
            ])
            derivative = derivative + weighted_diff
            
        return derivative
    
    def _sample_bezier_curve(self):
        """Sample points along the Bézier curve"""
        points = []
        for i in range(self.num_segments + 1):
            t = i / self.num_segments
            point = self._evaluate_bezier(t)
            points.append(point)
        return points
    
    def _get_tangent_quaternion(self, index):
        """Get quaternion representing tangent direction at curve point"""
        if index == -1:
            index = len(self.curve_points) - 1
            
        # Calculate tangent vector
        if index == 0:
            tangent = self.curve_points[1] - self.curve_points[0]
        elif index == len(self.curve_points) - 1:
            tangent = self.curve_points[-1] - self.curve_points[-2]
        else:
            tangent = self.curve_points[index + 1] - self.curve_points[index - 1]
        
        # Normalize tangent using Vector3's normalize method
        tangent = tangent.normalize()
        if tangent.magnitude() < 1e-6:
            tangent = Vector3([0.0, 0.0, 1.0])  # Default direction
        
        # Create quaternion that rotates z-axis to tangent direction
        z_axis = Vector3([0.0, 0.0, 1.0])
        
        # Check if vectors are parallel
        dot_product = z_axis.dot(tangent)
        if abs(dot_product) > 0.9999:  # Nearly parallel
            if dot_product > 0:
                return Quaternion()  # Identity quaternion
            else:
                # 180 degree rotation around x-axis
                return Quaternion([1.0, 0.0, 0.0, 0.0])
        
        # Calculate rotation quaternion using cross product
        axis = z_axis.cross(tangent)
        axis = axis.normalize()
        
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        return angle_to_quaternion(angle, (axis.x, axis.y, axis.z))
    
    def _create_cylinder_segments(self):
        """Create cylinder segment data"""
        segments = []
        
        for i in range(len(self.curve_points) - 1):
            start_point = self.curve_points[i]
            end_point = self.curve_points[i + 1]
            
            # Calculate segment properties
            center = (start_point + end_point) * 0.5
            direction = end_point - start_point
            length = direction.magnitude()
            
            if length > 1e-6:
                direction = direction.normalize()
            else:
                direction = Vector3([0.0, 0.0, 1.0])
            
            # Create quaternion for cylinder orientation
            quat = self._vector_to_quaternion(direction)
            
            segments.append({
                'center': center,
                'length': length,
                'orientation': quat,
                'radius': self.cylinder_radius
            })
        
        return segments
    
    def _vector_to_quaternion(self, direction):
        """Convert direction vector to quaternion"""
        z_axis = Vector3([0.0, 0.0, 1.0])
        
        # Check for parallel vectors
        dot_product = z_axis.dot(direction)
        if abs(dot_product) > 0.9999:  # Nearly parallel
            if dot_product > 0:
                return Quaternion()  # Identity quaternion
            else:
                # 180 degree rotation around x-axis
                return Quaternion([1.0, 0.0, 0.0, 0.0])
        
        # Calculate cross product and angle
        axis = z_axis.cross(direction)
        axis = axis.normalize()
        
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        return angle_to_quaternion(angle, (axis.x, axis.y, axis.z))
    
    def build(self, mjcf, entry_point, attachment_point_pos, attachment_point_quat):
        """Build the Bézier tube as multiple cylinder segments"""
        pos_list = ensure_list(attachment_point_pos)
        mj_quat = attachment_point_quat.to_mujoco_format()
        
        # Create main body for the bezier tube
        main_body = entry_point.add("body", name=self.name, pos=pos_list, quat=mj_quat)
        
        # Add each cylinder segment
        for i, segment in enumerate(self.cylinder_segments):
            segment_name = f"{self.name}_seg_{i}"
            
            # Position and orientation for this segment
            seg_pos = ensure_list(segment['center'])
            seg_quat = segment['orientation'].to_mujoco_format()
            
            # Create cylinder geometry
            main_body.add(
                "geom",
                type="cylinder",
                size=[segment['radius'], segment['length'] / 2],
                pos=seg_pos,
                quat=seg_quat,
                rgba=[1, 1, 1, 1],
                name=f"lynx_{segment_name}"
            )
        
        # Find the children
        tasks = []

        for child_index, attachment_point in self.attachment_points.items():
            child = self.children.get(child_index)
            if child is not None:
                unbuilt = UnbuiltChild(module=child, mj_body=main_body)
                unbuilt.make_pose(
                    position=self.control_points[-1], # Prev pos + offset rotated in right direction = attachment point pos
                    orientation=Quaternion(),
                )
                tasks.append(unbuilt)

        return tasks


# Helper function to create a simple Bézier curve
def create_simple_bezier_tube(start_point, end_point, control_offset, **kwargs):
    """
    Create a simple cubic Bézier tube with automatic control points
    
    Args:
        start_point: Vector3 - starting point
        end_point: Vector3 - ending point  
        control_offset: Vector3 - offset for control points from start/end
        **kwargs: additional arguments for BezierTube
    """
    control_points = [
        start_point,
        start_point + control_offset,
        end_point - control_offset,
        end_point
    ]
    
    return BezierTube(control_points, **kwargs)


# Example usage:
if __name__ == "__main__":
    # Create a curved tube from (0,0,0) to (1,1,1) with some curvature
    start = Vector3([0.0, 0.0, 0.0])
    end = Vector3([1.0, 1.0, 1.0])
    control_offset = Vector3([0.5, 0.0, 0.5])
    
    curved_tube = create_simple_bezier_tube(
        start_point=start,
        end_point=end,
        control_offset=control_offset,
        num_segments=30,
        cylinder_radius=0.03,
        name="example_curve"
    )