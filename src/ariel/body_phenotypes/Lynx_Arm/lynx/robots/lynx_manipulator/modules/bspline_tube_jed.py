import numpy as np
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.module import Module, AttachmentPoint
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.unbuilt_child import UnbuiltChild
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.math_utils import Vector3, Quaternion, ensure_list, angle_to_quaternion

class BSplineTube(Module):
    """
    A tube that follows a B-spline curve, built from multiple cylinder segments
    """
    
    ATTACH = 0  # Attachment point index
    
    def __init__(
        self,
        control_points: list,  # List of Vector3 control points
        degree: int = 3,  # Degree of the B-spline (3 = cubic)
        num_segments: int = 20,
        cylinder_radius: float = 0.05,
        mounting_length_start: float = None,  # Length for mounting cylinder at start (None = automatic)
        mounting_length_end: float = None,  # Length for mounting cylinder at end (None = automatic)
        pre_joint_radius: float = 0.062,
        next_joint_radius: float = 0.042,
        end_point_pos: Vector3 = Vector3([0.0, 0.0, 0.36]),  # Optional end point position constraint
        end_point_ori: Vector3 = Vector3([0.0, 1.0, 0.0]),  # Optional end point orientation constraint
        dual_point_distance: float = 0.15,  # Distance for dual control points when setting end constraints
        knot_vector: list = None,  # Optional custom knot vector
        name: str = "bspline_tube",
        angle: float = 0.0,  # Rotation angle around the attachment axis at the base
    ):
        self.control_points = control_points
        self.degree = min(degree, len(control_points) - 1)  # Degree can't exceed n-1
        self.num_segments = num_segments
        self.cylinder_radius = cylinder_radius
        self.mounting_length_start = mounting_length_start
        self.mounting_length_end = mounting_length_end
        self.name = name
        self.angle = angle

        self.pre_joint_radius = pre_joint_radius
        self.next_joint_radius = next_joint_radius
        self.end_point_pos = end_point_pos
        self.end_point_ori = end_point_ori
        self.dual_point_distance = dual_point_distance
        
        # Generate knot vector if not provided
        if knot_vector is None:
            self.knot_vector = self._generate_uniform_knot_vector()
        else:
            self.knot_vector = knot_vector
            
        # Validate knot vector
        self._validate_knot_vector()
        
        # Calculate curve points and create cylinder segments
        self.curve_points = self._sample_bspline_curve()
        self.cylinder_segments = self._create_cylinder_segments()
        
        # Create attachment points at start and end
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
    
    def _generate_uniform_knot_vector(self):
        """Generate a uniform knot vector for the B-spline"""
        n = len(self.control_points)  # Number of control points
        m = n + self.degree + 1  # Number of knots
        
        knots = []
        
        # First degree+1 knots are 0
        for i in range(self.degree + 1):
            knots.append(0.0)
        
        # Middle knots are uniformly spaced
        for i in range(1, n - self.degree):
            knots.append(float(i) / (n - self.degree))
        
        # Last degree+1 knots are 1
        for i in range(self.degree + 1):
            knots.append(1.0)
            
        return knots
    
    def _validate_knot_vector(self):
        """Validate that the knot vector is correct"""
        n = len(self.control_points)
        expected_length = n + self.degree + 1
        
        if len(self.knot_vector) != expected_length:
            raise ValueError(f"Knot vector length {len(self.knot_vector)} doesn't match expected length {expected_length}")
        
        # Check that knots are non-decreasing
        for i in range(1, len(self.knot_vector)):
            if self.knot_vector[i] < self.knot_vector[i-1]:
                raise ValueError("Knot vector must be non-decreasing")
    
    def _basis_function(self, i, p, u):
        """Calculate the B-spline basis function N_{i,p}(u) using Cox-de Boor recursion"""
        # Base case: degree 0
        if p == 0:
            if self.knot_vector[i] <= u < self.knot_vector[i + 1]:
                return 1.0
            else:
                return 0.0
        
        # Recursive case
        result = 0.0
        
        # First term
        denom1 = self.knot_vector[i + p] - self.knot_vector[i]
        if abs(denom1) > 1e-10:  # Avoid division by zero
            result += (u - self.knot_vector[i]) / denom1 * self._basis_function(i, p - 1, u)
        
        # Second term
        denom2 = self.knot_vector[i + p + 1] - self.knot_vector[i + 1]
        if abs(denom2) > 1e-10:  # Avoid division by zero
            result += (self.knot_vector[i + p + 1] - u) / denom2 * self._basis_function(i + 1, p - 1, u)
        
        return result
    
    def _basis_function_derivative(self, i, p, u):
        """Calculate the derivative of B-spline basis function"""
        if p == 0:
            return 0.0
        
        result = 0.0
        
        # First term
        denom1 = self.knot_vector[i + p] - self.knot_vector[i]
        if abs(denom1) > 1e-10:
            result += p / denom1 * self._basis_function(i, p - 1, u)
        
        # Second term
        denom2 = self.knot_vector[i + p + 1] - self.knot_vector[i + 1]
        if abs(denom2) > 1e-10:
            result -= p / denom2 * self._basis_function(i + 1, p - 1, u)
        
        return result
    
    def _evaluate_bspline(self, u):
        """Evaluate B-spline curve at parameter u"""
        n = len(self.control_points)
        point = Vector3([0.0, 0.0, 0.0])
        
        for i in range(n):
            basis = self._basis_function(i, self.degree, u)
            if abs(basis) > 1e-10:  # Only compute if basis function is significant
                weighted_point = Vector3([
                    self.control_points[i].x * basis,
                    self.control_points[i].y * basis,
                    self.control_points[i].z * basis
                ])
                point = point + weighted_point
        
        return point
    
    def _evaluate_bspline_derivative(self, u):
        """Evaluate first derivative of B-spline curve at parameter u"""
        n = len(self.control_points)
        derivative = Vector3([0.0, 0.0, 0.0])
        
        for i in range(n):
            basis_deriv = self._basis_function_derivative(i, self.degree, u)
            if abs(basis_deriv) > 1e-10:
                weighted_point = Vector3([
                    self.control_points[i].x * basis_deriv,
                    self.control_points[i].y * basis_deriv,
                    self.control_points[i].z * basis_deriv
                ])
                derivative = derivative + weighted_point
        
        return derivative
    
    def _get_parameter_range(self):
        """Get the valid parameter range for the B-spline"""
        return self.knot_vector[self.degree], self.knot_vector[len(self.control_points)]
    
    def _sample_bspline_curve(self):
        """Sample points along the B-spline curve"""
        points = []
        u_min, u_max = self._get_parameter_range()
        
        for i in range(self.num_segments + 1):
            # Map from [0,1] to [u_min, u_max]
            t = i / self.num_segments
            u = u_min + t * (u_max - u_min)
            
            # Clamp u to valid range to avoid numerical issues
            u = max(u_min, min(u_max - 1e-10, u))
            
            point = self._evaluate_bspline(u)
            points.append(point)
        
        return points
    
    def _get_tangent_quaternion(self, index):
        """Get quaternion representing tangent direction at curve point"""
        if index == -1:
            index = len(self.curve_points) - 1
        
        # Calculate tangent using derivative
        u_min, u_max = self._get_parameter_range()
        t = index / len(self.curve_points) if len(self.curve_points) > 1 else 0
        u = u_min + t * (u_max - u_min)
        u = max(u_min, min(u_max - 1e-10, u))
        
        tangent = self._evaluate_bspline_derivative(u)
        
        # Fallback to finite differences if derivative is too small
        if tangent.magnitude() < 1e-6:
            if index == 0 and len(self.curve_points) > 1:
                tangent = self.curve_points[1] - self.curve_points[0]
            elif index == len(self.curve_points) - 1 and len(self.curve_points) > 1:
                tangent = self.curve_points[-1] - self.curve_points[-2]
            elif len(self.curve_points) > 2:
                tangent = self.curve_points[index + 1] - self.curve_points[index - 1]
            else:
                tangent = Vector3([0.0, 0.0, 1.0])
        
        # Normalize tangent
        tangent = tangent.normalize()
        if tangent.magnitude() < 1e-6:
            tangent = Vector3([0.0, 0.0, 1.0])
        
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

        # Calculate offset needed if we have custom mounting lengths
        start_offset = Vector3([0.0, 0.0, 0.0])

        # Calculate start offset if mounting_length_start is set
        if self.mounting_length_start is not None and len(self.curve_points) > 1:
            start_direction = self.curve_points[1] - self.curve_points[0]
            auto_start_length = start_direction.magnitude()
            if auto_start_length > 1e-6:
                start_direction = start_direction.normalize()
                # Offset is the difference between custom and auto length
                start_offset = start_direction * (self.mounting_length_start - auto_start_length)

        for i in range(len(self.curve_points) - 1):
            start_point = self.curve_points[i]
            end_point = self.curve_points[i + 1]

            # Calculate segment properties
            direction = end_point - start_point
            auto_length = direction.magnitude()

            if auto_length > 1e-6:
                direction = direction.normalize()
            else:
                direction = Vector3([0.0, 0.0, 1.0])

            # Determine if this is a mounting segment and override length if needed
            is_first = (i == 0)
            is_last = (i == len(self.curve_points) - 2)

            if is_first and self.mounting_length_start is not None:
                # First cylinder: use custom length, anchor at start
                length = self.mounting_length_start
                center = start_point + direction * (length / 2)
            elif is_last and self.mounting_length_end is not None:
                # Last cylinder: use custom length, start where previous segment ended, extend forward
                length = self.mounting_length_end
                shifted_start = start_point + start_offset
                center = shifted_start + direction * (length / 2)
            else:
                # Middle cylinders: shift by start offset
                length = auto_length
                shifted_start = start_point + start_offset
                shifted_end = end_point + start_offset
                center = (shifted_start + shifted_end) * 0.5

            # Create quaternion for cylinder orientation
            quat = self._vector_to_quaternion(direction)

            segments.append({
                'center': center,
                'length': length,
                'orientation': quat,
                'radius': self.cylinder_radius
            })

            # Track the end position for the last segment
            if is_last:
                if self.mounting_length_end is not None:
                    # End position is start of last cylinder + custom length
                    shifted_start = start_point + start_offset
                    self.actual_end_position = shifted_start + direction * length
                else:
                    # End position is just the shifted end point
                    self.actual_end_position = end_point + start_offset

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
    
    def set_end_constraints(self, end_point_pos: Vector3, end_point_ori: Vector3, dual_point_distance: float):
        """
        Re-parameterize the tube with 4 control points (clamped cubic) so that:
        - start point is (0,0,0) and start tangent is strictly +Z
        - end point is end_point_pos and end tangent follows end_point_ori
        dual_point_distance controls the handle length at both ends.
        """
        d = float(max(1e-6, dual_point_distance))

        # Normalize end direction (fallback to +Z)
        t = end_point_ori
        if t is None or t.magnitude() < 1e-8:
            t = Vector3([0.0, 0.0, 1.0])
        else:
            t = t.normalize()

        p0 = Vector3([0.0, 0.0, 0.0])
        p1 = p0 + Vector3([0.0, 0.0, 1.0]) * d              # strict upward tangent

        start_offset = self.pre_joint_radius + self.mounting_length_start  # along +Z
        end_offset = self.next_joint_radius + self.mounting_length_end
        z_hat = Vector3([0.0, 0.0, 1.0])

        p3 = end_point_pos - z_hat * start_offset - t * end_offset     # <-- offset
        p2 = p3 - t * d
        # p3 = end_point_pos
        # p2 = p3 - t * d                                     # strict end tangent

        self.control_points = [p0, p1, p2, p3]

        # Force clamped cubic (4 control points => cubic Bézier equivalent)
        self.degree = 3
        self.knot_vector = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]

        # Recompute
        self._validate_knot_vector()
        self.curve_points = self._sample_bspline_curve()
        self.cylinder_segments = self._create_cylinder_segments()

        # Update attachment point offset to the exact end (use actual_end_position if mounting changes it)
        end_pos = getattr(self, "actual_end_position", self.control_points[-1])
        if hasattr(self, "attachment_points") and self.ATTACH in self.attachment_points:
            self.attachment_points[self.ATTACH].offset = end_pos

    
    def align_start_with_direction(self, direction_vector):
        """
        Adjust the first few control points to align the curve start with a given direction
        
        Args:
            direction_vector: Vector3 representing the desired starting direction
        """
        if len(self.control_points) < 2:
            return
        
        direction = direction_vector.normalize()
        
        # Calculate the distance to the second control point
        start_to_second = self.control_points[1] - self.control_points[0]
        distance = start_to_second.magnitude()
        
        # Set the second control point to align with the desired direction
        self.control_points[1] = self.control_points[0] + direction * distance
        
        # If we have more control points, adjust the third one for smoother transition
        if len(self.control_points) >= 3:
            # Blend the original direction with the desired direction
            original_dir = self.control_points[2] - self.control_points[0]
            original_distance = original_dir.magnitude()
            
            # 70% desired direction, 30% original for smooth transition
            blended_dir = direction * 0.7 + original_dir.normalize() * 0.3
            blended_dir = blended_dir.normalize()
            
            self.control_points[2] = self.control_points[0] + blended_dir * original_distance

        # print(f"self.control_points: {self.control_points}")
        
        # Recalculate curve points and segments
        self.curve_points = self._sample_bspline_curve()
        self.cylinder_segments = self._create_cylinder_segments()
        
        # Update attachment point orientation
        # self.attachment_points[0].orientation = self._get_tangent_quaternion(0)

    def align_end_with_direction(self, direction_vector):
        """
        Adjust the last few control points to align the curve end with a given direction
        
        Args:
            direction_vector: Vector3 representing the desired ending direction
        """
        if len(self.control_points) < 2:
            return
        
        direction = direction_vector.normalize()
        
        # Calculate the distance to the second-to-last control point
        end_to_second_last = self.control_points[-2] - self.control_points[-1]
        distance = end_to_second_last.magnitude()
        
        # Set the second-to-last control point to align with the desired direction
        self.control_points[-2] = self.control_points[-1] - direction * distance
        
        # If we have more control points, adjust the third-to-last one
        if len(self.control_points) >= 3:
            original_dir = self.control_points[-3] - self.control_points[-1]
            original_distance = original_dir.magnitude()
            
            # Blend directions for smooth transition
            blended_dir = direction * (-0.7) + original_dir.normalize() * 0.3
            blended_dir = blended_dir.normalize()
            
            self.control_points[-3] = self.control_points[-1] + blended_dir * original_distance
        
        # Recalculate curve points and segments
        self.curve_points = self._sample_bspline_curve()
        self.cylinder_segments = self._create_cylinder_segments()

    def build(self, mjcf, entry_point, attachment_point_pos, attachment_point_quat):
        """Build the B-spline tube as multiple cylinder segments"""
        print(f"End constraints provided, re-parameterizing control points")

        # self.set_end_constraints(self.end_point_pos, self.end_point_ori, self.dual_point_distance)
        # Get direction vector from attachment point quat
        vector_part = np.array(attachment_point_quat[:-1])
        q = Quaternion([0.0, -1.0/np.sqrt(2), 0.0, 1.0/np.sqrt(2)])
        direction_vector = q * Vector3(vector_part / np.linalg.norm(vector_part))
        # direction_vector = q * Vector3(vector_part)

        # direction_vector = attachment_point_quat * Vector3([0.0, 0.0, 1.0])

        self.align_start_with_direction(direction_vector)  # Align start with z-axis

        pos_list = ensure_list(attachment_point_pos)
        # attachment_point_quat = attachment_point_quat.normalise()
        # Calculate the axis given the attachment point quaternion
        axis = attachment_point_quat * Vector3([0, 0, 1])
        q = angle_to_quaternion(self.angle, axis.to_list())  # Rotate around the calculated axis
        new_quat = q * attachment_point_quat

        mj_quat = new_quat.to_mujoco_format()
        # mj_quat = Quaternion([0,0,0,1]).to_mujoco_format()  # No rotation at base

        # Create main body for the B-spline tube
        main_body = entry_point.add("body", name=self.name, pos=pos_list, quat=mj_quat)
        
        # Add each cylinder segment
        for i, segment in enumerate(self.cylinder_segments):
            segment_name = f"{self.name}_seg_{i}"
            
            # Position and orientation for this segment
            seg_pos = ensure_list(segment['center'])
            seg_quat = segment['orientation'].to_mujoco_format()

            # print(f"[tube] seg_quat: {seg_quat}, length: {segment['length']}")
            
            # Create cylinder geometry
            main_body.add(
                "geom",
                type="cylinder",
                size=[segment['radius'], segment['length'] / 2],
                pos=seg_pos,
                quat=seg_quat,
                rgba=[1., 0.745, 0.298, 1],
                name=f"lynx_{segment_name}"
            )

            last_quat = segment['orientation']

        # Find the children
        tasks = []

        for child_index, attachment_point in self.attachment_points.items():
            child = self.children.get(child_index)
            if child is not None:
                unbuilt = UnbuiltChild(module=child, mj_body=main_body)
                unbuilt.make_pose(
                    position=self.actual_end_position, # Use actual end position accounting for mounting lengths
                    orientation=last_quat,
                )
                tasks.append(unbuilt)

        return tasks

# Helper functions for creating common B-spline configurations

def create_uniform_bspline_tube(control_points, degree=3, **kwargs):
    """
    Create a B-spline tube with uniform knot spacing
    
    Args:
        control_points: List of Vector3 control points
        degree: Degree of the B-spline (default: 3 for cubic)
        **kwargs: Additional arguments for BSplineTube
    """
    return BSplineTube(control_points, degree=degree, **kwargs)

def create_clamped_bspline_tube(control_points, degree=3, **kwargs):
    """
    Create a clamped B-spline tube that passes through the first and last control points
    
    Args:
        control_points: List of Vector3 control points
        degree: Degree of the B-spline (default: 3 for cubic)
        **kwargs: Additional arguments for BSplineTube
    """
    n = len(control_points)
    degree = min(degree, n - 1)
    
    # Create clamped knot vector
    knot_vector = []
    
    # First degree+1 knots are 0 (clamping at start)
    for i in range(degree + 1):
        knot_vector.append(0.0)
    
    # Middle knots
    for i in range(1, n - degree):
        knot_vector.append(float(i) / (n - degree))
    
    # Last degree+1 knots are 1 (clamping at end)
    for i in range(degree + 1):
        knot_vector.append(1.0)
    
    return BSplineTube(control_points, degree=degree, knot_vector=knot_vector, **kwargs)

def create_smooth_path_bspline(waypoints, smoothness=0.3, degree=3, **kwargs):
    """
    Create a smooth B-spline path through waypoints with automatic control point generation
    
    Args:
        waypoints: List of Vector3 waypoints to pass near
        smoothness: How much to smooth the path (0.0 = sharp corners, 1.0 = very smooth)
        degree: Degree of the B-spline
        **kwargs: Additional arguments for BSplineTube
    """
    if len(waypoints) < 2:
        raise ValueError("Need at least 2 waypoints")
    
    # Generate control points based on waypoints
    control_points = [waypoints[0]]  # Start with first waypoint
    
    for i in range(1, len(waypoints) - 1):
        prev_point = waypoints[i - 1]
        curr_point = waypoints[i]
        next_point = waypoints[i + 1]
        
        # Create control points around each waypoint for smoothing
        approach_dir = curr_point - prev_point
        departure_dir = next_point - curr_point
        
        # Add control points before and after waypoint
        control_points.append(curr_point - approach_dir * smoothness)
        control_points.append(curr_point)
        control_points.append(curr_point + departure_dir * smoothness)
    
    control_points.append(waypoints[-1])  # End with last waypoint
    
    return create_clamped_bspline_tube(control_points, degree=degree, **kwargs)


# Example usage:
if __name__ == "__main__":
    # Example 1: Simple cubic B-spline
    control_points = [
        Vector3([0.0, 0.0, 0.0]),
        Vector3([0.5, 0.5, 0.2]),
        Vector3([1.0, 0.2, 0.5]),
        Vector3([1.5, 0.8, 0.8]),
        Vector3([2.0, 1.0, 1.0])
    ]
    
    bspline_tube = create_clamped_bspline_tube(
        control_points=control_points,
        degree=3,
        num_segments=40,
        cylinder_radius=0.025,
        name="example_bspline"
    )
    
    # Example 2: Smooth path through waypoints
    waypoints = [
        Vector3([0.0, 0.0, 0.0]),
        Vector3([1.0, 1.0, 0.5]),
        Vector3([2.0, 0.5, 1.0]),
        Vector3([3.0, 2.0, 1.5])
    ]
    
    smooth_tube = create_smooth_path_bspline(
        waypoints=waypoints,
        smoothness=0.4,
        degree=3,
        num_segments=50,
        cylinder_radius=0.03,
        name="smooth_path"
    )