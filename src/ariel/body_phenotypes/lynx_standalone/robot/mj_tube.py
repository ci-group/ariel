import mujoco
import numpy as np
from typing import List, Optional
from ariel.body_phenotypes.lynx_standalone.robot.mj_module import MjModule
from ariel.body_phenotypes.lynx_standalone.robot.math_utils import Vector3, Quaternion, angle_to_quaternion

class MjTube(MjModule):
    """
    Tube module supporting both straight and B-spline configurations using MjSpec.
    """
    def __init__(
        self,
        # Straight tube params
        cylinder_length: Optional[float] = None,
        # B-spline params
        control_points: Optional[List[Vector3]] = None,
        degree: int = 3,
        num_segments: int = 20,
        # Common params
        cylinder_radius: float = 0.05,
        name: str = "tube",
        pos=None,
        quat=None
    ):
        super().__init__(name=name, pos=pos, quat=quat)
        self.cylinder_radius = cylinder_radius

        if control_points is not None:
            self._setup_bspline(control_points, degree, num_segments)
        elif cylinder_length is not None:
            self._setup_straight(cylinder_length)
        else:
            raise ValueError("Either cylinder_length or control_points must be provided.")

    def _setup_straight(self, length: float):
        # Add straight geom
        self.body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[self.cylinder_radius, length / 2, 0],
            pos=[0, 0, length / 2],
            rgba=[0.15, 0.15, 0.15, 1],
            group=1
        )
        # Add attachment site at the end
        self.body.add_site(name="attach", pos=[0, 0, length])

    def _setup_bspline(self, control_points: List[Vector3], degree: int, num_segments: int):
        # ... (sampling logic)
        points = self._sample_bspline(control_points, degree, num_segments)
        
        last_quat = [1, 0, 0, 0]
        for i in range(len(points) - 1):
            p0 = points[i]
            p1 = points[i+1]
            center = (p0 + p1) * 0.5
            diff = p1 - p0
            length = diff.magnitude()
            if length > 1e-6:
                direction = diff.normalize()
                # Vector to quat logic
                z_axis = Vector3([0, 0, 1])
                dot = z_axis.dot(direction)
                if abs(dot) > 0.9999:
                    quat = [1, 0, 0, 0] if dot > 0 else [0, 1, 0, 0]
                else:
                    axis = z_axis.cross(direction).normalize()
                    angle = np.arccos(np.clip(dot, -1.0, 1.0))
                    quat = angle_to_quaternion(angle, axis.to_list()).to_mujoco_format()
                
                self.body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=[self.cylinder_radius, length / 2, 0],
                    pos=center.to_list(),
                    quat=quat,
                    rgba=[0.15, 0.15, 0.15, 1],
                    group=1
                )
                last_quat = quat

        # Attachment site at the last control point
        self.body.add_site(name="attach", pos=control_points[-1].to_list(), quat=last_quat)

    def _sample_bspline(self, control_points, degree, num_segments) -> List[Vector3]:
        # Minimal clamped B-spline sampling
        n = len(control_points)
        degree = min(degree, n - 1)
        knots = ([0.0] * (degree + 1)) + \
                [float(i) / (n - degree) for i in range(1, n - degree)] + \
                ([1.0] * (degree + 1))
        
        def basis(i, p, u):
            if p == 0:
                return 1.0 if knots[i] <= u < knots[i+1] else 0.0
            res = 0.0
            d1 = knots[i+p] - knots[i]
            if d1 > 1e-10:
                res += (u - knots[i]) / d1 * basis(i, p-1, u)
            d2 = knots[i+p+1] - knots[i+1]
            if d2 > 1e-10:
                res += (knots[i+p+1] - u) / d2 * basis(i+1, p-1, u)
            return res

        points = []
        for j in range(num_segments + 1):
            u = j / num_segments
            if u >= 1.0: u = 1.0 - 1e-10
            p = Vector3([0, 0, 0])
            for i in range(n):
                b = basis(i, degree, u)
                p = p + control_points[i] * b
            points.append(p)
        # Ensure last point is exactly the last control point
        points[-1] = control_points[-1]
        return points
