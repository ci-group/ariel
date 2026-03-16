
import mujoco
import numpy as np

from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.math_utils import (
    Quaternion,
    Vector3,
    angle_to_quaternion,
)
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.mj_module import MjModule

from ariel import ROOT, CWD


class MjBSplineTubeWithClamps(MjModule):
    """A B-spline tube for MjSpec that uses STL meshes for the first and last segments (clamps)."""

    def __init__(
        self,
        num_segments: int = 20,
        cylinder_radius: float = 0.05,
        mounting_length_start: float = 0.04,
        mounting_length_end: float = 0.04,
        pre_joint_radius: float = 0.062,
        next_joint_radius: float = 0.042,
        end_point_pos: Vector3 = Vector3([0.0, 0.0, 0.36]),
        end_point_theta: float = 0.0,
        dual_point_distance: float = 0.15,
        name: str = "bspline_tube_with_clamps",
        angle: float = 0.0,
        count_joint_volumes: bool = True,
        color: list[float] | None = None,
        clamp_stl: str = str(CWD / "src/ariel/body_phenotypes/Lynx_Arm/models/clamp_0205.stl"),
        collision_radius: float | None = None,
    ) -> None:
        # MjTube doesn't take color, and we need to provide either cylinder_length or control_points
        # But we want to override the setup anyway.
        super().__init__(name=name, pos=None, quat=None)
        self.color = color if color is not None else [0.15, 0.15, 0.15, 1]
        self.num_segments = num_segments
        self.mounting_length_start = mounting_length_start
        self.mounting_length_end = mounting_length_end
        self.pre_joint_radius = pre_joint_radius
        self.next_joint_radius = next_joint_radius
        self.end_point_pos = end_point_pos
        self.end_point_theta = end_point_theta
        self.dual_point_distance = dual_point_distance
        self.angle = angle
        self.count_joint_volumes = count_joint_volumes
        self.clamp_stl = clamp_stl
        self.cylinder_radius = cylinder_radius
        self.collision_radius = collision_radius if collision_radius is not None else cylinder_radius * 1.2

        # Generate the curve and segments
        self._generate_curve()
        self._build_spec()

    def _end_dir_from_theta_yz(self, theta: float) -> Vector3:
        s, c = np.sin(theta), np.cos(theta)
        return Vector3([0.0, s, c]).normalize()

    def _generate_curve(self) -> None:
        # 1. Set up control points (Cubic Bezier)
        d = float(max(1e-6, self.dual_point_distance))
        t = self._end_dir_from_theta_yz(self.end_point_theta)

        p0 = Vector3([0.0, 0.0, 0.0])
        p1 = p0 + Vector3([0.0, 0.0, 1.0]) * d

        # The end point pos is relative to the start of the tube
        if self.count_joint_volumes:
            start_offset = self.pre_joint_radius + self.mounting_length_start  # along +Z
            end_offset = self.next_joint_radius + self.mounting_length_end
        else:
            start_offset = self.mounting_length_start
            end_offset = self.mounting_length_end

        # p3 = self.end_point_pos
        # p2 = p3 - t * d

        p3 = self.end_point_pos - Vector3([0, 0, 1]) * start_offset - t * end_offset     # <-- offset
        p2 = p3 - t * d

        control_points = [p0, p1, p2, p3]

        # 2. Sample points along Bezier curve
        curve_points = []
        for i in range(self.num_segments + 1):
            u = i / self.num_segments
            # Bernstein polynomials for degree 3
            b0 = (1 - u)**3
            b1 = 3 * u * (1 - u)**2
            b2 = 3 * u**2 * (1 - u)
            b3 = u**3
            pt = control_points[0] * b0 + control_points[1] * b1 + control_points[2] * b2 + control_points[3] * b3
            curve_points.append(pt)

        # 3. Create segments
        self.segments = []

        # Calculate start direction for mounting offset
        start_direction = (curve_points[1] - curve_points[0]).normalize()
        # this means that the first segment will be extended backwards along the curve's initial tangent to the length of the mounting length:
        start_offset_vec = start_direction * (self.mounting_length_start - (curve_points[1] - curve_points[0]).magnitude())

        for i in range(len(curve_points) - 1):
            p_start = curve_points[i]
            p_end = curve_points[i + 1]
            direction = (p_end - p_start).normalize()
            auto_length = (p_end - p_start).magnitude()

            is_first = (i == 0)
            is_last = (i == len(curve_points) - 2)

            if is_first:
                length = self.mounting_length_start
                center = p_start + direction * (length / 2)
            elif is_last:
                length = self.mounting_length_end
                shifted_start = p_start + start_offset_vec
                center = shifted_start + direction * (length / 2)
            else:
                length = auto_length
                shifted_start = p_start + start_offset_vec
                shifted_end = p_end + start_offset_vec
                center = (shifted_start + shifted_end) * 0.5

            # Orientation: rotate Z to direction
            z_axis = Vector3([0, 0, 1])
            dot = z_axis.dot(direction)
            if abs(dot) > 0.9999:
                quat = Quaternion() if dot > 0 else Quaternion([1, 0, 0, 0])
            else:
                axis = z_axis.cross(direction).normalize()
                angle = np.arccos(np.clip(dot, -1.0, 1.0))
                quat = angle_to_quaternion(angle, axis.to_list())

            self.segments.append({
                "center": center,
                "length": length,
                "quat": quat,
                "is_first": is_first,
                "is_last": is_last,
            })

            if is_last:
                shifted_start = p_start + start_offset_vec
                self.actual_end_pos = shifted_start + direction * length
                self.actual_end_quat = quat

        # If no segments were created (shouldn't happen), set defaults
        if not self.segments:
            self.actual_end_pos = p3
            self.actual_end_quat = Quaternion()

    def _build_spec(self) -> None:
        # Clear existing geoms/sites from MjTube.__init__
        self.body.geoms.clear()
        self.body.sites.clear()

        # Add mesh asset to the spec
        mesh_name = str(self.clamp_stl).split("/")[-1].replace(".stl", "")
        # Use self.body.spec if it exists, otherwise we'll have to rely on the caller
        if hasattr(self.body, "spec"):
            self.body.spec.add_mesh(name=mesh_name, file=self.clamp_stl, scale=[0.001, 0.001, 0.001])

        for i, seg in enumerate(self.segments):
            seg_name = f"{self.name}_seg_{i}"

            if seg["is_first"] or seg["is_last"]:
                # Mesh geom
                if seg["is_last"]:
                    # Rotate 180 around local X for "upside down" clamp
                    x_rot_180 = Quaternion.from_axis_angle([1, 0, 0], np.pi)
                    mesh_quat = (seg["quat"] * x_rot_180).to_mujoco_format()
                    mesh_pos = (seg["center"] + seg["quat"] * Vector3([0, 0, seg["length"] / 2])).to_list()
                else:
                    mesh_quat = seg["quat"].to_mujoco_format()
                    mesh_pos = (seg["center"] - seg["quat"] * Vector3([0, 0, seg["length"] / 2])).to_list()

                self.body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_MESH,
                    meshname=mesh_name,
                    pos=mesh_pos,
                    quat=mesh_quat,
                    rgba=self.color,
                    mass=1e-6,
                    contype=0,
                    conaffinity=0,
                    name=f"{seg_name}_visual",
                )
            else:
                # Cylinder geom
                self.body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=[self.cylinder_radius, seg["length"] / 2, 0],
                    pos=seg["center"].to_list(),
                    quat=seg["quat"].to_mujoco_format(),
                    rgba=self.color,
                    mass=1e-6,
                    contype=0,
                    conaffinity=0,
                    name=f"{seg_name}_visual",
                )

            # Collision geom
            self.body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[self.collision_radius, seg["length"] / 2, 0],
                pos=seg["center"].to_list(),
                quat=seg["quat"].to_mujoco_format(),
                rgba=[0, 0, 0, 0],
                mass=0,
                contype=1,
                conaffinity=1,
                group=2,  # GROUP_TUBE
                name=f"{seg_name}_collision",
            )

        # Add attachment site at the end
        # IMPORTANT: The site must be named "attach" for attach_body to find it
        # We use a unique name to avoid "repeated name" errors when multiple tubes are used
        self.body.add_site(
            name=f"{self.name}_attach",
            pos=self.actual_end_pos.to_list(),
            quat=self.actual_end_quat.to_mujoco_format(),
        )

        # Debug print
