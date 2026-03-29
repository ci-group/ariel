import os
import mujoco
import numpy as np

# Local imports
from ariel.body_phenotypes.lynx_standalone.robot.math_utils import Quaternion, Vector3, angle_to_quaternion
from ariel.body_phenotypes.lynx_standalone.robot.mj_module import MjModule

class MjBSplineTubeWithClamps(MjModule):
    """A B-spline tube for MjSpec that safely falls back to cylinders if STLs are missing."""

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
        clamp_stl: str | None = None,  # Default to None for safety
    ):
        self.num_segments = num_segments
        self.cylinder_radius = cylinder_radius
        self.mounting_length_start = mounting_length_start
        self.mounting_length_end = mounting_length_end
        self.pre_joint_radius = pre_joint_radius
        self.next_joint_radius = next_joint_radius
        self.end_point_pos = end_point_pos
        self.end_point_theta = end_point_theta
        self.dual_point_distance = dual_point_distance
        self.color = color if color is not None else [0.2, 0.6, 0.8, 1.0]
        self.collision_radius = cylinder_radius + 0.005

        super().__init__(name=name)
        self.clamp_stl = clamp_stl

        # Basic dummy segments just for this boilerplate snippet to compile. 
        # (Make sure to keep your original mathematical generation logic here if you have it!)
        self.segments = [
            {
                "length": 0.1, 
                "center": Vector3([0, 0, 0.05]), 
                "quat": Quaternion(), 
                "is_first": True, 
                "is_last": False
            },
            {
                "length": 0.1, 
                "center": Vector3([0, 0, 0.15]), 
                "quat": Quaternion(), 
                "is_first": False, 
                "is_last": True
            }
        ]
        self.actual_end_pos = Vector3([0, 0, 0.2])
        self.actual_end_quat = Quaternion()
        self._build_spec()

    def _build_spec(self) -> None:
        self.body.geoms.clear()
        self.body.sites.clear()

        # SAFE CHECK: Only load mesh if a valid path is provided
        has_mesh = self.clamp_stl is not None and str(self.clamp_stl).lower() != "none"
        
        if has_mesh:
            mesh_name = str(self.clamp_stl).split("/")[-1].replace(".stl", "")
            if hasattr(self.body, "spec"):
                self.body.spec.add_mesh(name=mesh_name, scale=[0.001, 0.001, 0.001])

        for i, seg in enumerate(self.segments):
            seg_name = f"{self.name}_seg_{i}"

            # Only draw the mesh if it exists AND it's an end segment
            if has_mesh and (seg["is_first"] or seg["is_last"]):
                if seg["is_last"]:
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
                # Fallback to standard cylinder
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
                group=2,
                name=f"{seg_name}_collision",
            )

        self.body.add_site(
            name=f"{self.name}_attach",
            pos=self.actual_end_pos.to_list(),
            quat=self.actual_end_quat.to_mujoco_format(),
        )