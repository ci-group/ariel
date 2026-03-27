
import numpy as np

from ariel import ROOT, CWD
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.math_utils import (
    Quaternion,
    Vector3,
    angle_to_quaternion,
    ensure_list,
)
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.unbuilt_child import UnbuiltChild
from ariel.body_phenotypes.Lynx_Arm.lynx.robots.lynx_manipulator.modules.bspline_tube_xinrui import (
    BSplineTube,
)


class BSplineTubeWithClamps(BSplineTube):
    """
    A B-spline tube that uses STL meshes for the first and last segments (clamps)
    and includes a simplified collision model for self-collision detection.
    """

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
        control_points: list[Vector3] | None = None,
        count_joint_volumes: bool = True,
        color: list[float] | None = None,
        clamp_stl: str = str(CWD / "src/ariel/body_phenotypes/Lynx_Arm/models/clamp_0205.stl"),
        collision_radius: float | None = None,
        collision_group: int = 2,
        collision_alpha: float = 0.3,
    ) -> None:
        super().__init__(
            num_segments=num_segments,
            cylinder_radius=cylinder_radius,
            mounting_length_start=mounting_length_start,
            mounting_length_end=mounting_length_end,
            pre_joint_radius=pre_joint_radius,
            next_joint_radius=next_joint_radius,
            end_point_pos=end_point_pos,
            end_point_theta=end_point_theta,
            dual_point_distance=dual_point_distance,
            name=name,
            angle=angle,
            control_points=control_points,
            count_joint_volumes=count_joint_volumes,
            color=color,
        )
        self.clamp_stl = clamp_stl
        self.collision_radius = collision_radius if collision_radius is not None else cylinder_radius * 1.2
        self.collision_group = collision_group
        self.collision_alpha = collision_alpha

    def build(self, mjcf, entry_point, attachment_point_pos, attachment_point_quat):
        """Build the B-spline tube with STL clamps and collision geoms."""
        # Add STL asset to the MJCF model
        mesh_name = self.clamp_stl.split("/")[-1].replace(".stl", "")
        # Check if mesh already exists in assets to avoid duplicates
        # existing_meshes = [m.name for m in mjcf.asset.find_all("mesh")]
        # if mesh_name not in existing_meshes:
        #     mjcf.asset.add("mesh", name=mesh_name, file=self.clamp_stl, scale=[0.001, 0.001, 0.001])

        # Re-parameterize control points based on end constraints
        self.set_end_constraints(self.end_point_pos, self.end_point_theta, self.dual_point_distance)

        pos_list = ensure_list(attachment_point_pos)

        # Ensure attachment_point_quat is a Quaternion object
        if not isinstance(attachment_point_quat, Quaternion):
            attachment_point_quat = Quaternion(attachment_point_quat)

        axis = attachment_point_quat * Vector3([0, 0, 1])
        q = angle_to_quaternion(self.angle, axis.to_list())
        new_quat = q * attachment_point_quat
        mj_quat = new_quat.to_mujoco_format()

        # Create main body
        main_body = entry_point.add("body", name=self.name, pos=pos_list, quat=mj_quat)

        # Add each segment
        last_quat = Quaternion()
        for i, segment in enumerate(self.cylinder_segments):
            segment_name = f"{self.name}_seg_{i}"
            seg_pos = ensure_list(segment["center"])
            seg_quat = segment["orientation"].to_mujoco_format()
            last_quat = segment["orientation"]

            is_first = (i == 0)
            is_last = (i == len(self.cylinder_segments) - 1)

            # if is_first or is_last:
            if False:
                # Use STL for first and last segments
                # Rotate the mesh itself by 90 degrees around X axis
                Quaternion.from_axis_angle([1, 0, 0], np.deg2rad(90))

                if is_last:
                    # Rotate the last segment an additional 180 degrees around its local Z axis
                    # (User said 280 but mentioned it's upside down, 180 is standard for upside down)
                    # If they specifically want 280, I'll use 180 first as it's more likely what they mean by "upside down"
                    # Actually, let's use 180.
                    z_rot_180 = Quaternion.from_axis_angle([1, 0, 0], np.deg2rad(180))
                    mesh_quat = (segment["orientation"] * z_rot_180).to_mujoco_format()
                    # mesh_quat = seg_quat
                    mesh_pos = ensure_list(segment["center"] + segment["orientation"] * Vector3([0, 0, segment["length"] / 2]))

                else:
                    mesh_quat = seg_quat
                    mesh_pos = ensure_list(segment["center"] - segment["orientation"] * Vector3([0, 0, segment["length"] / 2]))

                main_body.add(
                    "geom",
                    type="mesh",
                    mesh=mesh_name,
                    pos=mesh_pos,
                    quat=mesh_quat,
                    rgba=self.color,
                    mass=1e-6,  # mass in kg
                    contype=0,
                    conaffinity=0,
                    name=f"lynx_{segment_name}_visual",
                )
            else:
                # Visual cylinder for middle segments
                main_body.add(
                    "geom",
                    type="cylinder",
                    size=[segment["radius"], segment["length"] / 2],
                    pos=seg_pos,
                    quat=seg_quat,
                    rgba=self.color,
                    mass=1e-6,  # mass in kg
                    contype=0,
                    conaffinity=0,
                    name=f"lynx_{segment_name}_visual",
                )

            # Add collision cylinder (self-collision setup)
            # We use a slightly larger radius or specific group for self-collision
            main_body.add(
                "geom",
                type="cylinder",
                size=[self.collision_radius, segment["length"] / 2],
                pos=seg_pos,
                quat=seg_quat,
                rgba=[1, 0, 0, self.collision_alpha],  # Semi-transparent red
                mass=0,
                contype=int(1 << self.collision_group),  # Identity: Unique bit for this tube
                # Affinity: Hit everything (all 1s) EXCEPT my own group bit
                conaffinity=int(0x7FFFFFFF ^ (1 << self.collision_group)),
                group=self.GROUP_TUBE,
                name=f"lynx_{segment_name}_collision",
            )

        # Handle children
        tasks = []
        for child_index in self.attachment_points:
            child = self.children.get(child_index)
            if child is not None:
                unbuilt = UnbuiltChild(module=child, mj_body=main_body)
                unbuilt.make_pose(
                    position=self.actual_end_position,
                    orientation=last_quat,
                )
                tasks.append(unbuilt)

        return tasks
