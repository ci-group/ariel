import numpy as np

from ariel import ROOT, CWD
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.math_utils import Vector3

from .modules.base import Base
from .modules.bspline_tube_with_clamps import BSplineTubeWithClamps
from .modules.cup import CupModule
from .modules.end_effector import EndEffector
from .modules.joints import JointInline, JointOrthogonal
from .modules.straight_tube import StraightTube

CONTROL_MODE = "position"  # Default control mode for joints


def construct(
    robot_description_dict: dict | None = None,
    ):
    """
    Builds the robot arm structure from a compact dict containing all the parameter values.

    Parameters
    ----------
    params : dict
        Dictionary with the keys described in the module docstring:
          - "genotype_tube": list[int]
          - "genotype_joints": int
          - "tube_lengths": list[float]
          - "rotation_angles": list[float] (len == 6, radians)
          - "robot_description_dict": dict with {"control_mode": "..."} (optional)
          - "collision_alpha": float (optional, default 0.3)

    Returns
    -------
    Base
        The root `Base` module. Its `.attach` chain encodes joints,
        tubes and the end-effector.
    """
    if robot_description_dict is None:
        robot_description_dict: dict = {
            # morphology:
            "num_joints": 6,  # default: 6. range: {1,2,3,4,5,6}
            # tube order:
            "genotype_tube": [0, 1, 0, 1, 0],  # default: [0, 1, 0, 1, 0]
            # tube 1:
            "l1_pre_joint_radius": 0.062,
            "l1_next_joint_radius": 0.042,
            "l1_end_point_pos": [0.0, 0.0, 0.2805],
            "l1_end_point_theta": 0.0,  # deg, rotation in YZ plane from +Z
            # tube 2:
            "l2_pre_joint_radius": 0.042,
            "l2_next_joint_radius": 0.042,
            "l2_end_point_pos": [0.0, 0.0, 0.2805],
            "l2_end_point_theta": 0.0,  # deg, rotation in YZ plane from +Z
            # tube 3:
            "l3_pre_joint_radius": 0.042,
            "l3_next_joint_radius": 0.042,
            "l3_end_point_pos": [0.0, 0.0, 0.36],
            "l3_end_point_theta": 0.0,  # deg, rotation in YZ plane from +Z
            # tube 4:
            "l4_pre_joint_radius": 0.042,
            "l4_next_joint_radius": 0.042,
            "l4_end_point_pos": [0.0, 0.0, 0.36],
            "l4_end_point_theta": 0.0,  # deg, rotation in YZ plane from +Z
            # tube 5:
            "l5_pre_joint_radius": 0.042,
            "l5_next_joint_radius": 0.042,
            "l5_end_point_pos": [0.0, 0.0, 0.36],
            "l5_end_point_theta": 0.0,  # deg, rotation in YZ plane from +Z
            # joint type:
            "genotype_joints": 2,  # default: 1. range: {1,2,3,4,5}
            # fixed-side rotation angles:
            "rotation_angles": [180, 0, 0, -180, 0, 0],  # default: [np.pi, 0, 0, -np.pi, 0, 0]
            # ===== other static params =====:
            "clamp_length": 0.051,  # version: models/clamp_1216.stl
            "dual_point_distance": 0.15,
            "num_segments": 100,
            "tube_radiuses": [0.0396, 0.0396, 0.0396, 0.0396, 0.0396],
            "tube_colors": [
                [0.085, 0.394, 0.350, 1],
                [0.085, 0.394, 0.350, 1],
                [0.085, 0.394, 0.350, 1],
                [0.085, 0.394, 0.350, 1],
                [0.085, 0.394, 0.350, 1],
                ],  # black: [0.1, 0.1, 0.1, 1.0]  gray blue: [0.68, 0.76, 0.82, 1.0]  orange: [1., 0.745, 0.298, 1] dark green: [0.085, 0.394, 0.350, 1]
            "clamp_stl": str(CWD / "src/ariel/body_phenotypes/Lynx_Arm/models/clamp_0226.stl"),
            "task": "reach",  # default: "reach". range: {"reach", "ball in cup", "push"}
            "collision_alpha": 0.3,
            }
    genotype_tube = robot_description_dict["genotype_tube"]
    genotype_joints = robot_description_dict["genotype_joints"]
    rotation_angles = robot_description_dict["rotation_angles"]
    collision_alpha = robot_description_dict.get("collision_alpha", 0.3)
    num_joints = robot_description_dict.get("num_joints", 6)

    def _fixed_from_angle(j_index: int, angle: float) -> float:
        return angle

    # costruct the list for joint type, True means inline,  False means orthogonal
    joints = [True, True, True, True, True]
    joints[genotype_joints:] = [False] * (len(joints) - genotype_joints)

    # Limit joints and tubes based on num_joints
    # num_joints is between 1 and 6.
    # There are num_joints joints and num_joints - 1 potential tube slots.
    genotype_tube = genotype_tube[:num_joints - 1]
    rotation_angles = rotation_angles[:num_joints]
    joints = joints[:num_joints - 1]  # joints[0] corresponds to j2, so we need num_joints-1 entries for j2..j6

    # Define the Base and 1 joint - nonmodifiable by evolution
    root = Base(
        base_length1=0.001,
        base_radius1=0.08,
        base_length2=0.017 + 0.001,  # offset for the base
        base_radius2=0.08,
        name="lynx_base",
    )

    j1 = JointInline(
        cylinder_length0=0.036,
        cylinder_radius0=0.058,
        cylinder_length1=0.128,
        cylinder_radius1=0.062,
        cylinder_length2=0.013,
        cylinder_radius2=0.062,
        angle=np.deg2rad(rotation_angles[0]),
        name="joint1",
        control_mode="position",
        armature=0.01, damping=100000, frictionloss=0.000001,
        collision_type="cylinder",
        collision_alpha=collision_alpha,
    )

    # Define joints
    # j2
    if num_joints >= 2:
        JointType2 = JointInline if joints[0] else JointOrthogonal
        j2_kwargs = {
            "cylinder_length0": 0.036,
            "cylinder_radius0": 0.058,
            "cylinder_length1": 0.128,
            "cylinder_radius1": 0.062,
            "cylinder_length2": 0.013,
            "cylinder_radius2": 0.062,
            "angle": np.deg2rad(rotation_angles[1]),
            "name": "joint2",
            "control_mode": "position",
            "armature": 0.01, "damping": 100000, "frictionloss": 0.000001,
            "collision_type": "cylinder",
            "collision_alpha": collision_alpha,
        }
        j2 = JointType2(**j2_kwargs)
    else:
        j2 = None

    # j3
    if num_joints >= 3:
        JointType3 = JointInline if joints[1] else JointOrthogonal
        j3_kwargs = {
            "cylinder_length0": 0.029,
            "cylinder_radius0": 0.04,
            "cylinder_length1": 0.092,
            "cylinder_radius1": 0.042,
            "cylinder_length2": 0.008,
            "cylinder_radius2": 0.042,
            "angle": np.deg2rad(rotation_angles[2]),
            "name": "joint3",
            "control_mode": "position",
            "armature": 0.01, "damping": 100000, "frictionloss": 0.000001,
            "collision_alpha": collision_alpha,
        }
        j3 = JointType3(**j3_kwargs)
    else:
        j3 = None

    # j4
    if num_joints >= 4:
        JointType4 = JointInline if joints[2] else JointOrthogonal
        j4_kwargs = {
            "cylinder_length0": 0.029,
            "cylinder_radius0": 0.04,
            "cylinder_length1": 0.092,
            "cylinder_radius1": 0.042,
            "cylinder_length2": 0.008,
            "cylinder_radius2": 0.042,
            "angle": np.deg2rad(rotation_angles[3]),
            "name": "joint4",
            "control_mode": "position",
            "armature": 0.01, "damping": 100000, "frictionloss": 0.000001,
            "collision_alpha": collision_alpha,
        }
        j4 = JointType4(**j4_kwargs)
    else:
        j4 = None

    # j5
    if num_joints >= 5:
        JointType5 = JointInline if joints[3] else JointOrthogonal
        j5_kwargs = {
            "cylinder_length0": 0.029,
            "cylinder_radius0": 0.04,
            "cylinder_length1": 0.096,
            "cylinder_radius1": 0.042,
            "cylinder_length2": 0.008,
            "cylinder_radius2": 0.042,
            "angle": np.deg2rad(rotation_angles[4]),
            "name": "joint5",
            "control_mode": "position",
            "armature": 0.01, "damping": 100000, "frictionloss": 0.000001,
            "collision_alpha": collision_alpha,
        }
        j5 = JointType5(**j5_kwargs)
    else:
        j5 = None

    # j6
    if num_joints >= 6:
        JointType6 = JointInline if joints[4] else JointOrthogonal
        j6_kwargs = {
            "cylinder_length0": 0.029,
            "cylinder_radius0": 0.04,
            "cylinder_length1": 0.096,
            "cylinder_radius1": 0.042,
            "cylinder_length2": 0.008,
            "cylinder_radius2": 0.042,
            "angle": np.deg2rad(rotation_angles[5]),  # np.pi
            "name": "joint6",
            "control_mode": "position",
            "armature": 0.01, "damping": 100000, "frictionloss": 0.000001,
            "collision_alpha": collision_alpha,
        }
        j6 = JointType6(**j6_kwargs)
    else:
        j6 = None

    # Define straight tubes and end part
    if num_joints >= 2:
        tube1 = BSplineTubeWithClamps(
            num_segments=robot_description_dict.get("num_segments", 100),
            cylinder_radius=robot_description_dict.get("tube_radiuses", [0.0396, 0.0396, 0.0396, 0.0396, 0.0396])[0],  # 0.0416 0.0396
            mounting_length_start=robot_description_dict.get("clamp_length", 0.045),
            mounting_length_end=robot_description_dict.get("clamp_length", 0.045),
            pre_joint_radius=robot_description_dict.get("l1_pre_joint_radius", 0.062),
            next_joint_radius=robot_description_dict.get("l1_next_joint_radius", 0.042),
            end_point_pos=Vector3(list(robot_description_dict.get("l1_end_point_pos", [0.0, 0.0, 0.36]))),  # Optional end point position constraint
            end_point_theta=np.deg2rad(robot_description_dict.get("l1_end_point_theta", 0.0)),  # rad, rotation in YZ plane from +Z
            dual_point_distance=robot_description_dict.get("dual_point_distance", 0.15),  # Distance for dual control points when setting end constraints
            name="tube1",
            angle=np.deg2rad(robot_description_dict.get("joint2_rotation_angle", 0.0)),  # Default to 0.0 if not provided
            count_joint_volumes=False,
            color=robot_description_dict.get("tube_colors", [[0.1, 0.1, 0.1, 1.0], [0.1, 0.1, 0.1, 1.0], [0.1, 0.1, 0.1, 1.0], [0.1, 0.1, 0.1, 1.0], [0.1, 0.1, 0.1, 1.0]])[0],  # gray blue: [0.68, 0.76, 0.82, 1.0]  orange: [1., 0.745, 0.298, 1] dark green: [0.085, 0.394, 0.350, 1]
            clamp_stl=robot_description_dict.get("clamp_stl", str(CWD / "src/ariel/body_phenotypes/Lynx_Arm/models/clamp_0226.stl")),
            collision_group=2,
            collision_alpha=collision_alpha,
        )
    else:
        tube1 = None

    if num_joints >= 3:
        tube2 = BSplineTubeWithClamps(
            num_segments=robot_description_dict.get("num_segments", 100),
            cylinder_radius=robot_description_dict.get("tube_radiuses", [0.0396, 0.0396, 0.0396, 0.0396, 0.0396])[1],  # 0.0416 0.0396
            mounting_length_start=robot_description_dict.get("clamp_length", 0.045),
            mounting_length_end=robot_description_dict.get("clamp_length", 0.045),
            pre_joint_radius=robot_description_dict.get("l2_pre_joint_radius", 0.042),
            next_joint_radius=robot_description_dict.get("l2_next_joint_radius", 0.042),
            end_point_pos=Vector3(list(robot_description_dict.get("l2_end_point_pos", [0.0, 0.0, 0.2]))),  # Optional end point position constraint
            end_point_theta=np.deg2rad(robot_description_dict.get("l2_end_point_theta", 0.0)),  # rad, rotation in YZ plane from +Z
            dual_point_distance=robot_description_dict.get("dual_point_distance", 0.15),  # Distance for dual control points when setting end constraints
            name="tube2",
            angle=0,  # Default to 0.0 if not provided
            count_joint_volumes=False,
            color=robot_description_dict.get("tube_colors", [[0.1, 0.1, 0.1, 1.0], [0.1, 0.1, 0.1, 1.0], [0.1, 0.1, 0.1, 1.0], [0.1, 0.1, 0.1, 1.0], [0.1, 0.1, 0.1, 1.0]])[1],  # gray blue: [0.68, 0.76, 0.82, 1.0]  orange: [1., 0.745, 0.298, 1] dark green: [0.085, 0.394, 0.350, 1]
            clamp_stl=robot_description_dict.get("clamp_stl", str(CWD / "src/ariel/body_phenotypes/Lynx_Arm/models/clamp_0226.stl")),
            collision_group=3,
            collision_alpha=collision_alpha,
        )
    else:
        tube2 = None

    if num_joints >= 4:
        tube3 = BSplineTubeWithClamps(
            num_segments=robot_description_dict.get("num_segments", 100),
            cylinder_radius=robot_description_dict.get("tube_radiuses", [0.0396, 0.0396, 0.0396, 0.0396, 0.0396])[2],
            mounting_length_start=robot_description_dict.get("clamp_length", 0.0359),
            mounting_length_end=robot_description_dict.get("clamp_length", 0.0359),
            pre_joint_radius=robot_description_dict.get("l3_pre_joint_radius", 0.042),
            next_joint_radius=robot_description_dict.get("l3_next_joint_radius", 0.042),
            end_point_pos=Vector3(list(robot_description_dict.get("l3_end_point_pos", [0.0, 0.0, 0.36]))),
            end_point_theta=np.deg2rad(robot_description_dict.get("l3_end_point_theta", 0.0)),
            dual_point_distance=robot_description_dict.get("dual_point_distance", 0.07),
            name="tube3",
            angle=0.0,
            count_joint_volumes=False,
            color=robot_description_dict.get("tube_colors", [[0.1, 0.1, 0.1, 1.0], [0.1, 0.1, 0.1, 1.0], [0.1, 0.1, 0.1, 1.0], [0.1, 0.1, 0.1, 1.0], [0.1, 0.1, 0.1, 1.0]])[2],  # gray blue: [0.68, 0.76, 0.82, 1.0]  orange: [1., 0.745, 0.298, 1] dark green: [0.085, 0.394, 0.350, 1]
            clamp_stl=robot_description_dict.get("clamp_stl", str(CWD / "src/ariel/body_phenotypes/Lynx_Arm/models/clamp_0226.stl")),
            collision_group=4,
            collision_alpha=collision_alpha,
        )
    else:
        tube3 = None

    if num_joints >= 5:
        tube4 = BSplineTubeWithClamps(
            num_segments=robot_description_dict.get("num_segments", 100),
            cylinder_radius=robot_description_dict.get("tube_radiuses", [0.0396, 0.0396, 0.0396, 0.0396, 0.0396])[3],
            mounting_length_start=robot_description_dict.get("clamp_length", 0.0359),
            mounting_length_end=robot_description_dict.get("clamp_length", 0.0359),
            pre_joint_radius=robot_description_dict.get("l4_pre_joint_radius", 0.042),
            next_joint_radius=robot_description_dict.get("l4_next_joint_radius", 0.042),
            end_point_pos=Vector3(list(robot_description_dict.get("l4_end_point_pos", [0.0, 0.0, 0.36]))),
            end_point_theta=np.deg2rad(robot_description_dict.get("l4_end_point_theta", 0.0)),
            dual_point_distance=robot_description_dict.get("dual_point_distance", 0.07),
            name="tube4",
            angle=0.0,
            count_joint_volumes=False,
            color=robot_description_dict.get("tube_colors", [[0.1, 0.1, 0.1, 1.0], [0.1, 0.1, 0.1, 1.0], [0.1, 0.1, 0.1, 1.0], [0.1, 0.1, 0.1, 1.0], [0.1, 0.1, 0.1, 1.0]])[3],  # gray blue: [0.68, 0.76, 0.82, 1.0]  orange: [1., 0.745, 0.298, 1] dark green: [0.085, 0.394, 0.350, 1]
            clamp_stl=robot_description_dict.get("clamp_stl", str(CWD / "src/ariel/body_phenotypes/Lynx_Arm/models/clamp_0226.stl")),
            collision_group=5,
            collision_alpha=collision_alpha,
        )
    else:
        tube4 = None

    if num_joints >= 6:
        tube5 = BSplineTubeWithClamps(
            num_segments=robot_description_dict.get("num_segments", 100),
            cylinder_radius=robot_description_dict.get("tube_radiuses", [0.0396, 0.0396, 0.0396, 0.0396, 0.0396])[4],
            mounting_length_start=robot_description_dict.get("clamp_length", 0.0359),
            mounting_length_end=robot_description_dict.get("clamp_length", 0.0359),
            pre_joint_radius=robot_description_dict.get("l5_pre_joint_radius", 0.042),
            next_joint_radius=robot_description_dict.get("l5_next_joint_radius", 0.042),
            end_point_pos=Vector3(list(robot_description_dict.get("l5_end_point_pos", [0.0, 0.0, 0.36]))),
            end_point_theta=np.deg2rad(robot_description_dict.get("l5_end_point_theta", 0.0)),
            dual_point_distance=robot_description_dict.get("dual_point_distance", 0.07),
            name="tube5",
            angle=0.0,
            count_joint_volumes=False,
            color=robot_description_dict.get("tube_colors", [[0.1, 0.1, 0.1, 1.0], [0.1, 0.1, 0.1, 1.0], [0.1, 0.1, 0.1, 1.0], [0.1, 0.1, 0.1, 1.0], [0.1, 0.1, 0.1, 1.0]])[4],  # gray blue: [0.68, 0.76, 0.82, 1.0]  orange: [1., 0.745, 0.298, 1] dark green: [0.085, 0.394, 0.350, 1]
            clamp_stl=robot_description_dict.get("clamp_stl", str(CWD / "src/ariel/body_phenotypes/Lynx_Arm/models/clamp_0226.stl")),
            collision_group=6,
            collision_alpha=collision_alpha,
        )
    else:
        tube5 = None

    ee_cyl = StraightTube(
        cylinder_length=0.07,  # default 0.033 for the optitrack set
        cylinder_radius=0.010,
        name="ee_cylinder",
    )

    ee = EndEffector(
        name="end_effector",
    )

    # Attach in the right order
    root.attach = j1
    joints_sequence = [j1, j2, j3, j4, j5, j6][:num_joints]
    available_tubes = [tube1, tube2, tube3, tube4, tube5][:num_joints - 1]

    inserted_tubes = 0
    for idx in range(len(genotype_tube)):
        current_idx = idx + inserted_tubes
        if genotype_tube[idx] == 0:
            joints_sequence[current_idx].attach = joints_sequence[current_idx + 1]
        elif genotype_tube[idx] == 1:
            joints_sequence.insert(current_idx + 1, available_tubes[0])
            del available_tubes[0]
            joints_sequence[current_idx].attach = joints_sequence[current_idx + 1]
            joints_sequence[current_idx +
                             1].attach = joints_sequence[current_idx + 2]
            inserted_tubes += 1

    # attaching the end parts
    joints_sequence[-1].attach = ee_cyl

    # Task-specific end-effector
    task = robot_description_dict.get("task", "reach")
    if task == "ball in cup":
        cup = CupModule()
        ee_cyl.attach = cup
    else:
        ee_cyl.attach = ee

    return root


def traverse_robot(root) -> None:
    """Traverses the robot's chain and prints module info."""
    current = root
    idx = 0
    while current is not None:
        module_type = type(current).__name__
        info = f"{idx}: {module_type}"

        # If it's a joint, add inline/orthogonal info
        if hasattr(current, "angle"):
            if isinstance(current, JointInline):
                info += " (Inline)"
            elif isinstance(current, JointOrthogonal):
                info += " (Orthogonal)"
            else:
                info += " (Unknown joint type)"

        # If it has a name attribute, include it
        if hasattr(current, "name"):
            info += f", name={current.name}"

        current = getattr(current, "attach", None)
        idx += 1


if __name__ == "__main__":
    # Example usage
    params = {
        "num_joints": 4,
        "genotype_tube": [1, 0, 0, 1, 0],
        "genotype_joints": 4,
        "rotation_angles": [0, np.pi / 4, np.pi / 2, np.pi / 4, np.pi / 6, np.pi / 3],
        "robot_description_dict": {
            "control_mode": CONTROL_MODE,
        },
    }

    robot_base = construct(params)
    traverse_robot(robot_base)
