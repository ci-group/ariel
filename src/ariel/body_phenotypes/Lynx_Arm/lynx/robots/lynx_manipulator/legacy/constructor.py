from pyrr import Quaternion
import numpy as np
import copy
from ..modules.base import Base
from ..modules.legacy.joint import JointInline, JointOrthogonal, JointOrthogonal_new
from ..modules.straight_tube import StraightTube
from ..modules.end_effector import EndEffector
from ..modules.bspline_tube_xinrui import BSplineTube
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.math_utils import Vector3

CONTROL_MODE = "position"  # Default control mode for joints


def construct(
        robot_description_dict: dict = {
            # tube order:
            "genotype_tube": [0, 1, 0, 1, 0],  # default: [0, 1, 0, 1, 0]
            # tube 1:
            "link2_type": "next_joint_pose",  # "straight_tube", "next_joint_pose"
            "l2_pre_joint_radius": 0.062,
            "l2_next_joint_radius": 0.042,
            "l2_end_point_pos": [0.0, 0.0, 0.36],
            "l2_end_point_theta": 0.0,  # deg, rotation in YZ plane from +Z
            # tube 2:
            "link3_type": "next_joint_pose",
            "l3_pre_joint_radius": 0.042,
            "l3_next_joint_radius": 0.042,
            "l3_end_point_pos": [0.0, 0.0, 0.36],
            "l3_end_point_theta": 0.0,  # deg, rotation in YZ plane from +Z
            # joint type:
            "genotype_joints": 2, # default: 1. range: {1,2,3,4,5}
            # fixed-side rotation angles:
            "rotation_angles": [180, 0, 0, -180, 0, 0],  # default: [np.pi, 0, 0, -np.pi, 0, 0]
            # ===== other static params =====:
            "clamp_length": 0.051,  # version: models/clamp_1216.stl
            "dual_point_distance": 0.15,
            "num_segments": 100,
            },
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

    Returns
    -------
    Base
        The root `Base` module. Its `.attach` chain encodes joints,
        tubes and the end-effector.
    """

    genotype_tube = robot_description_dict["genotype_tube"]
    genotype_joints = robot_description_dict["genotype_joints"]
    rotation_angles = robot_description_dict["rotation_angles"]

    def _fixed_from_angle(j_index: int, angle: float) -> float:
        return angle

    # costruct the list for joint type, True means inline,  False means orthogonal
    joints = [True, True, True, True, True]
    joints[genotype_joints:] = [False]*(len(joints) - genotype_joints)

    # Define the Base and 1 joint - nonmodifiable by evolution
    root = Base(
        base_length1=0.001,
        base_radius1=0.08,
        base_length2=0.017 + 0.001,  # offset for the base
        base_radius2=0.08,
        name="lynx_base",
    )

    j1 = JointInline(
        cylinder_length1=0.13,
        cylinder_radius1=0.062,
        cylinder_length2=(0.013 + 0.035) * 2 + 0.001,  # offset of the motor 1
        cylinder_radius2=0.062,
        angle=np.deg2rad(rotation_angles[0]),
        name="joint1",
        control_mode="position",
        armature=0.01, damping=100000, frictionloss=0.000001,
    )

    # Define joints
    # j2
    JointType2 = JointInline if joints[0] else JointOrthogonal_new
    j2_kwargs = dict(
        cylinder_length1=0.13,
        cylinder_radius1=0.062,
        cylinder_length2=0.013 * 2,  # fixed-side offset of the joint 2
        cylinder_radius2=0.062,
        angle=np.deg2rad(rotation_angles[1]),
        name="joint2",
        control_mode="position",
        armature=0.01, damping=100000, frictionloss=0.000001,
    )
    j2 = JointType2(**j2_kwargs)

    # j3
    JointType3 = JointInline if joints[1] else JointOrthogonal_new
    j3_kwargs = dict(
        cylinder_length1=0.09218,
        cylinder_radius1=0.042,
        cylinder_length2=0.028 + 0.001,  # offset for the joint gap 3-4
        cylinder_radius2=0.042,
        angle=np.deg2rad(rotation_angles[2]),
        name="joint3",
        control_mode="position",
        armature=0.01, damping=100000, frictionloss=0.000001,
    )
    j3 = JointType3(**j3_kwargs)

    # j4
    JointType4 = JointInline if joints[2] else JointOrthogonal_new
    j4_kwargs = dict(
        cylinder_length1=0.09218,
        cylinder_radius1=0.042,
        cylinder_length2=0.028 + 0.001,  # offset for the joint gap 4-tube
        cylinder_radius2=0.042,
        angle=np.deg2rad(rotation_angles[3]),
        name="joint4",
        control_mode="position",
        armature=0.01, damping=100000, frictionloss=0.000001,
    )
    j4 = JointType4(**j4_kwargs)

    # j5
    JointType5 = JointInline if joints[3] else JointOrthogonal_new
    j5_kwargs = dict(
        cylinder_length1=0.096,
        cylinder_radius1=0.042,
        cylinder_length2=0.026 + 0.001,  # offset for the joint gap 5-6
        cylinder_radius2=0.042,
        angle=np.deg2rad(rotation_angles[4]),
        name="joint5",
        control_mode="position",
        armature=0.01, damping=100000, frictionloss=0.000001,
    )
    j5 = JointType5(**j5_kwargs)

    # j6
    JointType6 = JointInline if joints[4] else JointOrthogonal_new
    j6_kwargs = dict(
        cylinder_length1=0.096,
        cylinder_radius1=0.042,
        cylinder_length2=0.026 + 0.001,
        cylinder_radius2=0.042,
        angle=np.deg2rad(rotation_angles[5]),  # np.pi
        name="joint6",
        control_mode="position",
        armature=0.01, damping=100000, frictionloss=0.000001,
    )
    j6 = JointType6(**j6_kwargs)

    # Define straight tubes and end part
    tube1 = BSplineTube(
        num_segments=robot_description_dict.get("num_segments", 100),
        cylinder_radius=0.0416,
        mounting_length_start=robot_description_dict.get("clamp_length", 0.045),
        mounting_length_end=robot_description_dict.get("clamp_length", 0.045),
        pre_joint_radius=robot_description_dict.get("l2_pre_joint_radius", 0.062),
        next_joint_radius=robot_description_dict.get("l2_next_joint_radius", 0.042),
        end_point_pos=Vector3(list(robot_description_dict.get("l2_end_point_pos", [0.0, 0.0, 0.36]))),  # Optional end point position constraint
        end_point_theta=np.deg2rad(robot_description_dict.get("l2_end_point_theta", 0.0)),  # rad, rotation in YZ plane from +Z
        dual_point_distance=robot_description_dict.get("dual_point_distance", 0.15),  # Distance for dual control points when setting end constraints
        name="tube1",
        angle=np.deg2rad(robot_description_dict.get("joint2_rotation_angle", 0.0)),  # Default to 0.0 if not provided
        count_joint_volumes=False,
    )

    tube2 = BSplineTube(
        num_segments=robot_description_dict.get("num_segments", 100),
        cylinder_radius=0.0416,
        mounting_length_start=robot_description_dict.get("clamp_length", 0.045),
        mounting_length_end=robot_description_dict.get("clamp_length", 0.045),
        pre_joint_radius=robot_description_dict.get("l3_pre_joint_radius", 0.042),
        next_joint_radius=robot_description_dict.get("l3_next_joint_radius", 0.042),
        end_point_pos=Vector3(list(robot_description_dict.get("l3_end_point_pos", [0.0, 0.0, 0.2]))),  # Optional end point position constraint
        end_point_theta=np.deg2rad(robot_description_dict.get("l3_end_point_theta", 0.0)),  # rad, rotation in YZ plane from +Z
        dual_point_distance=robot_description_dict.get("dual_point_distance", 0.15),  # Distance for dual control points when setting end constraints
        name="tube2",
        angle=0,  # Default to 0.0 if not provided
        count_joint_volumes=False,
    )

    ee_cyl = StraightTube(
        cylinder_length=0.033,
        cylinder_radius=0.010,
        name="ee_cylinder",
    )

    ee = EndEffector(
        name="end_effector",
    )

    # Attach in the right order
    root.attach = j1
    joints_sequence = [j1, j2, j3, j4, j5, j6]
    available_tubes = [tube1, tube2]

    inserted_tubes = 0
    for idx in range(len(genotype_tube)):
        current_idx = idx + inserted_tubes
        if genotype_tube[idx] == 0:
            joints_sequence[current_idx].attach = joints_sequence[current_idx + 1]
        elif genotype_tube[idx] == 1:
            joints_sequence.insert(current_idx + 1, available_tubes[0])
            del available_tubes[0]
            joints_sequence[current_idx].attach = joints_sequence[current_idx+1]
            joints_sequence[current_idx +
                            1].attach = joints_sequence[current_idx + 2]
            inserted_tubes += 1

    # attaching the end parts
    j6.attach = ee_cyl
    ee_cyl.attach = ee

    return root


def traverse_robot(root):
    """
    Traverses the robot's chain and prints module info.
    """
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

        print(info)
        current = getattr(current, "attach", None)
        idx += 1


if __name__ == "__main__":
    # Example usage
    params = {
        "genotype_tube": [1, 0, 0, 1, 0],
        "genotype_joints": 4,
        "tube_lengths": [0.2, 0.15],
        "rotation_angles": [0, np.pi/4, np.pi/2, np.pi/4, np.pi/6, np.pi/3],
        "robot_description_dict": {
            "control_mode": CONTROL_MODE
        }
    }

    robot_base = construct(params)
    traverse_robot(robot_base)