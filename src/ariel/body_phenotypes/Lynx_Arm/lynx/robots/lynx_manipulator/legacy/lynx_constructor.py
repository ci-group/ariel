from pyrr import Quaternion
import numpy as np
import copy
from ..modules.base import Base
from ..modules.legacy.joint import JointInline, JointOrphogonal, JointOrthogonal
from ..modules.straight_tube import StraightTube
from ..modules.right_angle_tube import RightAngleTube
from ..modules.end_effector import EndEffector
from ..modules.bezier_tube import BezierTube
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.math_utils import Vector3

CONTROL_MODE = "position"  # Default control mode for joints


def construct(params: dict):
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

    genotype_tube = params["genotype_tube"]
    genotype_joints = params["genotype_joints"]
    tube_lengths = params["tube_lengths"]
    rotation_angles = params["rotation_angles"]
    robot_description_dict = params.get("robot_description_dict", {
        "control_mode": "position"
    })

    def _fixed_from_angle(j_index: int, angle: float) -> float:
        return angle

    # costruct the list for joint type, True means inline,  False means orthogonal
    joints = [True, True, True, True, True]
    joints[genotype_joints:] = [False]*(len(joints) - genotype_joints)

    # Define the Base and 1 joint - nonmodifiable by evolution
    root = Base(
        base_length1=0.021,
        base_radius1=0.08,
        base_length2=0.021 + 0.001,  # offset for the base
        base_radius2=0.070,
        name="lynx_base",
    )

    j1 = JointInline(
        cylinder_length1=0.13,
        cylinder_radius1=0.062,
        cylinder_length2=(0.013 + 0.035) * 2 + 0.001,  # offset of the motor 1
        cylinder_radius2=0.062,
        angle=0,
        fixed_part_angle=rotation_angles[0],
        name="joint1",
        # Use the default control mode
        control_mode=robot_description_dict["control_mode"],
        armature=0.01, damping=100000, frictionloss=0.000001,
    )

    # Define joints
    # j2
    JointType2 = JointInline if joints[0] else JointOrthogonal
    j2_kwargs = dict(
        cylinder_length1=0.13,
        cylinder_radius1=0.062,
        cylinder_length2=0.013 * 2,
        cylinder_radius2=0.062,
        angle=rotation_angles[1],
        name="joint2",
        control_mode=robot_description_dict["control_mode"],
        armature=0.01, damping=100000, frictionloss=0.000001,
    )
    if joints[0]:
        j2_kwargs["fixed_part_angle"] = _fixed_from_angle(
            2, rotation_angles[1])
    j2 = JointType2(**j2_kwargs)

    # j3
    JointType3 = JointInline if joints[1] else JointOrthogonal
    j3_kwargs = dict(
        cylinder_length1=0.09218,
        cylinder_radius1=0.042,
        cylinder_length2=0.028 + 0.001,  # offset for the joint gap 3-4
        cylinder_radius2=0.042,
        angle=rotation_angles[2],
        name="joint3",
        control_mode=robot_description_dict["control_mode"],
        armature=0.01, damping=100000, frictionloss=0.000001,
    )
    if joints[1]:
        j3_kwargs["fixed_part_angle"] = _fixed_from_angle(
            3, rotation_angles[2])
    j3 = JointType3(**j3_kwargs)

    # j4
    JointType4 = JointInline if joints[2] else JointOrthogonal
    j4_kwargs = dict(
        cylinder_length1=0.09218,
        cylinder_radius1=0.042,
        cylinder_length2=0.028 + 0.001,  # offset for the joint gap 4-tube
        cylinder_radius2=0.042,
        angle=rotation_angles[3],
        name="joint4",
        control_mode=robot_description_dict["control_mode"],
        armature=0.01, damping=100000, frictionloss=0.000001,
    )
    if joints[2]:
        j4_kwargs["fixed_part_angle"] = _fixed_from_angle(
            4, rotation_angles[3])
    j4 = JointType4(**j4_kwargs)

    # j5
    JointType5 = JointInline if joints[3] else JointOrthogonal
    j5_kwargs = dict(
        cylinder_length1=0.096,
        cylinder_radius1=0.042,
        cylinder_length2=0.026 + 0.001,  # offset for the joint gap 5-6
        cylinder_radius2=0.042,
        angle=rotation_angles[4],
        name="joint5",
        control_mode=robot_description_dict["control_mode"],
        armature=0.01, damping=100000, frictionloss=0.000001,
    )
    if joints[3]:
        j5_kwargs["fixed_part_angle"] = _fixed_from_angle(
            5, rotation_angles[4])
    j5 = JointType5(**j5_kwargs)

    # j6
    JointType6 = JointInline if joints[4] else JointOrthogonal
    j6_kwargs = dict(
        cylinder_length1=0.096,
        cylinder_radius1=0.042,
        cylinder_length2=0.026 + 0.001,
        cylinder_radius2=0.042,
        angle=rotation_angles[5],  # np.pi
        name="joint6",
        # Use the default control mode
        control_mode=robot_description_dict["control_mode"],
        armature=0.01, damping=100000, frictionloss=0.000001,
    )
    if joints[4]:
        j6_kwargs["fixed_part_angle"] = _fixed_from_angle(
            6, rotation_angles[5])
    j6 = JointType6(**j6_kwargs)

    # Define straight tubes and end part
    st2 = StraightTube(
        cylinder_length=tube_lengths[0] + 0.008,
        cylinder_radius=0.035,
        name="straight_tube2",
    )

    st1 = StraightTube(
        cylinder_length=tube_lengths[1] + 0.008,  # offset for the motor 3
        cylinder_radius=0.035,
        name="straight_tube",
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
    available_tubes = [st1, st2]

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