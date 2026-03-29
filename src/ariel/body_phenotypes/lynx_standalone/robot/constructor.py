import numpy as np

# Local imports
from ariel.body_phenotypes.lynx_standalone.robot.math_utils import Vector3
from ariel.body_phenotypes.lynx_standalone.robot.mj_base import MjBase
from ariel.body_phenotypes.lynx_standalone.robot.mj_bspline_tube import MjBSplineTubeWithClamps
from ariel.body_phenotypes.lynx_standalone.robot.mj_end_effector import MjEndEffector
from ariel.body_phenotypes.lynx_standalone.robot.mj_joint import MjJointInline, MjJointOrthogonal
from ariel.body_phenotypes.lynx_standalone.robot.mj_tube import MjTube

import pathlib
import sys
from ariel import CWD

sys.path.append(pathlib.Path.cwd())

def construct_lynx(
    robot_description_dict: dict | None = None,
):
    """
    Assembles the Lynxmotion robot using MjSpec-based modules.

    The assembly follows the chain:
    Base -> Joint 1 -> Joint 2 -> Tube 1 -> Joint 3 -> Joint 4 -> Tube 2 -> Joint 5 -> Joint 6 -> EE Cylinder -> End Effector

    This configuration is consistent with the existing Lynxmotion robot structure.
    """
    if robot_description_dict is None:
        robot_description_dict: dict = {
            # tube order:
            "genotype_tube": [0, 1, 0, 1, 0],
            # tube 1:
            "l2_end_point_pos": [0.0, 0.0, 0.36],
            "l2_end_point_theta": 0.0,
            # tube 2:
            "l3_end_point_pos": [0.0, 0.0, 0.36],
            "l3_end_point_theta": 0.0,
            # joint type:
            "genotype_joints": 1,
            # fixed-side rotation angles:
            "rotation_angles": [180, 0, 0, -180, 0, 0],
            "clamp_length": 0.0359,
            "dual_point_distance": 0.07,
            "num_segments": 100,
        }
    tube_lengths = robot_description_dict.get("tube_lengths", [0.36, 0.36, 0.36, 0.36, 0.36])
    tube_lengths = np.asarray(tube_lengths, dtype=np.float64)
    if tube_lengths.size < 5:
        # Pad to 5 tubes if a shorter list is provided.
        tube_lengths = np.pad(tube_lengths, (0, 5 - tube_lengths.size), constant_values=0.36)
    tube_lengths = np.clip(tube_lengths[:5], 0.05, 0.6)

    # If explicit tube endpoint positions are not supplied, infer from tube_lengths.
    for idx in range(5):
        key = f"l{idx + 1}_end_point_pos"
        if key not in robot_description_dict:
            robot_description_dict[key] = [0.0, 0.0, float(tube_lengths[idx])]

    genotype_tube = robot_description_dict["genotype_tube"]
    genotype_joints = robot_description_dict["genotype_joints"]
    rotation_angles = robot_description_dict["rotation_angles"]

    # construct the list for joint type, True means inline, False means orthogonal
    # Note: In the original code, joints is length 5, but there are 6 joints total.
    # j1 is always inline. j2-j6 types are determined by genotype_joints.
    joint_types = [True] * 5
    joint_types[genotype_joints:] = [False] * (len(joint_types) - (genotype_joints))

    # 1. Create Base
    root = MjBase(
        base_length1=0.001,
        base_radius1=0.08,
        base_length2=0.018,
        base_radius2=0.08,
        name="lynx_base",
    )

    # 2. Create Joints
    # Joint 1 (Always Inline)
    j1 = MjJointInline(
        cylinder_length0=0.036, cylinder_radius0=0.058,
        cylinder_length1=0.128, cylinder_radius1=0.062,
        cylinder_length2=0.013, cylinder_radius2=0.062,
        angle=np.deg2rad(rotation_angles[0]),
        name="joint1",
        armature=0.01, damping=20.0, frictionloss=0.01,
    )

    # Joint 2
    JointType2 = MjJointInline if joint_types[0] else MjJointOrthogonal
    j2 = JointType2(
        cylinder_length0=0.036, cylinder_radius0=0.058,
        cylinder_length1=0.128, cylinder_radius1=0.062,
        cylinder_length2=0.013, cylinder_radius2=0.062,
        angle=np.deg2rad(rotation_angles[1]),
        name="joint2",
        armature=0.01, damping=20.0, frictionloss=0.01,
    )

    # Joint 3
    JointType3 = MjJointInline if joint_types[1] else MjJointOrthogonal
    j3 = JointType3(
        cylinder_length0=0.029, cylinder_radius0=0.04,
        cylinder_length1=0.092, cylinder_radius1=0.042,
        cylinder_length2=0.008, cylinder_radius2=0.042,
        angle=np.deg2rad(rotation_angles[2]),
        name="joint3",
        armature=0.01, damping=20.0, frictionloss=0.01,
    )

    # Joint 4
    JointType4 = MjJointInline if joint_types[2] else MjJointOrthogonal
    j4 = JointType4(
        cylinder_length0=0.029, cylinder_radius0=0.04,
        cylinder_length1=0.092, cylinder_radius1=0.042,
        cylinder_length2=0.008, cylinder_radius2=0.042,
        angle=np.deg2rad(rotation_angles[3]),
        name="joint4",
        armature=0.01, damping=20.0, frictionloss=0.01,
    )

    # Joint 5
    JointType5 = MjJointInline if joint_types[3] else MjJointOrthogonal
    j5 = JointType5(
        cylinder_length0=0.029, cylinder_radius0=0.04,
        cylinder_length1=0.096, cylinder_radius1=0.042,
        cylinder_length2=0.008, cylinder_radius2=0.042,
        angle=np.deg2rad(rotation_angles[4]),
        name="joint5",
        armature=0.01, damping=20.0, frictionloss=0.01,
    )

    # Joint 6
    JointType6 = MjJointInline if joint_types[4] else MjJointOrthogonal
    j6 = JointType6(
        cylinder_length0=0.029, cylinder_radius0=0.04,
        cylinder_length1=0.096, cylinder_radius1=0.042,
        cylinder_length2=0.008, cylinder_radius2=0.042,
        angle=np.deg2rad(rotation_angles[5]),
        name="joint6",
        armature=0.01, damping=20.0, frictionloss=0.01,
    )

    # 3. Create Tubes
    tube1 = MjBSplineTubeWithClamps(
        num_segments=robot_description_dict.get("num_segments", 100),
        cylinder_radius=0.0416,
        mounting_length_start=robot_description_dict.get("clamp_length", 0.0359),
        mounting_length_end=robot_description_dict.get("clamp_length", 0.0359),
        pre_joint_radius=robot_description_dict.get("l1_pre_joint_radius", 0.062),
        next_joint_radius=robot_description_dict.get("l1_next_joint_radius", 0.042),
        end_point_pos=Vector3(list(robot_description_dict.get("l1_end_point_pos", [0.0, 0.0, 0.36]))),
        end_point_theta=np.deg2rad(robot_description_dict.get("l1_end_point_theta", 0.0)),
        dual_point_distance=robot_description_dict.get("dual_point_distance", 0.07),
        name="tube1",
        angle=0.0,
        count_joint_volumes=False,
        color=[0.68, 0.76, 0.82, 1.0],  # gray blue: [0.68, 0.76, 0.82, 1.0]  orange: [1., 0.745, 0.298, 1] dark green: [0.085, 0.394, 0.350, 1]
    )

    tube2 = MjBSplineTubeWithClamps(
        num_segments=robot_description_dict.get("num_segments", 100),
        cylinder_radius=0.0416,
        mounting_length_start=robot_description_dict.get("clamp_length", 0.0359),
        mounting_length_end=robot_description_dict.get("clamp_length", 0.0359),
        pre_joint_radius=robot_description_dict.get("l2_pre_joint_radius", 0.062),
        next_joint_radius=robot_description_dict.get("l2_next_joint_radius", 0.042),
        end_point_pos=Vector3(list(robot_description_dict.get("l2_end_point_pos", [0.0, 0.0, 0.36]))),
        end_point_theta=np.deg2rad(robot_description_dict.get("l2_end_point_theta", 0.0)),
        dual_point_distance=robot_description_dict.get("dual_point_distance", 0.07),
        name="tube2",
        angle=0.0,
        count_joint_volumes=False,
        color=[0.68, 0.76, 0.82, 1.0],  # gray blue: [0.68, 0.76, 0.82, 1.0]  orange: [1., 0.745, 0.298, 1] dark green: [0.085, 0.394, 0.350, 1]
    )

    tube3 = MjBSplineTubeWithClamps(
        num_segments=robot_description_dict.get("num_segments", 100),
        cylinder_radius=0.0416,
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
        color=[0.68, 0.76, 0.82, 1.0],  # gray blue: [0.68, 0.76, 0.82, 1.0]  orange: [1., 0.745, 0.298, 1] dark green: [0.085, 0.394, 0.350, 1]
    )

    tube4 = MjBSplineTubeWithClamps(
        num_segments=robot_description_dict.get("num_segments", 100),
        cylinder_radius=0.0416,
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
        color=[0.68, 0.76, 0.82, 1.0],  # gray blue: [0.68, 0.76, 0.82, 1.0]  orange: [1., 0.745, 0.298, 1] dark green: [0.085, 0.394, 0.350, 1]
    )

    tube5 = MjBSplineTubeWithClamps(
        num_segments=robot_description_dict.get("num_segments", 100),
        cylinder_radius=0.0416,
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
        color=[0.68, 0.76, 0.82, 1.0],  # gray blue: [0.68, 0.76, 0.82, 1.0]  orange: [1., 0.745, 0.298, 1] dark green: [0.085, 0.394, 0.350, 1]
    )

    ee_cyl = MjTube(
        cylinder_length=0.033,
        cylinder_radius=0.010,
        name="ee_cylinder",
    )

    # 4. Create End Effector
    ee = MjEndEffector(name="end_effector")

    # 5. Assemble the chain using site-based attachment
    # Helper to find site by name
    def get_site(body, name):
        for s in body.sites:
            # Check for exact name or name with prefix
            if s.name == name or s.name.endswith("_" + name) or s.name.endswith(name):
                return s
        msg = f"Site {name} not found in body {body.name}. Available: {[s.name for s in body.sites]}"
        raise ValueError(msg)

    # Define the sequence of modules based on genotype_tube
    # Initial sequence: [j1, j2, j3, j4, j5, j6]
    # We insert tubes into this sequence
    modules_sequence = [j1, j2, j3, j4, j5, j6]
    available_tubes = [tube1, tube2, tube3, tube4, tube5]

    inserted_count = 0
    for i, has_tube in enumerate(genotype_tube):
        if has_tube == 1 and available_tubes:
            # Insert tube after the joint at index i + inserted_count
            modules_sequence.insert(i + inserted_count + 1, available_tubes.pop(0))
            inserted_count += 1

    # Add EE parts at the end
    modules_sequence.extend([ee_cyl, ee])

    # Attach them in order
    # root -> modules_sequence[0] -> modules_sequence[1] -> ...
    get_site(root.body, "mount").attach_body(modules_sequence[0].body, prefix=f"{modules_sequence[0].name}_")

    for i in range(len(modules_sequence) - 1):
        parent = modules_sequence[i]
        child = modules_sequence[i + 1]

        # Determine attachment site on parent
        if hasattr(parent, "rotor"):
            parent_site = get_site(parent.rotor, "attach")
        else:
            # For tubes, we use the unique site name we created
            site_name = "attach"
            # Check for both MjBSplineTubeWithClamps and MjTube
            if isinstance(parent, (MjBSplineTubeWithClamps, MjTube)):
                site_name = f"{parent.name}_attach"

            parent_site = get_site(parent.body, site_name)

        parent_site.attach_body(child.body, prefix=f"{child.name}_")

    return root


if __name__ == "__main__":
    # Test construction
    robot = construct_lynx()

    # Compile to check for errors
    model = robot.spec.compile()