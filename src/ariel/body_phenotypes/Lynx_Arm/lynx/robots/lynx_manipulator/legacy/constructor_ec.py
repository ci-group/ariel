from pyrr import Quaternion
import numpy as np
import copy
import os, sys
from ..modules.base import Base
from ..modules.legacy.joint import JointInline, JointOrphogonal, JointOrthogonal
from ..modules.straight_tube import StraightTube
from ..modules.right_angle_tube import RightAngleTube
from ..modules.end_effector import EndEffector
from ..modules.bezier_tube import BezierTube
# from .modules.bspline_tube_jed import BSplineTube
from ..modules.bspline_tube_xinrui import BSplineTube

from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.math_utils import Vector3


def constructor(
        robot_description_dict: dict = {
            "l_link2": 0.2805  + 0.0035 * 2,  # offset of the two clamps
            "l_link3": 0.2805  + 0.0035 * 2,
            # tube 1:
            "link2_type": "next_joint_pose",  # "straight_tube", "next_joint_pose"
            "l2_pre_joint_radius": 0.062,
            "l2_next_joint_radius": 0.042,
            "l2_end_point_pos": [0.0, 0.0, 0.36],
            "l2_end_point_ori": [0.0, 1.0, 0.0],
            # tube 2:
            "link3_type": "next_joint_pose",
            "l3_pre_joint_radius": 0.042,
            "l3_next_joint_radius": 0.042,
            "l3_end_point_pos": [0.0, 0.0, 0.36],
            "l3_end_point_ori": [0.0, 1.0, 0.0],
            "joint1_rotation_angle": 0.0,
            "joint2_rotation_angle": 0.0,
            "joint3_rotation_angle": 0.0,
            "joint4_rotation_angle": 0.0,
            "joint5_rotation_angle": 0.0,
            "joint6_rotation_angle": 0.0,
            "control_mode": "position",
            "clamp_length": 0.045,
            "dual_point_distance": 0.15,
            "num_segments": 100,
            },
        ):
    
    print(f"robot_description_dict: {robot_description_dict}")
    
    if robot_description_dict["link2_type"] == "straight_tube":
        tube1 = StraightTube(
            cylinder_length=robot_description_dict["l_link2"],  # TODO: offset of the orthogonal joints are missing
            cylinder_radius=0.035,
            name="tube1",
            # attachment_quat=[1,0,0,1], # Rotate attachment point to be horizontal
        )
    elif robot_description_dict["link2_type"] == "next_joint_pose":
        tube1 = BSplineTube(
            num_segments=robot_description_dict.get("num_segments", 100),
            cylinder_radius=0.0416,
            mounting_length_start=robot_description_dict.get("clamp_length", 0.045),
            mounting_length_end=robot_description_dict.get("clamp_length", 0.045),
            pre_joint_radius=robot_description_dict.get("l2_pre_joint_radius", 0.062),
            next_joint_radius=robot_description_dict.get("l2_next_joint_radius", 0.042),
            end_point_pos=Vector3(list(robot_description_dict.get("l2_end_point_pos", [0.0, 0.0, 0.36]))),  # Optional end point position constraint
            end_point_ori=Vector3(list(robot_description_dict.get("l2_end_point_ori", [0.0, 1.0, 0.0]))),  # Optional end point orientation constraint
            dual_point_distance=robot_description_dict.get("dual_point_distance", 0.15),  # Distance for dual control points when setting end constraints
            name="tube1",
            angle=np.deg2rad(robot_description_dict.get("joint2_rotation_angle", 0.0)),  # Default to 0.0 if not provided
        )
    else:
        raise ValueError(f"Unknown link2_type: {robot_description_dict['link2_type']}")
    
    if robot_description_dict["link3_type"] == "straight_tube":
        tube2 = StraightTube(
            cylinder_length=robot_description_dict["l_link3"],  # TODO: offset of the orthogonal joints are missing
            cylinder_radius=0.035,
            name="tube2",
        )
    elif robot_description_dict["link3_type"] == "next_joint_pose":
        tube2 = BSplineTube(
            num_segments=robot_description_dict.get("num_segments", 100),
            cylinder_radius=0.0416,
            mounting_length_start=robot_description_dict.get("clamp_length", 0.045),
            mounting_length_end=robot_description_dict.get("clamp_length", 0.045),
            pre_joint_radius=robot_description_dict.get("l3_pre_joint_radius", 0.042),
            next_joint_radius=robot_description_dict.get("l3_next_joint_radius", 0.042),
            end_point_pos=Vector3(list(robot_description_dict.get("l3_end_point_pos", [0.0, 0.0, 0.2]))),  # Optional end point position constraint
            end_point_ori=Vector3(list(robot_description_dict.get("l3_end_point_ori", [0.0, 1.0, 0.0]))),  # Optional end point orientation constraint
            dual_point_distance=robot_description_dict.get("dual_point_distance", 0.15),  # Distance for dual control points when setting end constraints
            name="tube2",
            angle=0,  # Default to 0.0 if not provided
        )
    else:
        raise ValueError(f"Unknown link3_type: {robot_description_dict['link3_type']}")

    # Define the root body
    root = Base(
        base_length1=0.021, 
        base_radius1=0.08,
        base_length2=0.021 + 0.001,  # offset for the base
        base_radius2=0.062,
        name="lynx_base",
    )

    j1 = JointInline(
        cylinder_length1=0.13,
        cylinder_radius1=0.062 ,
        cylinder_length2=(0.013 + 0.035) * 2 + 0.001,  # offset of the motor 1
        cylinder_radius2=0.062,
        angle=np.deg2rad(robot_description_dict.get("joint1_rotation_angle", 0.0)),
        fixed_part_angle=0,  # for the rotation of the fixed side of the joint
        name="joint1",
        control_mode=robot_description_dict["control_mode"],  # Use the default control mode
        armature=0.01, damping=100000, frictionloss=0.000001,
    )

    j2_angle = np.deg2rad(robot_description_dict.get("joint2_rotation_angle", 0.0))

    j2 = JointInline(
        cylinder_length1=0.13,
        cylinder_radius1=0.062,
        cylinder_length2=0.013 * 2,
        cylinder_radius2=0.062,
        angle=-np.pi + j2_angle if robot_description_dict["link2_type"] == "straight_tube" else -np.pi,  # -np.pi/2 for 90 degrees CCW rotation
        fixed_part_angle=0,  # for the rotation of the fixed side of the joint
        name="joint2",
        control_mode=robot_description_dict["control_mode"],  # Use the default control mode
        armature=0.01, damping=100000, frictionloss=0.000001,
    )



    # TODO: How to change the direction of this motor?
    j3 = JointOrthogonal(
        cylinder_length1=0.09218,
        cylinder_radius1=0.042 ,
        cylinder_length2=0.028 + 0.001,  # offset for the joint gap 3-4
        cylinder_radius2=0.042,
        angle=np.deg2rad(robot_description_dict.get("joint3_rotation_angle", 0.0)),
        name="joint3",
        control_mode=robot_description_dict["control_mode"],  # Use the default control mode
        armature=0.01, damping=100000, frictionloss=0.000001,
    )

    j4 = JointOrthogonal(
        cylinder_length1=0.09218,
        cylinder_radius1=0.042,
        cylinder_length2=0.028 + 0.001,  # offset for the joint gap 4-tube
        cylinder_radius2=0.042,
        angle=np.pi + np.deg2rad(robot_description_dict.get("joint4_rotation_angle", 0.0)),
        name="joint4",
        control_mode=robot_description_dict["control_mode"],  # Use the default control mode
        armature=0.01, damping=100000, frictionloss=0.000001,
    )

    j5 = JointOrthogonal(
        cylinder_length1=0.096,
        cylinder_radius1=0.042,
        cylinder_length2=0.026 + 0.001,  # offset for the joint gap 5-6
        cylinder_radius2=0.042,
        angle=np.deg2rad(robot_description_dict.get("joint5_rotation_angle", 0.0)),
        name="joint5",
        control_mode=robot_description_dict["control_mode"],  # Use the default control mode
        armature=0.01, damping=100000, frictionloss=0.000001,
    )

    j6 = JointOrthogonal(
        cylinder_length1=0.096,
        cylinder_radius1=0.042 ,
        cylinder_length2=0.026 + 0.001,
        cylinder_radius2=0.042,
        angle=np.deg2rad(robot_description_dict.get("joint6_rotation_angle", 0.0)),  # np.pi
        name="joint6",
        control_mode=robot_description_dict["control_mode"],  # Use the default control mode
        armature=0.01, damping=100000, frictionloss=0.000001,
    )

    ee_cyl = StraightTube(
        cylinder_length=0.033,
        cylinder_radius=0.01,
        name="ee_cylinder",
    )

    ee = EndEffector(
        name="end_effector",
    )

    # root.attach = bz
    root.attach = j1
    j1.attach = j2
    j2.attach = tube1
    tube1.attach = j3
    j3.attach = j4
    j4.attach = tube2
    tube2.attach = j5
    j5.attach = j6
    j6.attach = ee_cyl
    ee_cyl.attach = ee
    return root