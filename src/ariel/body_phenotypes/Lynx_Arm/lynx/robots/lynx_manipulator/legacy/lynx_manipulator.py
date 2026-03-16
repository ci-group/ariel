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
from ..modules.bspline_tube_jed import BSplineTube

from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.math_utils import Vector3

# Nominal approximate lengths for conceptual links, corresponding to StraightTube definitions below
# These values are used where a fixed length is required for calculations (e.g., safety watchdog)
NOMINAL_LINK2_LENGTH_M = 0.2805 + 0.0035 * 2  # offset of the two clamps
NOMINAL_LINK3_LENGTH_M = 0.3055 + 0.0035 * 2
CONTROL_MODE = "position"  # Default control mode for joints

def lynx_manipulator(
        robot_description_dict: dict = {
            "l_link2": 0.2805  + 0.0035 * 2,  # offset of the two clamps
            "l_link3": 0.2805  + 0.0035 * 2,
            "link2_type": "straight_tube",  # "straight_tube", "bspline_tube_slightly_rotated", "bspline_tube_rotate_90"
            "link3_type": "straight_tube",
            "link2_rotation_angle": 0.0,
            "link3_rotation_angle": 0.0,
            "joint1_rotation_angle": 0.0,
            "joint2_rotation_angle": 0.0,  # 120 degrees in radians
            "joint3_rotation_angle": 0.0,
            "joint4_rotation_angle": 0.0,
            "joint5_rotation_angle": 0.0,
            "joint6_rotation_angle": 0.0,
            "control_mode": "position",
            },
            ):

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
        angle=0,
        fixed_part_angle=np.deg2rad(robot_description_dict["joint1_rotation_angle"]),  # for the rotation of the fixed side of the joint
        name="joint1",
        control_mode=robot_description_dict["control_mode"],  # Use the default control mode
        armature=0.01, damping=100000, frictionloss=0.000001,
    )

    j2 = JointInline(
        cylinder_length1=0.13,
        cylinder_radius1=0.062,
        cylinder_length2=0.013 * 2,
        cylinder_radius2=0.062,
        angle=-np.pi,  # -np.pi/2 for 90 degrees CCW rotation
        fixed_part_angle=np.deg2rad(robot_description_dict["joint2_rotation_angle"]),  # for the rotation of the fixed side of the joint
        name="joint2",
        control_mode=robot_description_dict["control_mode"],  # Use the default control mode
        armature=0.01, damping=100000, frictionloss=0.000001,
    )

    # Define link 2:
    if robot_description_dict["link2_type"] == "straight_tube":
        link2 = StraightTube(
            cylinder_length=robot_description_dict["l_link2"] + 0.008,  # offset for the motor 3
            cylinder_radius=0.035,
            name="link2",
            # attachment_quat=[1,0,0,1], # Rotate attachment point to be horizontal
        )
    elif robot_description_dict["link2_type"] == "bspline_tube_slightly_rotated":
        # slightly rotate:
        scaled_tube_design = {  # Coordinates already in meters
            "control_points": [
                Vector3([0.0, 0.0, 0.0]),
                Vector3([0.0, 0.1, 0.0]),
                Vector3([0.0, 0.2, 0.0]),
            ],
            "degree": 2,
            "num_segments": 10,
            "cylinder_radius": 0.035,
            "name": "link2",
            "mounting_length_start": 0.06,
            "mounting_length_end": 0.06,
        }
        
        link2 = BSplineTube(
            control_points=scaled_tube_design["control_points"],
            degree=scaled_tube_design["degree"],
            num_segments=scaled_tube_design["num_segments"],
            cylinder_radius=scaled_tube_design["cylinder_radius"],
            angle=np.deg2rad(robot_description_dict["link2_rotation_angle"]),  # slight rotation
            name=scaled_tube_design["name"],
            mounting_length_start=scaled_tube_design["mounting_length_start"],
            mounting_length_end=scaled_tube_design["mounting_length_end"],
        )
    elif robot_description_dict["link2_type"] == "bspline_tube_rotate_90":
        # slightly rotate:
        scaled_tube_design = {  # Coordinates already in meters
            "control_points": [
                Vector3([0.0, 0.0, 0.0]),
                Vector3([0.0, 0.0, 0.1]),
                Vector3([0.0, 0.1, 0.1]),
                Vector3([0.0, 0.14, 0.14]),
            ],
            "degree": 3,
            "num_segments": 20,
            "cylinder_radius": 0.035,
            "name": "link2",
            "mounting_length_start": 0.06,
            "mounting_length_end": 0.06,
        }
        
        link2 = BSplineTube(
            control_points=scaled_tube_design["control_points"],
            degree=scaled_tube_design["degree"],
            num_segments=scaled_tube_design["num_segments"],
            cylinder_radius=scaled_tube_design["cylinder_radius"],
            angle=np.deg2rad(robot_description_dict["link2_rotation_angle"]),  # slight rotation
            name=scaled_tube_design["name"],
            mounting_length_start=scaled_tube_design["mounting_length_start"],
            mounting_length_end=scaled_tube_design["mounting_length_end"],
        )
    else:
        raise ValueError(f"Unknown link2_type: {robot_description_dict['link2_type']}")

    j3 = JointOrphogonal(
        cylinder_length1=0.09218,
        cylinder_radius1=0.042 ,
        cylinder_length2=0.028 + 0.001,  # offset for the joint gap 3-4
        cylinder_radius2=0.042,
        angle=np.deg2rad(robot_description_dict["joint3_rotation_angle"]),  # for the rotation of the fixed side of the joint
        name="joint3",
        control_mode=robot_description_dict["control_mode"],  # Use the default control mode
        armature=0.01, damping=100000, frictionloss=0.000001,
        joint_id=3,
    )

    j4 = JointOrphogonal(
        cylinder_length1=0.09218,
        cylinder_radius1=0.042,
        cylinder_length2=0.028 + 0.001,  # offset for the joint gap 4-tube
        cylinder_radius2=0.042,
        angle=np.pi + np.deg2rad(robot_description_dict["joint4_rotation_angle"]),  # for the rotation of the fixed side of the joint
        name="joint4",
        control_mode=robot_description_dict["control_mode"],  # Use the default control mode
        armature=0.01, damping=100000, frictionloss=0.000001,
        joint_id=4,
    )

    # Define link 3:
    if robot_description_dict["link3_type"] == "straight_tube":
        link3 = StraightTube(
            cylinder_length=robot_description_dict["l_link3"] + 0.008,  # offset for the motor 3
            cylinder_radius=0.035,
            name="link3",
            # attachment_quat=[1,0,0,1], # Rotate attachment point to be horizontal
        )
    elif robot_description_dict["link3_type"] == "bspline_tube_slightly_rotated":
        # slightly rotate:
        scaled_tube_design = {  # Coordinates already in meters
            "control_points": [
                Vector3([0.0, 0.0, 0.0]),
                Vector3([0.0, 0.14, 0.0]),
                Vector3([0.0, 0.28, 0.0]),
            ],
            "degree": 2,
            "num_segments": 10,
            "cylinder_radius": 0.035,
            "name": "link3",
            "mounting_length_start": 0.06,
            "mounting_length_end": 0.06,
        }
        
        link3 = BSplineTube(
            control_points=scaled_tube_design["control_points"],
            degree=scaled_tube_design["degree"],
            num_segments=scaled_tube_design["num_segments"],
            cylinder_radius=scaled_tube_design["cylinder_radius"],
            angle=np.deg2rad(robot_description_dict["link3_rotation_angle"]),  # slight rotation
            name=scaled_tube_design["name"],
            mounting_length_start=scaled_tube_design["mounting_length_start"],
            mounting_length_end=scaled_tube_design["mounting_length_end"],
        )
    elif robot_description_dict["link3_type"] == "bspline_tube_rotate_90":
        # slightly rotate:
        scaled_tube_design = {  # Coordinates already in meters
            "control_points": [
                Vector3([0.0, 0.0, 0.0]),
                Vector3([0.0, 0.0, 0.1]),
                Vector3([0.0, 0.1, 0.1]),
                Vector3([0.0, 0.14, 0.14]),
            ],
            "degree": 3,
            "num_segments": 20,
            "cylinder_radius": 0.035,
            "name": "link3",
            "mounting_length_start": 0.06,
            "mounting_length_end": 0.06,
        }
        
        link3 = BSplineTube(
            control_points=scaled_tube_design["control_points"],
            degree=scaled_tube_design["degree"],
            num_segments=scaled_tube_design["num_segments"],
            cylinder_radius=scaled_tube_design["cylinder_radius"],
            angle=np.deg2rad(robot_description_dict["link3_rotation_angle"]),  # slight rotation
            attachment_rotate_angle=np.deg2rad(90.0),
            name=scaled_tube_design["name"],
            mounting_length_start=scaled_tube_design["mounting_length_start"],
            mounting_length_end=scaled_tube_design["mounting_length_end"],
        )
    else:
        raise ValueError(f"Unknown link3_type: {robot_description_dict['link3_type']}")

    j5 = JointOrphogonal(
        cylinder_length1=0.096,
        cylinder_radius1=0.042,
        cylinder_length2=0.026 + 0.001,  # offset for the joint gap 5-6
        cylinder_radius2=0.042,
        angle=np.deg2rad(robot_description_dict["joint5_rotation_angle"]),  # for the rotation of the fixed side of the joint
        name="joint5",
        control_mode=robot_description_dict["control_mode"],  # Use the default control mode
        armature=0.01, damping=100000, frictionloss=0.000001,
        joint_id=5,
    )

    j6 = JointOrphogonal(
        cylinder_length1=0.096,
        cylinder_radius1=0.042 ,
        cylinder_length2=0.026 + 0.001,
        cylinder_radius2=0.042,
        angle=np.deg2rad(robot_description_dict["joint6_rotation_angle"]),  # np.pi # for the rotation of the fixed side of the joint
        name="joint6",
        control_mode=robot_description_dict["control_mode"],  # Use the default control mode
        armature=0.01, damping=100000, frictionloss=0.000001,
        joint_id=6,
    )

    ee_cyl = StraightTube(
        cylinder_length=0.033,
        cylinder_radius=0.01,
        name="ee_cylinder",
    )

    ee = EndEffector(
        name="end_effector",
    )

    start = Vector3([0.0, 0.0, 0.0])
    end = Vector3([0.0, 0.0, robot_description_dict["l_link2"]])
    control_offset = Vector3([0.2, 0.0, 0.1])
    control_points1 = [
        start,
        start + control_offset,
        end - control_offset,
        end
    ]

    start = Vector3([0.0, 0.0, 0.0])
    end = Vector3([0.0, 0.0, robot_description_dict["l_link3"]])
    control_offset = Vector3([0.6, 0.0, 0.1])
    control_points2 = [
        start,
        start + control_offset,
        end - control_offset,
        end
    ]
    
    # bz1 = BezierTube(control_points1,
    #                 num_segments=3,
    #                 cylinder_radius=0.042,
    #                 name="example_curve1")
    
    # bz2 = BezierTube(control_points2,
    #                 num_segments=3,
    #                 cylinder_radius=0.042,
    #                 name="example_curve2")
    
    # rotated 90 degrees about X axis
    # scaled_tube_design = {  # Coordinates already in meters
    #     "control_points": [
    #         Vector3([0.0, 0.0, 0.0]),
    #         Vector3([0.0, 0.0, 0.1]),
    #         Vector3([0.0, 0.1, 0.1]),
    #         Vector3([0.0, 0.14, 0.14]),
    #     ],
    #     "degree": 3,
    #     "num_segments": 20,
    #     "cylinder_radius": 0.035,
    #     "name": "right_angle_tube_m"
    # }

    # slightly rotate:
    # scaled_tube_design = {  # Coordinates already in meters
    #     "control_points": [
    #         Vector3([0.0, 0.0, 0.0]),
    #         Vector3([0.0, 0.14, 0.0]),
    #         Vector3([0.0, 0.28, 0.0]),
    #     ],
    #     "degree": 2,
    #     "num_segments": 10,
    #     "cylinder_radius": 0.035,
    #     "name": "straight_tube_m"
    # }
    
    # bs1 = BSplineTube(
    #     control_points=scaled_tube_design["control_points"],
    #     degree=scaled_tube_design["degree"],
    #     num_segments=scaled_tube_design["num_segments"],
    #     cylinder_radius=scaled_tube_design["cylinder_radius"],
    #     angle=np.pi,  # slight rotation
    #     name=scaled_tube_design["name"]
    # )
    
    # root.attach = bz
    root.attach = j1
    j1.attach = j2
    j2.attach = link2
    link2.attach = j3
    j3.attach = j4
    j4.attach = link3
    link3.attach = j5
    # print(f"link3: {link3}")
    j5.attach = j6
    j6.attach = ee_cyl
    ee_cyl.attach = ee
    return root


def lynx_manipulator_bspline(
        robot_description_dict: dict = {
            "l_link2": 0.2805  + 0.0035 * 2,  # offset of the two clamps
            "l_link3": 0.2805  + 0.0035 * 2,
            "link2_type": "straight_tube",  # "straight_tube", "bspline_tube_slightly_rotated", "bspline_tube_rotate_90"
            "link3_type": "straight_tube",
            "joint1_rotation_angle": 0.0,
            "joint2_rotation_angle": 0.0,
            "joint3_rotation_angle": 0.0,
            "joint4_rotation_angle": 0.0,
            "joint5_rotation_angle": 0.0,
            "joint6_rotation_angle": 0.0,
            "control_mode": "position",
            "clamp_length": 0.045,
            },
        ):
    
    print(f"robot_description_dict: {robot_description_dict}")
    
    bspline1_params = { # Slight bend
        "control_points" : [
            Vector3([0.0, 0.0, 0.0]),
            Vector3([0.0, 0.07, 0.0]),  # 0.0, 0.14, 0.0
            Vector3([0.0, 0.14, 0.0]),  # 0.0, 0.28, 0.0
        ],
        "degree": 2,
        "num_segments": 10,
        "cylinder_radius": 0.0416,
        # "name": "bspline1",
        # "angle": 0.0,  # Rotate 45 degrees around the attachment axis at the base
    }

    bspline2_params = { # Smooth right angle
        "control_points" : [
            Vector3([0.0, 0.0, 0.0]),
            Vector3([0.0, 0.0, 0.1]),
            Vector3([0.0, 0.1, 0.1]),
            Vector3([0.0, 0.14, 0.14]),
        ],
        "degree": 3,
        "num_segments": 10,
        "cylinder_radius": 0.0416,
        # "name": "bspline2"
    }

    if robot_description_dict["link2_type"] == "straight_tube":
        tube1 = StraightTube(
            cylinder_length=robot_description_dict["l_link2"],  # TODO: offset of the orthogonal joints are missing
            cylinder_radius=0.035,
            name="tube1",
            # attachment_quat=[1,0,0,1], # Rotate attachment point to be horizontal
        )
    elif robot_description_dict["link2_type"] == "bspline_tube_slightly_rotated":
        tube1 = BSplineTube(
            control_points=copy.deepcopy(bspline1_params["control_points"]),
            degree=copy.deepcopy(bspline1_params["degree"]),
            num_segments=copy.deepcopy(bspline1_params["num_segments"]),
            cylinder_radius=copy.deepcopy(bspline1_params["cylinder_radius"]),
            mounting_length_start=robot_description_dict.get("clamp_length", 0.045),
            mounting_length_end=robot_description_dict.get("clamp_length", 0.045),
            pre_joint_radius=0.062,
            next_joint_radius=0.042,
            end_point_pos=Vector3([0.0, 0.0, 0.36]),  # Optional end point position constraint
            end_point_ori=Vector3([0.0, 1.0, 0.0]),  # Optional end point orientation constraint
            dual_point_distance=0.15,  # Distance for dual control points when setting end constraints
            name="tube1",
            angle=np.deg2rad(robot_description_dict.get("joint2_rotation_angle", 0.0)),  # Default to 0.0 if not provided
        )

    elif robot_description_dict["link2_type"] == "bspline_tube_rotate_90":
        tube1 = BSplineTube(
            control_points=copy.deepcopy(bspline2_params["control_points"]),
            degree=copy.deepcopy(bspline2_params["degree"]),
            num_segments=copy.deepcopy(bspline2_params["num_segments"]),
            cylinder_radius=copy.deepcopy(bspline2_params["cylinder_radius"]),
            mounting_length_start=robot_description_dict.get("clamp_length", 0.045),
            mounting_length_end=robot_description_dict.get("clamp_length", 0.045),
            pre_joint_radius=0.062,
            next_joint_radius=0.042,
            end_point_pos=Vector3([0.0, 0.0, 0.36]),  # Optional end point position constraint
            end_point_ori=Vector3([0.0, 1.0, 0.0]),  # Optional end point orientation constraint
            dual_point_distance=0.15,  # Distance for dual control points when setting end constraints
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
    elif robot_description_dict["link3_type"] == "bspline_tube_slightly_rotated":
        tube2 = BSplineTube(
            control_points=copy.deepcopy(bspline1_params["control_points"]),
            degree=copy.deepcopy(bspline1_params["degree"]),
            num_segments=copy.deepcopy(bspline1_params["num_segments"]),
            cylinder_radius=copy.deepcopy(bspline1_params["cylinder_radius"]),
            mounting_length_start=robot_description_dict.get("clamp_length", 0.045),
            mounting_length_end=robot_description_dict.get("clamp_length", 0.045),
            pre_joint_radius=0.042,
            next_joint_radius=0.042,
            end_point_pos=Vector3([0.0, 0.0, 0.6]),  # Optional end point position constraint
            end_point_ori=Vector3([0.0, 1.0, 0.0]),  # Optional end point orientation constraint
            dual_point_distance=0.15,  # Distance for dual control points when setting end constraints
            name="tube2",
            angle=0,  # Default to 0.0 if not provided
        )
    elif robot_description_dict["link3_type"] == "bspline_tube_rotate_90":
        tube2 = BSplineTube(
            control_points=copy.deepcopy(bspline2_params["control_points"]),
            degree=copy.deepcopy(bspline2_params["degree"]),
            num_segments=copy.deepcopy(bspline2_params["num_segments"]),
            cylinder_radius=copy.deepcopy(bspline2_params["cylinder_radius"]),
            mounting_length_start=robot_description_dict.get("clamp_length", 0.045),
            mounting_length_end=robot_description_dict.get("clamp_length", 0.045),
            pre_joint_radius=0.042,
            next_joint_radius=0.042,
            end_point_pos=Vector3([0.0, 0.0, 0.36]),  # Optional end point position constraint
            end_point_ori=Vector3([0.0, 1.0, 0.0]),  # Optional end point orientation constraint
            dual_point_distance=0.15,  # Distance for dual control points when setting end constraints
            name="tube2",
            angle=0,  # Default to 0.0 if not provided
        )

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