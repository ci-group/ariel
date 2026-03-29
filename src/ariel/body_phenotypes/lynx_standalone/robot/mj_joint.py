import mujoco
import numpy as np
from typing import Optional
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.mj_module import MjModule
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.math_utils import angle_to_quaternion, Vector3

class MjJointInline(MjModule):
    """
    Inline joint module using MjSpec.
    """
    def __init__(
        self,
        cylinder_length0: float = 0.5,
        cylinder_radius0: float = 0.05,
        cylinder_length1: float = 0.5,
        cylinder_radius1: float = 0.05,
        cylinder_length2: float = 0.5,
        cylinder_radius2: float = 0.05,
        angle: float = 0.0,
        name: str = "joint_inline",
        control_mode: str = "position",
        # MuJoCo compiler angle unit is typically degrees; these map to ~+-2.79 rad.
        joint_range: tuple[float, float] = (-160.0, 160.0),
        armature: Optional[float] = None,
        damping: Optional[float] = None,
        frictionloss: Optional[float] = None,
        pos=None,
        quat=None
    ):
        super().__init__(name=name, pos=pos, quat=quat)
        
        self.cylinder_length0 = cylinder_length0
        self.cylinder_radius0 = cylinder_radius0
        self.cylinder_length1 = cylinder_length1
        self.cylinder_radius1 = cylinder_radius1
        self.cylinder_length2 = cylinder_length2
        self.cylinder_radius2 = cylinder_radius2
        self.angle = angle

        # Inline configuration
        joint_pos = [0, 0, self.cylinder_length0 + self.cylinder_length1 / 2]
        
        # Geoms for stator
        self.body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[self.cylinder_radius0, self.cylinder_length0 / 2, 0],
            pos=[0, 0, self.cylinder_length0 / 2],
            rgba=[0.3, 0.3, 0.3, 1],
            group=1
        )

        # Rotor body
        self.rotor = self.body.add_body(name=f"{self.name}_rotor")
        
        # Add joint to rotor
        joint = self.rotor.add_joint(
            name=f"{self.name}_joint",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            pos=joint_pos,
            axis=[0, 0, -1]
        )
        joint.range = [joint_range[0], joint_range[1]]
        joint.limited = True
        if armature is not None: joint.armature = armature
        if damping is not None: joint.damping = damping
        if frictionloss is not None: joint.frictionloss = frictionloss

        # Add actuator
        dynprm = np.zeros(10)
        gainprm = np.zeros(10)
        biasprm = np.zeros(10)

        if control_mode == "position":
            # High gains can cause aggressive penetration in contact-rich scenes.
            kp = 300
            kv = 30
            gainprm[0] = kp
            biasprm[:3] = [0, -kp, -kv]
            
            self.spec.add_actuator(
                name=f"{self.name}_actuator",
                dyntype=mujoco.mjtDyn.mjDYN_NONE,
                gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
                target=f"{self.name}_joint",
                dynprm=dynprm,
                gainprm=gainprm,
                biasprm=biasprm,
                ctrllimited=True,
                ctrlrange=[-2.8, 2.8],
            )
        elif control_mode == "velocity":
            self.spec.add_actuator(
                name=f"{self.name}_actuator",
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
                target=f"{self.name}_joint"
            )

        # Geoms for rotor
        self.rotor.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[self.cylinder_radius1, self.cylinder_length1 / 2, 0],
            pos=joint_pos,
            rgba=[0, 0, 0, 1],
            group=1
        )

        # Output part rotation
        rx = angle_to_quaternion(np.pi/2, [1, 0, 0])
        twist = angle_to_quaternion(self.angle, [0, 0, 1])
        rel_quat = (rx * twist).to_mujoco_format()

        # Calculate cylinder2_pos manually for the geom
        q = rx * twist
        offset_vec = q * Vector3([0, 0, self.cylinder_radius1 + self.cylinder_length2 / 2])
        cylinder2_pos = [joint_pos[i] + offset_vec[i] for i in range(3)]

        self.rotor.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[self.cylinder_radius2, self.cylinder_length2 / 2, 0],
            pos=cylinder2_pos,
            quat=rel_quat,
            rgba=[0, 0, 0, 1],
            group=1
        )

        # Attachment site on rotor
        site_offset_vec = q * Vector3([0, 0, self.cylinder_radius1 + self.cylinder_length2])
        site_pos = [joint_pos[i] + site_offset_vec[i] for i in range(3)]
        self.rotor.add_site(name="attach", pos=site_pos, quat=rel_quat)

    def rotate(self, angle: float) -> None:
        self.angle = angle
        pass

class MjJointOrthogonal(MjModule):
    """
    Orthogonal joint module using MjSpec.
    """
    def __init__(
        self,
        cylinder_length0: float = 0.5,
        cylinder_radius0: float = 0.05,
        cylinder_length1: float = 0.5,
        cylinder_radius1: float = 0.05,
        cylinder_length2: float = 0.5,
        cylinder_radius2: float = 0.05,
        angle: float = 0.0,
        name: str = "joint_orthogonal",
        control_mode: str = "position",
        # MuJoCo compiler angle unit is typically degrees; these map to ~+-2.79 rad.
        joint_range: tuple[float, float] = (-160.0, 160.0),
        armature: Optional[float] = None,
        damping: Optional[float] = None,
        frictionloss: Optional[float] = None,
        pos=None,
        quat=None
    ):
        super().__init__(name=name, pos=pos, quat=quat)
        
        self.cylinder_length0 = cylinder_length0
        self.cylinder_radius0 = cylinder_radius0
        self.cylinder_length1 = cylinder_length1
        self.cylinder_radius1 = cylinder_radius1
        self.cylinder_length2 = cylinder_length2
        self.cylinder_radius2 = cylinder_radius2
        self.angle = angle

        # Orthogonal configuration (swapping lengths as in original code)
        l2, r2 = self.cylinder_length0, self.cylinder_radius0
        l1, r1 = self.cylinder_length1, self.cylinder_radius1
        l0, r0 = self.cylinder_length2, self.cylinder_radius2

        cylinder0_pos = [0, 0, l0 / 2]
        joint_pos = [0, 0, l0 + r1]
        
        # Geoms for stator
        self.body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[r0, l0 / 2, 0],
            pos=cylinder0_pos,
            rgba=[0, 0, 0, 1],
            group=1
        )

        # Rotor body
        self.rotor = self.body.add_body(name=f"{self.name}_rotor")
        
        rx = angle_to_quaternion(np.pi/2, [1, 0, 0])
        twist = angle_to_quaternion(self.angle, [0, 0, 1])
        rel_quat = (twist * rx)
        axis = (rel_quat * Vector3([0, 0, 1])).to_list()

        # Add joint to rotor
        joint = self.rotor.add_joint(
            name=f"{self.name}_joint",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            pos=joint_pos,
            axis=axis
        )
        joint.range = [joint_range[0], joint_range[1]]
        joint.limited = True
        if armature is not None: joint.armature = armature
        if damping is not None: joint.damping = damping
        if frictionloss is not None: joint.frictionloss = frictionloss

        # Add actuator
        dynprm = np.zeros(10)
        gainprm = np.zeros(10)
        biasprm = np.zeros(10)

        if control_mode == "position":
            # High gains can cause aggressive penetration in contact-rich scenes.
            kp = 300
            kv = 30
            gainprm[0] = kp
            biasprm[:3] = [0, -kp, -kv]
            
            self.spec.add_actuator(
                name=f"{self.name}_actuator",
                dyntype=mujoco.mjtDyn.mjDYN_NONE,
                gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
                target=f"{self.name}_joint",
                dynprm=dynprm,
                gainprm=gainprm,
                biasprm=biasprm,
                ctrllimited=True,
                ctrlrange=[-2.8, 2.8],
            )
        elif control_mode == "velocity":
            self.spec.add_actuator(
                name=f"{self.name}_actuator",
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
                target=f"{self.name}_joint"
            )

        # Geoms for rotor
        self.rotor.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[r1, l1 / 2, 0],
            pos=joint_pos,
            quat=rel_quat.to_mujoco_format(),
            rgba=[0, 0, 0, 1],
            group=1
        )

        cylinder2_offset = rel_quat * Vector3([0, 0, l1 / 2 + l2 / 2])
        cylinder2_pos = [joint_pos[i] + cylinder2_offset[i] for i in range(3)]

        self.rotor.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[r2, l2 / 2, 0],
            pos=cylinder2_pos,
            quat=rel_quat.to_mujoco_format(),
            rgba=[0.3, 0.3, 0.3, 1],
            group=1
        )

        # Attachment site on rotor
        site_offset = rel_quat * Vector3([0, 0, l1 / 2 + l2])
        site_pos = [joint_pos[i] + site_offset[i] for i in range(3)]
        self.rotor.add_site(name="attach", pos=site_pos, quat=rel_quat.to_mujoco_format())

    def rotate(self, angle: float) -> None:
        self.angle = angle
        pass