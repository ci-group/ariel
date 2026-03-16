import numpy as np
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.module import Module, AttachmentPoint
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.unbuilt_child import UnbuiltChild
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.math_utils import Vector3, Quaternion, ensure_list, angle_to_quaternion


class JointInline(Module):

    """
    Joint with a fixed part and a rotating part. The rotating part is aligned inline with the parent attachment pos.
    The joint rotates around the z-axis of the attachment point.
    Params:
        cylinder_length0: Length of the rotating part of the joint.
        cylinder_radius0: Radius of the rotating part of the joint.
        cylinder_length1: Length of the main part of the joint.
        cylinder_radius1: Radius of the main part of the joint.
        cylinder_length2: Length of the fixed part of the joint.
        cylinder_radius2: Radius of the fixed part of the joint.
        angle: Angle of the fixed part in radians.
        name: Name of the joint module.
        control_mode: Control mode for the joint ("position" or "velocity").
    """

    ATTACH = 0
    GROUP_TUBE = 2
    GROUP_JOINT = 1
    GROUP_ENV = 0

    def __init__(
        self,
        cylinder_length0 : float = 0.5,
        cylinder_radius0 : float = 0.05,
        cylinder_length1 : float = 0.5,
        cylinder_radius1 : float = 0.05,
        cylinder_length2 : float = 0.5,
        cylinder_radius2 : float = 0.05,
        angle : float = 0.0, # Angle in radians
        name: str = None,
        control_mode: str = "position",
        armature: float = None, # New: Joint armature
        damping: float = None,  # New: Joint damping
        frictionloss: float = None, # New: Joint frictionloss
        kp: float = 5000000, # New: Position control stiffness
        kd: float = 1, # New: Velocity control damping
        collision_type: str = "capsule",
        collision_alpha: float = 0.3,
        *args, **kwargs,
    ):  
        self.cylinder_length0 = cylinder_length0
        self.cylinder_radius0 = cylinder_radius0
        self.cylinder_length1 = cylinder_length1
        self.cylinder_radius1 = cylinder_radius1
        self.cylinder_length2 = cylinder_length2
        self.cylinder_radius2 = cylinder_radius2
        self.collision_type = collision_type
        self.collision_alpha = collision_alpha

        self.angle = angle  # Angle in radians
        # self.fixed_part_angle = fixed_part_angle
        self.name = name if name is not None else "joint"
        self.control_mode = control_mode
        self.armature = armature
        self.damping = damping
        self.frictionloss = frictionloss
        self.kp = kp
        self.kd = kd

        attachment_points = { 
            self.ATTACH: AttachmentPoint(
                offset=Vector3([0.0, self.cylinder_length0+self.cylinder_length1/2, 0.0]),
                orientation=Quaternion([1,0,0,1]),  # Rotate around the z-axis
            ),
        }

        super().__init__(attachment_points)

    @property
    def attach(self) -> Module:
        return self._children.get(self.ATTACH)

    @attach.setter
    def attach(self, module: Module) -> None:
        self.set_child(module, self.ATTACH)

    def build(self, mjcf, entry_point, attachment_point_pos, attachment_point_quat) -> list:

        # Convert attachment_point_pos to list if it's a Vector3 object
        pos_list = ensure_list(attachment_point_pos)

        # Convert to our custom Quaternion class if needed
        if not isinstance(attachment_point_quat, Quaternion):
            attachment_point_quat = Quaternion(attachment_point_quat)
            
        # Normalize the quaternion to ensure it is a valid rotation
        attachment_point_quat.normalise()

        # Build the module
        mj_quat_body = attachment_point_quat.to_mujoco_format()

        rx      = angle_to_quaternion(np.pi/2, [1, 0, 0])
        twist   = angle_to_quaternion(self.angle, [0, 0, 1])
        new_relative_quat = rx * twist

        new_relative_quat.normalise()
        mj_quat2 = new_relative_quat.to_mujoco_format()

        joint_pos = Vector3([0,0,self.cylinder_length0+self.cylinder_length1/2])
        # new_pos1 = Vector3([0,0,self.length/2])
        cylinder2_pos = joint_pos + new_relative_quat * Vector3([0, 0, self.cylinder_radius1+ self.cylinder_length2/2])
        cylinder2_add_pos = joint_pos + new_relative_quat * Vector3([0, 0, self.cylinder_radius1/2])
        print(f"JointInline build: joint_pos={joint_pos}, cylinder2_pos={cylinder2_pos}")

        body_fixed = entry_point.add("body", name=self.name+"_fixed", pos=pos_list, quat=mj_quat_body)
        body_fixed.add(
            "geom", 
            type="cylinder", 
            size=[self.cylinder_radius0, self.cylinder_length0/2], 
            pos=Vector3([0,0,self.cylinder_length0/2]).to_list(), 
            group=self.GROUP_JOINT, contype=0, conaffinity=0,
            rgba=[0.3,0.3,0.3,1], 
        )

        body = body_fixed.add("body", name=self.name, pos=ensure_list(Vector3([0,0,0])), quat=Quaternion([0,0,0,1]).to_mujoco_format())

        # Add joint with optional armature, damping, frictionloss
        joint_attrs = {"name": self.name+"_joint", "axis": [0, 0, -1], "pos": ensure_list(joint_pos)}
        if self.armature is not None:
            joint_attrs["armature"] = self.armature
        if self.damping is not None:
            joint_attrs["damping"] = self.damping
        if self.frictionloss is not None:
            joint_attrs["frictionloss"] = self.frictionloss
        body.add("joint", **joint_attrs)

        if self.control_mode == "velocity":
            mjcf.actuator.add("velocity", kv=str(self.kd), joint=self.name+"_joint")
        elif self.control_mode == "position":
            mjcf.actuator.add("position", kp=str(self.kp), kv=str(self.kd), joint=self.name+"_joint")
        else:
            raise ValueError(f"Unsupported control mode: {self.control_mode}")
        # <geom type="cylinder" size="0.075 0.0829" pos="0 0 0" material="jointgray" class="visual"/> 
        # body.add(
        #     "geom", 
        #     type="cylinder", 
        #     group=self.GROUP_JOINT,
        #     contype=1 << self.GROUP_JOINT,
        #     conaffinity=(1 << self.GROUP_JOINT) | (1 << self.GROUP_TUBE) | (1 << self.GROUP_ENV),
        #     size=[self.cylinder_radius0, self.cylinder_length0 / 2], 
        #     pos=ensure_list(Vector3([0, 0, self.cylinder_length0 / 2])), 
        #     rgba=[0.3,0.3,0.3,1], 
        #     name="lynxg0_"+self.name
        # ) # 

        body.add(
            "geom", 
            type="cylinder", 
            group=self.GROUP_JOINT,
            contype=0,
            conaffinity=0,
            size=[self.cylinder_radius1, self.cylinder_length1 / 2], 
            pos=ensure_list(joint_pos), 
            rgba=[0,0,0,1], 
            name="lynxg1_"+self.name
        ) # 

        # for structural alignment:
        body.add(
            "geom", 
            type="cylinder", 
            group=self.GROUP_JOINT,
            contype=0,
            conaffinity=0,
            size=[self.cylinder_radius2, self.cylinder_radius1 / 2], 
            pos=ensure_list(cylinder2_add_pos), 
            rgba=[0,0,0,1], 
            quat=mj_quat2, 
            # name="lynxg2_"+self.name
        ) # 

        body.add(
            "geom", 
            type="cylinder", 
            group=self.GROUP_JOINT,
            contype=0,
            conaffinity=0,
            size=[self.cylinder_radius2, self.cylinder_length2 / 2], 
            pos=ensure_list(cylinder2_pos), 
            rgba=[0,0,0,1], 
            quat=mj_quat2, 
            name="lynxg2_"+self.name
        ) # 

        # Additional collision model for the joint (capsule covering the main joint area)
        collision_size = [self.cylinder_radius1 * 1.1, self.cylinder_length1 / 2 * 0.58]
        if self.collision_type == "cylinder":
            collision_size = [self.cylinder_radius1 * 1.1, self.cylinder_length1 / 2 * 1.1]

        body.add(
            "geom",
            type=self.collision_type,
            size=collision_size,
            pos=ensure_list(joint_pos),
            group=self.GROUP_JOINT,
            contype=1 << self.GROUP_JOINT,
            conaffinity=(1 << self.GROUP_JOINT) | (1 << self.GROUP_TUBE) | (1 << self.GROUP_ENV),
            rgba=[1, 0, 0, self.collision_alpha], # Semi-transparent red for visualization
            name=self.name+"_collision_geom"
        )

        # Find the children
        tasks = []

        attachment_site_pos = cylinder2_pos + new_relative_quat * Vector3([0, 0, self.cylinder_length2 / 2])
        print(f"JointInline build: attachment_site_pos={attachment_site_pos}")

        for child_index, attachment_point in self.attachment_points.items():
            child = self.children.get(child_index)
            if child is not None:
                unbuilt = UnbuiltChild(module=child, mj_body=body)
                unbuilt.make_pose(
                    position=attachment_site_pos, # Prev pos + offset rotated in right direction = attachment point pos
                    orientation=new_relative_quat,  # new_quat2
                )
                tasks.append(unbuilt)

        return tasks


class JointOrthogonal(Module):

    ATTACH = 0
    GROUP_TUBE = 2
    GROUP_JOINT = 1
    GROUP_ENV = 0

    def __init__(
        self,
        cylinder_length0 : float = 0.5,
        cylinder_radius0 : float = 0.05,
        cylinder_length1 : float = 0.5,
        cylinder_radius1 : float = 0.05,
        cylinder_length2 : float = 0.5,
        cylinder_radius2 : float = 0.05,
        angle : float = 0.0, # Angle in radians
        # fixed_part_angle: float = 0.0,
        name: str = None,
        control_mode: str = "position",
        armature: float = None, # New: Joint armature
        damping: float = None,  # New: Joint damping
        frictionloss: float = None, # New: Joint frictionloss
        kp: float = 5000000, # New: Position control stiffness
        kd: float = 1, # New: Velocity control damping
        collision_type: str = "capsule",
        collision_alpha: float = 0.3,
        *args, **kwargs,
    ):  
        # the name of the joints are reversed compared to the inline joints
        self.cylinder_length2 = cylinder_length0
        self.cylinder_radius2 = cylinder_radius0
        self.cylinder_length1 = cylinder_length1
        self.cylinder_radius1 = cylinder_radius1
        self.cylinder_length0 = cylinder_length2
        self.cylinder_radius0 = cylinder_radius2
        self.collision_type = collision_type
        self.collision_alpha = collision_alpha
        self.length = cylinder_length1
        self.radius = cylinder_radius1 + cylinder_length2
        self.angle = angle  # Angle in radians
        # self.fixed_part_angle = fixed_part_angle
        self.name = name if name is not None else "joint"

        self.control_mode = control_mode
        self.armature = armature
        self.damping = damping
        self.frictionloss = frictionloss
        self.kp = kp
        self.kd = kd

        attachment_points = { 
            self.ATTACH: AttachmentPoint(
                offset=Vector3([0.0, self.cylinder_length0+self.cylinder_length1/2, 0.0]),
                orientation=Quaternion([1,0,0,1]),  # Rotate around the z-axis
            ),
        }

        super().__init__(attachment_points)

    @property
    def attach(self) -> Module:
        return self._children.get(self.ATTACH)

    @attach.setter
    def attach(self, module: Module) -> None:
        self.set_child(module, self.ATTACH)

    def build(self, mjcf, entry_point, attachment_point_pos, attachment_point_quat) -> list:

        # Convert attachment_point_pos to list if it's a Vector3 object
        pos_list = ensure_list(attachment_point_pos)

        # Convert to our custom Quaternion class if needed
        if not isinstance(attachment_point_quat, Quaternion):
            attachment_point_quat = Quaternion(attachment_point_quat)
            
        # Normalize the quaternion to ensure it is a valid rotation
        attachment_point_quat.normalise()
        mj_attachment_point_quat = attachment_point_quat.to_mujoco_format()

        # Build the module
        rx      = angle_to_quaternion(np.pi/2, [1, 0, 0])
        twist   = angle_to_quaternion(self.angle, [0, 0, 1])
        new_relative_quat = twist * rx
        axis = new_relative_quat * Vector3([0, 0, -1])

        cylinder0_pos = Vector3([0,0,self.cylinder_length0/2])
        cylinder0_add_pos = Vector3([0,0,self.cylinder_length0 + self.cylinder_radius1/2])
        joint_pos_related_to_body = Vector3([0,0,self.cylinder_length0+self.cylinder_radius1])
        cylinder2_pos_related_to_body = joint_pos_related_to_body + new_relative_quat * Vector3([0, 0, self.cylinder_length1/2 + self.cylinder_length2/2])

        body_fixed = entry_point.add("body", name=self.name+"_fixed", pos=pos_list, quat=mj_attachment_point_quat)
        # add the additional part of the fixed side:
        body_fixed.add(
            "geom", 
            type="cylinder", 
            size=[self.cylinder_radius0, self.cylinder_length0/2], 
            pos=cylinder0_pos.to_list(), 
            group=self.GROUP_JOINT, contype=0, conaffinity=0,
            rgba=[0,0,0,1], 
            )
        # add the structural alignment part:
        body_fixed.add(
            "geom",
            type="cylinder",
            size=[self.cylinder_radius0, self.cylinder_radius1/2],
            pos=cylinder0_add_pos.to_list(),
            group=self.GROUP_JOINT, contype=0, conaffinity=0,
            rgba=[0,0,0,1], 
            )
        # add the shell of the joint:
        body_fixed.add(
            "geom",
            type="cylinder",
            size=[self.cylinder_radius1, self.cylinder_length1/2],
            pos=joint_pos_related_to_body.to_list(),
            quat=new_relative_quat.to_mujoco_format(),
            group=self.GROUP_JOINT, contype=0, conaffinity=0,
            rgba=[0,0,0,1], 
        )
        # always relevant quat when creating inner body/geom!
        body = body_fixed.add("body", name=self.name, pos=ensure_list(Vector3([0,0,0])), quat=Quaternion([0,0,0,1]).to_mujoco_format())

        # body = entry_point.add("body", name=self.name, pos=pos_list, quat=mj_quat1)

        joint_attrs = {"name": self.name+"_joint", "axis": axis.to_list(), "pos": ensure_list(joint_pos_related_to_body), "type": "hinge"}
        if self.armature is not None:
            joint_attrs["armature"] = self.armature
        if self.damping is not None:
            joint_attrs["damping"] = self.damping
        if self.frictionloss is not None:
            joint_attrs["frictionloss"] = self.frictionloss
        body.add("joint", **joint_attrs)

        # <joint name="shoulder_pan_joint" class="size4" axis="0 0 1"/>

        if self.control_mode == "velocity":
            mjcf.actuator.add("velocity", kv=str(self.kd), joint=self.name+"_joint")
        elif self.control_mode == "position":
            mjcf.actuator.add("position", kp=str(self.kp), kv=str(self.kd), joint=self.name+"_joint")
        else:
            raise ValueError(f"Unsupported control mode: {self.control_mode}")

        # add the motor (movable) part of the joint:
        body.add(
            "geom", 
            type="cylinder",
            group=self.GROUP_JOINT,
            contype=0,
            conaffinity=0,
            size=[self.cylinder_radius2, self.cylinder_length2 / 2], 
            pos=ensure_list(cylinder2_pos_related_to_body), 
            rgba=[0.3,0.3,0.3,1], 
            # just no relative rotation:
            quat=new_relative_quat.to_mujoco_format(), 
            name="lynxg2_"+self.name
        ) # 

        # Additional collision model for the joint (capsule covering the main joint area)
        collision_size = [self.cylinder_radius1 * 1.10, self.cylinder_length1 / 2 * 0.58]
        if self.collision_type == "cylinder":
            collision_size = [self.cylinder_radius1 * 1.10, self.cylinder_length1 / 2 * 1.1]

        body.add(
            "geom",
            type=self.collision_type,
            size=collision_size,
            pos=ensure_list(joint_pos_related_to_body),
            quat=new_relative_quat.to_mujoco_format(),
            group=self.GROUP_JOINT,
            contype=1 << self.GROUP_JOINT,
            conaffinity=(1 << self.GROUP_JOINT) | (1 << self.GROUP_TUBE) | (1 << self.GROUP_ENV),
            rgba=[1, 0, 0, self.collision_alpha], # Semi-transparent red for visualization
            name=self.name+"_collision_geom"
        )

        # Find the children
        tasks = []

        attachment_site_pos = cylinder2_pos_related_to_body + new_relative_quat * Vector3([0, 0, self.cylinder_length2/2])
        for child_index, attachment_point in self.attachment_points.items():
            child = self.children.get(child_index)
            if child is not None:
                unbuilt = UnbuiltChild(module=child, mj_body=body)
                unbuilt.make_pose(
                    position=attachment_site_pos, # Prev pos + offset rotated in right direction = attachment point pos
                    orientation=new_relative_quat,
                    # orientation=attachment_point_quat,
                    # orientation=Quaternion([0,0,0,1]), # Combine the joint rotation with the attachment point rotation
                )
                tasks.append(unbuilt)

        return tasks
