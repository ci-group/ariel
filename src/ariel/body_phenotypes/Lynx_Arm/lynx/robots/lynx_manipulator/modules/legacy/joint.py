import numpy as np
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.module import Module, AttachmentPoint
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.unbuilt_child import UnbuiltChild
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.math_utils import Vector3, Quaternion, ensure_list, angle_to_quaternion

class JointInline(Module):

    ATTACH = 0
    GROUP_TUBE = 2
    GROUP_JOINT = 1
    GROUP_ENV = 0

    def __init__(
        self,
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
    ):  
        
        self.cylinder_length1 = cylinder_length1
        self.cylinder_radius1 = cylinder_radius1
        self.cylinder_length2 = cylinder_length2
        self.cylinder_radius2 = cylinder_radius2
        self.length = cylinder_length1
        self.radius = cylinder_radius1 + cylinder_length2
        self.angle = angle  # Angle in radians
        # self.fixed_part_angle = fixed_part_angle
        self.name = name if name is not None else "joint"
        self.control_mode = control_mode
        self.armature = armature
        self.damping = damping
        self.frictionloss = frictionloss

        attachment_points = { 
            self.ATTACH: AttachmentPoint(
                offset=Vector3([0.0, self.length/2, 0.0]),
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

        print(f"[jointinline] with quat {attachment_point_quat}")

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
        print(f"[jointinline] rx: {rx}")
        print(f"[jointinline] twist: {twist}")
        print(f"[jointinline] new_relative_quat: {new_relative_quat}")

        new_relative_quat.normalise()
        mj_quat2 = new_relative_quat.to_mujoco_format()

        # print(f"mj quat2: {mj_quat2}")

        new_pos1 = Vector3([0,0,self.length/2])
        new_pos2 = new_pos1 + new_relative_quat * Vector3([0, 0, self.cylinder_radius1])

        body = entry_point.add("body", name=self.name, pos=pos_list, quat=mj_quat_body)

        # Add joint with optional armature, damping, frictionloss
        joint_attrs = {"name": self.name+"_joint", "axis": [0, 0, -1], "pos": ensure_list(new_pos1)}
        if self.armature is not None:
            joint_attrs["armature"] = self.armature
        if self.damping is not None:
            joint_attrs["damping"] = self.damping
        if self.frictionloss is not None:
            joint_attrs["frictionloss"] = self.frictionloss
        body.add("joint", **joint_attrs)

        if self.control_mode == "velocity":
            mjcf.actuator.add("velocity", kv="10000", joint=self.name+"_joint")
        elif self.control_mode == "position":
            mjcf.actuator.add("position", kp="100000", joint=self.name+"_joint")
        else:
            raise ValueError(f"Unsupported control mode: {self.control_mode}")
        # <geom type="cylinder" size="0.075 0.0829" pos="0 0 0" material="jointgray" class="visual"/> 
        body.add(
            "geom", 
            type="cylinder", 
            group=self.GROUP_JOINT,
            contype=1 << self.GROUP_JOINT,
            conaffinity=(1 << self.GROUP_JOINT) | (1 << self.GROUP_TUBE) | (1 << self.GROUP_ENV),
            size=[self.cylinder_radius1, self.cylinder_length1 / 2], 
            pos=ensure_list(new_pos1), 
            rgba=[0,0,0,1], 
            name="lynxg1_"+self.name
        ) # 

        # <geom type="cylinder" size="0.075 0.01" pos="0 0.085 0" quat="0 0 1 1" material="black" class="visual"/>
        body.add(
            "geom", 
            type="cylinder", 
            group=self.GROUP_JOINT,
            contype=1 << self.GROUP_JOINT,
            conaffinity=(1 << self.GROUP_JOINT) | (1 << self.GROUP_TUBE) | (1 << self.GROUP_ENV),
            size=[self.cylinder_radius2, self.cylinder_length2 / 2], 
            pos=ensure_list(new_pos2), 
            rgba=[0.05,0.05,0.05,1], 
            quat=mj_quat2, 
            name="lynxg2_"+self.name
        ) # 

        # Find the children
        tasks = []

        attachment_site_pos = new_pos2 + new_relative_quat * Vector3([0, 0, self.cylinder_length2 / 2])

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
    

class JointOrphogonal(Module):

    ATTACH = 0

    def __init__(
        self,
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
        joint_id: int = 0,
    ):  
        
        self.cylinder_length1 = cylinder_length1
        self.cylinder_radius1 = cylinder_radius1
        self.cylinder_length2 = cylinder_length2
        self.cylinder_radius2 = cylinder_radius2
        self.length = cylinder_length1
        self.radius = cylinder_radius1 + cylinder_length2
        self.angle = angle  # Angle in radians
        self.name = name if name is not None else "joint"
        self.control_mode = control_mode
        self.armature = armature
        self.damping = damping
        self.frictionloss = frictionloss
        self.joint_id = joint_id

        attachment_points = { 
            self.ATTACH: AttachmentPoint(
                offset=Vector3([0.0, self.length/2, 0.0]),
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

        # attachment_point_quat = Quaternion([0,0,0,1])
            
        # Normalize the quaternion to ensure it is a valid rotation
        attachment_point_quat.normalise()
        print(f"[jointorthogonal] joint or attachment quat: {attachment_point_quat}")

        # # Calculate the axis given the attachment point quaternion
        axis_x = attachment_point_quat * Vector3([1, 0, 0])
        axis_y = attachment_point_quat * Vector3([0, 1, 0])
        axis_z = attachment_point_quat * Vector3([0, 0, 1])


        # if self.joint_id == 4:
        #     qx = angle_to_quaternion(-np.pi/2, axis_x.to_list())
        #     new_quat = qx * attachment_point_quat
        #     qy = angle_to_quaternion(np.pi, axis_y.to_list())  # Rotate around the calculated axis
        #     new_quat = qy * new_quat
        # else:
        #     qx = angle_to_quaternion(np.pi/2, axis_x.to_list())
        #     new_quat = qx * attachment_point_quat
        qx = angle_to_quaternion(np.pi/2, axis_x.to_list())
        new_quat = qx * attachment_point_quat

        qz = angle_to_quaternion(self.angle, axis_z.to_list())  # Rotate around the calculated axis
        new_quat = qz * new_quat
        # new_quat = attachment_point_quat

        print(f"[jointorthogonal] body quat: {new_quat}")

        mj_quat1 = new_quat.to_mujoco_format()

        # 90 degree rotation around the x-axis
        new_quat2 = Quaternion([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
        new_quat2.normalise()
        axis_x = new_quat2 * Vector3([1, 0, 0])
        motor_part_rotation = angle_to_quaternion(np.pi/2, axis_x.to_list())  # np.pi/2
        new_quat2 = motor_part_rotation * new_quat2

        mj_quat2 = new_quat2.to_mujoco_format()

        new_pos1 = Vector3([0,self.cylinder_radius1,0])
        new_pos2 = new_pos1 + new_quat2 * Vector3([0, 0, -self.length/2- self.cylinder_length2/2])

        body = entry_point.add("body", name=self.name, pos=pos_list, quat=mj_quat1)

        joint_attrs = {"name": self.name+"_joint", "axis": [0, 0, -1], "pos": ensure_list(new_pos1)}
        if self.armature is not None:
            joint_attrs["armature"] = self.armature
        if self.damping is not None:
            joint_attrs["damping"] = self.damping
        if self.frictionloss is not None:
            joint_attrs["frictionloss"] = self.frictionloss
        body.add("joint", **joint_attrs)

        # <joint name="shoulder_pan_joint" class="size4" axis="0 0 1"/>

        if self.control_mode == "velocity":
            mjcf.actuator.add("velocity", kv="10000", joint=self.name+"_joint")
        elif self.control_mode == "position":
            mjcf.actuator.add("position", kp="100000", joint=self.name+"_joint")
        else:
            raise ValueError(f"Unsupported control mode: {self.control_mode}")
        # <geom type="cylinder" size="0.075 0.0829" pos="0 0 0" material="jointgray" class="visual"/> 
        body.add(
            "geom", 
            type="cylinder", 
            contype="1", 
            conaffinity="1", 
            group="1", 
            size=[self.cylinder_radius1, self.cylinder_length1 / 2], 
            pos=ensure_list(new_pos1), 
            rgba=[0,0,0,1], 
            name="lynxg1_"+self.name
            ) # 

        # <geom type="cylinder" size="0.075 0.01" pos="0 0.085 0" quat="0 0 1 1" material="black" class="visual"/>
        body.add(
            "geom", 
            type="cylinder", 
            contype="1", 
            conaffinity="1", 
            group="1", 
            size=[self.cylinder_radius2, self.cylinder_length2 / 2], 
            pos=ensure_list(new_pos2), 
            rgba=[0.05,0.05,0.05,1],
            quat=mj_quat2, 
            name="lynxg2_"+self.name
            ) # 

        # Find the children
        tasks = []

        attachment_site_pos = new_pos2 + new_quat2 * Vector3([0, 0, -self.cylinder_length2/2])
        for child_index, attachment_point in self.attachment_points.items():
            child = self.children.get(child_index)
            if child is not None:
                unbuilt = UnbuiltChild(module=child, mj_body=body)
                unbuilt.make_pose(
                    position=attachment_site_pos, # Prev pos + offset rotated in right direction = attachment point pos
                    orientation=attachment_point_quat,  # attachment_point_quat, new_quat2, Quaternion([0,0,0,1])
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
    ):  
        
        self.cylinder_length1 = cylinder_length1
        self.cylinder_radius1 = cylinder_radius1
        self.cylinder_length2 = cylinder_length2
        self.cylinder_radius2 = cylinder_radius2
        self.length = cylinder_length1
        self.radius = cylinder_radius1 + cylinder_length2
        self.angle = angle  # Angle in radians
        # self.fixed_part_angle = fixed_part_angle
        self.name = name if name is not None else "joint"

        self.control_mode = control_mode
        self.armature = armature
        self.damping = damping
        self.frictionloss = frictionloss

        attachment_points = { 
            self.ATTACH: AttachmentPoint(
                offset=Vector3([0.0, self.length/2, 0.0]),
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
        
        # Calculate the axis given the attachment point quaternion
        axis = attachment_point_quat * Vector3([0, 1, 0])
        q = angle_to_quaternion(self.angle, axis.to_list())  # Rotate around the calculated axis
        tmp_new_quat =  q*attachment_point_quat
        # Rotate 90 degrees around the x-axis
        q2 = angle_to_quaternion(np.pi/2, [1, 0, 0])
        new_quat = q2 * tmp_new_quat

        mj_quat1 = new_quat.to_mujoco_format()

        # 90 degree rotation around the x-axis
        # new_quat2 = Quaternion([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
        new_quat2 = angle_to_quaternion(np.pi/2, [1, 0, 0])
        new_quat2.normalise()
        new_quat3 = q2 * new_quat2

        mj_quat2 = new_quat3.to_mujoco_format()

        new_pos1 = Vector3([0,self.cylinder_radius1,0])
        new_pos2 = new_pos1 + new_quat3 * Vector3([0, 0, -self.length/2- self.cylinder_length2/2])

        print(f"[jointorthogonal] quat: {new_quat}, pos1: {new_pos1}, pos2: {new_pos2}")

        body = entry_point.add("body", name=self.name, pos=pos_list, quat=mj_quat1)

        # <joint name="shoulder_pan_joint" class="size4" axis="0 0 1"/>
        # body.add("joint", name=self.name+"_joint", axis=[0, 0, 1], pos=ensure_list(new_pos1), type="hinge")

        joint_attrs = {"name": self.name+"_joint", "axis": [0, 0, 1], "pos": ensure_list(new_pos1), "type": "hinge"}
        if self.armature is not None:
            joint_attrs["armature"] = self.armature
        if self.damping is not None:
            joint_attrs["damping"] = self.damping
        if self.frictionloss is not None:
            joint_attrs["frictionloss"] = self.frictionloss
        body.add("joint", **joint_attrs)

        # <joint name="shoulder_pan_joint" class="size4" axis="0 0 1"/>

        if self.control_mode == "velocity":
            mjcf.actuator.add("velocity", kv="10000", joint=self.name+"_joint")
        elif self.control_mode == "position":
            mjcf.actuator.add("position", kp="100000", joint=self.name+"_joint")
        else:
            raise ValueError(f"Unsupported control mode: {self.control_mode}")
        
        # mjcf.actuator.add("velocity", kv="10000", joint=self.name+"_joint")
        # <geom type="cylinder" size="0.075 0.0829" pos="0 0 0" material="jointgray" class="visual"/> 
        body.add(
            "geom", 
            type="cylinder", 
            group=self.GROUP_JOINT,
            contype=1 << self.GROUP_JOINT,
            conaffinity=(1 << self.GROUP_JOINT) | (1 << self.GROUP_TUBE) | (1 << self.GROUP_ENV),
            size=[self.cylinder_radius1, self.cylinder_length1 / 2], 
            pos=ensure_list(new_pos1), 
            rgba=[0,0,0,1], 
            name="lynxg1_"+self.name
        ) # 

        # <geom type="cylinder" size="0.075 0.01" pos="0 0.085 0" quat="0 0 1 1" material="black" class="visual"/>
        body.add(
            "geom", 
            type="cylinder",
            group=self.GROUP_JOINT,
            contype=1 << self.GROUP_JOINT,
            conaffinity=(1 << self.GROUP_JOINT) | (1 << self.GROUP_TUBE) | (1 << self.GROUP_ENV),
            size=[self.cylinder_radius2, self.cylinder_length2 / 2], 
            pos=ensure_list(new_pos2), 
            rgba=[0.15,0.15,0.15,1], 
            quat=mj_quat2, 
            name="lynxg2_"+self.name
        ) # 

        # Find the children
        tasks = []

        q = angle_to_quaternion(-np.pi/2, [1, 0, 0])
        new_attachment_point_quat = q * new_quat2
        attachment_site_pos = new_pos2 + new_quat3 * Vector3([0, 0, -self.cylinder_length2/2])
        for child_index, attachment_point in self.attachment_points.items():
            child = self.children.get(child_index)
            if child is not None:
                unbuilt = UnbuiltChild(module=child, mj_body=body)
                unbuilt.make_pose(
                    position=attachment_site_pos, # Prev pos + offset rotated in right direction = attachment point pos
                    orientation=new_attachment_point_quat,
                )
                tasks.append(unbuilt)

        return tasks
    

class JointOrthogonal_new(Module):

    ATTACH = 0
    GROUP_TUBE = 2
    GROUP_JOINT = 1
    GROUP_ENV = 0

    def __init__(
        self,
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
        *args, **kwargs,
    ):  
        
        self.cylinder_length1 = cylinder_length1
        self.cylinder_radius1 = cylinder_radius1
        self.cylinder_length2 = cylinder_length2
        self.cylinder_radius2 = cylinder_radius2
        self.length = cylinder_length1
        self.radius = cylinder_radius1 + cylinder_length2
        self.angle = angle  # Angle in radians
        # self.fixed_part_angle = fixed_part_angle
        self.name = name if name is not None else "joint"

        self.control_mode = control_mode
        self.armature = armature
        self.damping = damping
        self.frictionloss = frictionloss

        attachment_points = { 
            self.ATTACH: AttachmentPoint(
                offset=Vector3([0.0, self.length/2, 0.0]),
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
        rx      = angle_to_quaternion(np.pi/2, [1, 0, 0])
        twist   = angle_to_quaternion(self.angle, [0, 0, 1])
        new_relative_quat = twist * rx
        axis = new_relative_quat * Vector3([0, 0, 1])

        mj_quat1 = attachment_point_quat.to_mujoco_format()

        joint_pos_related_to_body = Vector3([0,0,self.cylinder_radius1])
        cylinder2_pos_related_to_body = joint_pos_related_to_body + new_relative_quat * Vector3([0, 0, self.length/2 + self.cylinder_length2/2])

        body = entry_point.add("body", name=self.name, pos=pos_list, quat=mj_quat1)

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
            mjcf.actuator.add("velocity", kv="10000", joint=self.name+"_joint")
        elif self.control_mode == "position":
            mjcf.actuator.add("position", kp="100000", joint=self.name+"_joint")
        else:
            raise ValueError(f"Unsupported control mode: {self.control_mode}")
        
        # mjcf.actuator.add("velocity", kv="10000", joint=self.name+"_joint")
        # <geom type="cylinder" size="0.075 0.0829" pos="0 0 0" material="jointgray" class="visual"/> 
        body.add(
            "geom", 
            type="cylinder", 
            group=self.GROUP_JOINT,
            contype=1 << self.GROUP_JOINT,
            conaffinity=(1 << self.GROUP_JOINT) | (1 << self.GROUP_TUBE) | (1 << self.GROUP_ENV),
            size=[self.cylinder_radius1, self.cylinder_length1 / 2], 
            pos=ensure_list(joint_pos_related_to_body),
            quat=new_relative_quat.to_mujoco_format(),
            rgba=[0,0,0,1], 
            name="lynxg1_"+self.name
        ) # 

        # <geom type="cylinder" size="0.075 0.01" pos="0 0.085 0" quat="0 0 1 1" material="black" class="visual"/>
        body.add(
            "geom", 
            type="cylinder",
            group=self.GROUP_JOINT,
            contype=1 << self.GROUP_JOINT,
            conaffinity=(1 << self.GROUP_JOINT) | (1 << self.GROUP_TUBE) | (1 << self.GROUP_ENV),
            size=[self.cylinder_radius2, self.cylinder_length2 / 2], 
            pos=ensure_list(cylinder2_pos_related_to_body), 
            rgba=[0.15,0.15,0.15,1], 
            # just no relative rotation:
            quat=new_relative_quat.to_mujoco_format(), 
            name="lynxg2_"+self.name
        ) # 

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
                )
                tasks.append(unbuilt)

        return tasks
