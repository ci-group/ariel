import numpy as np
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.module import Module, AttachmentPoint
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.unbuilt_child import UnbuiltChild
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.math_utils import Vector3, Quaternion, ensure_list, angle_to_quaternion

class JointOrthogonal(Module):

    ATTACH = 0

    def __init__(
        self,
        cylinder_length1 : float = 0.5,
        cylinder_radius1 : float = 0.05,
        cylinder_length2 : float = 0.5,
        cylinder_radius2 : float = 0.05,
        angle : float = 0.0, # Angle in radians
        name: str = None,
    ):  
        
        self.cylinder_length1 = cylinder_length1
        self.cylinder_radius1 = cylinder_radius1
        self.cylinder_length2 = cylinder_length2
        self.cylinder_radius2 = cylinder_radius2
        self.length = cylinder_length1
        self.radius = cylinder_radius1 + cylinder_length2
        self.angle = angle  # Angle in radians
        self.name = name if name is not None else "joint"

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
        new_quat2 = Quaternion([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
        new_quat2.normalise()
        new_quat2 = q2 * new_quat2

        mj_quat2 = new_quat2.to_mujoco_format()

        new_pos1 = Vector3([0,self.cylinder_radius1,0])
        new_pos2 = new_pos1 + new_quat2 * Vector3([0, 0, -self.length/2- self.cylinder_length2/2])

        body = entry_point.add("body", name=self.name, pos=pos_list, quat=mj_quat1)

        # <joint name="shoulder_pan_joint" class="size4" axis="0 0 1"/>
        body.add("joint", name=self.name+"_joint", axis=[0, 0, 1], pos=ensure_list(new_pos1), type="hinge")
        mjcf.actuator.add("velocity", kv="10000", joint=self.name+"_joint")
        # <geom type="cylinder" size="0.075 0.0829" pos="0 0 0" material="jointgray" class="visual"/> 
        body.add("geom", type="cylinder", size=[self.cylinder_radius1, self.cylinder_length1 / 2], pos=ensure_list(new_pos1), rgba=[0,0,0,1], name="lynxg1_"+self.name) # 

        # <geom type="cylinder" size="0.075 0.01" pos="0 0.085 0" quat="0 0 1 1" material="black" class="visual"/>
        body.add("geom", type="cylinder", size=[self.cylinder_radius2, self.cylinder_length2 / 2], pos=ensure_list(new_pos2), quat=mj_quat2, name="lynxg2_"+self.name) # 

        # Find the children
        tasks = []

        attachment_site_pos = new_pos2 + new_quat2 * Vector3([0, 0, -self.cylinder_length2/2])
        for child_index, attachment_point in self.attachment_points.items():
            child = self.children.get(child_index)
            if child is not None:
                unbuilt = UnbuiltChild(module=child, mj_body=body)
                unbuilt.make_pose(
                    position=attachment_site_pos, # Prev pos + offset rotated in right direction = attachment point pos
                    orientation=attachment_point_quat,
                )
                tasks.append(unbuilt)

        return tasks