import numpy as np
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.module import Module, AttachmentPoint
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.unbuilt_child import UnbuiltChild
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.math_utils import Vector3, Quaternion, ensure_list, angle_to_quaternion

class RightAngleTube(Module):

    ATTACH = 0

    def __init__(
        self,
        cube_size : list = [0.1, 0.1, 0.1],
        name: str = "right_angle_tube",
    ):  
        
        self.cube_size = cube_size
        self.attachment_point_pos = Vector3([0.0, cube_size[1] , cube_size[2] ])  # Center of the cube
        self.name = name if name is not None else "RightAngleTube"
        attachment_points = { 
            self.ATTACH: AttachmentPoint(
                offset=self.attachment_point_pos,
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

        new_pos = Vector3([0,0,self.cube_size[2]])

        body = entry_point.add("body", name=self.name, pos=pos_list, quat=attachment_point_quat.to_mujoco_format())

        body.add("geom", type="box", size=self.cube_size, pos=ensure_list(new_pos), rgba=[1,0,1,1], name="lynx_"+self.name) # 

        # Find the children
        tasks = []

        # 90 degree rotation around the x-axis
        new_quat = Quaternion([-np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
        new_quat.normalise()
        body.add("site", type="sphere", size=[0.01], rgba=[0, 1, 0, 1], pos=ensure_list(self.attachment_point_pos)) 
        for child_index, attachment_point in self.attachment_points.items():
            child = self.children.get(child_index)
            if child is not None:
                unbuilt = UnbuiltChild(module=child, mj_body=body)
                unbuilt.make_pose(
                    position=attachment_point.offset, # Prev pos + offset rotated in right direction = attachment point pos
                    orientation=new_quat,
                )
                tasks.append(unbuilt)

        return tasks