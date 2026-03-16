import numpy as np
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.module import Module, AttachmentPoint
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.unbuilt_child import UnbuiltChild
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.math_utils import Vector3, Quaternion, ensure_list

class Base(Module):

    ATTACH = 0
    GROUP_TUBE = 2
    GROUP_JOINT = 1
    GROUP_ENV = 0

    def __init__(
        self,
        base_length1 : float = 0.5,
        base_radius1 : float = 0.05,
        base_length2 : float = 0.5,
        base_radius2 : float = 0.05,
        name: str = None,
    ):  
        
        self.base_length1 = base_length1
        self.base_radius1 = base_radius1
        self.base_length2 = base_length2
        self.base_radius2 = base_radius2
        self.length = base_length1 + base_length2
        self.radius = max(base_radius1, base_radius2)
        self.name = name if name is not None else "Base"

        attachment_points = { 
            self.ATTACH: AttachmentPoint(
                offset=Vector3([0.0, 0.0, self.length]),
                orientation=Quaternion(),
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
        
        attachment_point_quat = Quaternion(attachment_point_quat)
        attachment_point_quat.normalise()
        if np.all(np.array(attachment_point_quat.xyzw) == 0):
            print("Warning: Attachment point quaternion is zero, using default orientation.")
            attachment_point_quat = Quaternion()
        print(f"base attachment quat: {attachment_point_quat}")
        quat = attachment_point_quat.to_mujoco_format()
        
        # Build the module
        root_body = entry_point.add("body", name=self.name, pos=pos_list, quat=quat)

        # <camera name="top_view" pos="{base_x} {base_y} {base_z + 2.0}" xyaxes="1 0 0 0 1 0" fovy="45"/>
        root_body.add("camera", name="top_view", pos=[0, 0, 3.0], xyaxes=[1, 0, 0, 0, 1, 0], fovy=45)
        # <camera name="side_view" pos="{base_x + 1.5} {base_y} {base_z + 0.5}" xyaxes="0 0 -1 0 1 0" fovy="45"/>
        root_body.add("camera", name="side_view", pos=[2.5, 0, 0.5], xyaxes=[0, 1, 0, 0, 0, 1], fovy=45)
        # <camera name="isometric_view" pos="{base_x + 1.2} {base_y + 1.2} {base_z + 1.2}" xyaxes="-0.707 0.707 0 -0.408 -0.408 0.816" fovy="45"/>
        root_body.add("camera", name="isometric_view", pos=[1.6, 1.6, 1.6], xyaxes=[-0.707, 0.707, 0, -0.408, -0.408, 0.816], fovy=45)
        # added camera:
        root_body.add("camera", name="specialized_view", pos=[0.5, 0, 0.5], xyaxes=[0, 1, 0, -0.707, 0, 0.707], fovy=45)
        
        # <geom type="cylinder" size="0.095 0.0425" pos="0 0 0.0425"material="jointgray" class="visual"/>
        root_body.add(
            "geom", 
            type="cylinder", 
            group=self.GROUP_JOINT,
            contype=1 << self.GROUP_JOINT,
            conaffinity=(1 << self.GROUP_JOINT) | (1 << self.GROUP_TUBE) | (1 << self.GROUP_ENV),
            size=[self.base_radius1, self.base_length1 / 2], 
            rgba=[0,0,0,1], 
            pos=[0, 0, self.base_length1 / 2]
        )
        # <geom type="cylinder" size="0.075 0.0072" pos="0 0 0.0922"material="black" class="visual"/>
        root_body.add(
            "geom", 
            type="cylinder", 
            group=self.GROUP_JOINT,
            contype=1 << self.GROUP_JOINT,
            conaffinity=(1 << self.GROUP_JOINT) | (1 << self.GROUP_TUBE) | (1 << self.GROUP_ENV),
            size=[self.base_radius2, self.base_length2 / 2], 
            rgba=[0,0,0,1], 
            pos=[0, 0, self.base_length1 + self.base_length2 / 2]
        )
        
        # Find the children
        tasks = []
        for child_index, attachment_point in self.attachment_points.items():
            child = self.children.get(child_index)
            if child is not None:
                unbuilt = UnbuiltChild(module=child, mj_body=root_body)
                unbuilt.make_pose(
                    position=attachment_point.offset, # Prev pos + offset rotated in right direction = attachment point pos
                    orientation=Quaternion(), # Prev quat * offset quat = attachment point quat
                )
                tasks.append(unbuilt)
        return tasks