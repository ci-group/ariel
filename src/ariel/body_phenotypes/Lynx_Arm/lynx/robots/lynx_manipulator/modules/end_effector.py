import numpy as np
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.module import Module, AttachmentPoint
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.unbuilt_child import UnbuiltChild
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.math_utils import Vector3, Quaternion, ensure_list, angle_to_quaternion

class EndEffector(Module):

    ATTACH = 0

    def __init__(
        self,
        name: str = "end_effector",
    ):  
        self.name = name

        attachment_points = { }

        super().__init__(attachment_points)

    def build(self, mjcf, entry_point, attachment_point_pos, attachment_point_quat) -> list:

        entry_point.add("site", type="sphere", size=[0.01], rgba=[1, 0, 0, 1], pos=ensure_list(attachment_point_pos), name=self.name) 

        return []