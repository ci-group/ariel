from dataclasses import dataclass, field
from dm_control import mjcf
from .math_utils import Vector3, Quaternion

from .module import Module

@dataclass
class UnbuiltChild:
    """A dataclass to store unbuilt children for the builders."""

    module: Module 
    mj_body : mjcf.Element

    def make_pose(
        self, position: Vector3, orientation: Quaternion = Quaternion()
    ) -> None:
        """
        Make the pose of the unbuilt child.

        :param position: The position argument from the parent.
        :param orientation: The orientation of the attachment on the parent.
        """
        self.position = position
        self.orientation = orientation