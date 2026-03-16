import mujoco
import numpy as np
from typing import Optional
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.mj_module import MjModule

class MjBase(MjModule):
    """
    Base module for the Lynx manipulator using MjSpec.
    """
    def __init__(
        self,
        base_length1: float = 0.5,
        base_radius1: float = 0.05,
        base_length2: float = 0.5,
        base_radius2: float = 0.05,
        name: str = "base",
        pos=None,
        quat=None
    ):
        super().__init__(name=name, pos=pos, quat=quat)
        
        self.base_length1 = base_length1
        self.base_radius1 = base_radius1
        self.base_length2 = base_length2
        self.base_radius2 = base_radius2
        self.total_length = base_length1 + base_length2

        # Add cameras
        self.body.add_camera(name="top_view", pos=[0, 0, 3.0], xyaxes=[1, 0, 0, 0, 1, 0], fovy=45)
        self.body.add_camera(name="side_view", pos=[2.5, 0, 0.5], xyaxes=[0, 1, 0, 0, 0, 1], fovy=45)
        self.body.add_camera(name="isometric_view", pos=[1.6, 1.6, 1.6], xyaxes=[-0.707, 0.707, 0, -0.408, -0.408, 0.816], fovy=45)

        # Add geoms
        # Group 1 is typically for joints/visuals in this project
        self.body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[base_radius1, base_length1 / 2, 0],
            pos=[0, 0, base_length1 / 2],
            rgba=[0, 0, 0, 1],
            group=1
        )
        self.body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[base_radius2, base_length2 / 2, 0],
            pos=[0, 0, base_length1 + base_length2 / 2],
            rgba=[0, 0, 0, 1],
            group=1
        )

        # Add mounting site at the top
        self.body.add_site(name="mount", pos=[0, 0, self.total_length])
