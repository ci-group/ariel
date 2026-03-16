import mujoco
from typing import Optional
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.mj_module import MjModule

class MjEndEffector(MjModule):
    """
    End effector module using MjSpec.
    """
    def __init__(
        self,
        name: str = "end_effector",
        pos=None,
        quat=None
    ):
        super().__init__(name=name, pos=pos, quat=quat)
        
        # Add a site for tracking (TCP)
        self.body.add_site(
            name="tcp",
            pos=[0, 0, 0],
            # Visual representation of the TCP
        )
        # Add attachment site (though it's the end)
        self.body.add_site(name="attach", pos=[0, 0, 0])
        self.body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.01, 0, 0],
            rgba=[1, 0, 0, 1],
            group=1
        )
