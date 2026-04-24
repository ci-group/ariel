import math
import mujoco
import numpy as np
import mujoco.viewer
from ariel.body_phenotypes.lynx_mjspec.table import TableWorld


class LynxArm:
    """
    A unified MjSpec builder for the Lynxmotion Arm.
    Builds the entire robot in a single kinematic tree.

    """

    def __init__(self, config=None):
        self.spec = mujoco.MjSpec()
        self.spec.compiler.degree = False

        # Default configuration
        self.config = config or {
            "num_joints": 6,
            "genotype_tube": [1, 1, 1, 1, 1],
            "genotype_joints": 6,
            "tube_lengths": [0.1, 0.1, 0.1, 0.1, 0.1],
            "rotation_angles": [0.0, -1.57, 0.0, 0.0, 0.0, 0.0],
            "task": "reach",
        }

        self.base_height = 0.1
        self.joint_radius = 0.04
        self.tube_radius = 0.03

        self._build_robot()

    def _rotate_axis_z(self, axis: np.ndarray, yaw: float) -> np.ndarray:
        """Rotate a vector around Z axis."""
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1],
        ])
        return R @ axis

    def _build_robot(self):
        """Constructs the kinematic chain hierarchically."""

        # 1. ROOT / BASE
        base_link = self.spec.worldbody.add_body(
            name="lynx_base", pos=[0, 0, 0]
        )

        base_link.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.08, self.base_height / 2, 0],
            pos=[0, 0, self.base_height / 2],
            rgba=[0.2, 0.2, 0.2, 1],
            mass=2.0,
            group=1,
        )

        current_parent = base_link
        current_z_offset = self.base_height

        num_joints = min(
            self.config["num_joints"], self.config["genotype_joints"]
        )

        # 2. ITERATIVE CHAIN
        for i in range(num_joints):
            link_name = f"link_{i+1}"
            link_body = current_parent.add_body(
                name=link_name,
                pos=[0, 0, current_z_offset],
            )

            # Base alternating axes
            base_axis = (
                np.array([0, 1, 0]) if i % 2 == 0 else np.array([0, 0, 1])
            )

            # Apply yaw rotation to axis instead of body
            yaw = (
                self.config["rotation_angles"][i]
                if i < len(self.config["rotation_angles"])
                else 0.0
            )

            joint_axis = self._rotate_axis_z(base_axis, yaw).tolist()

            # Add Joint
            joint_name = f"joint_{i+1}"
            link_body.add_joint(
                name=joint_name,
                type=mujoco.mjtJoint.mjJNT_HINGE,
                axis=joint_axis,
                range=[-2.8, 2.8],
                limited=True,
                damping=2.0,
                frictionloss=0.01,
            )

            # Add Actuator (PD position control)
            self.spec.add_actuator(
                name=f"motor_{i+1}",
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
                target=joint_name,
                ctrlrange=[-2.8, 2.8],
                ctrllimited=True,
                dynprm=[1, 0, 0] + [0]*7,
                gainprm=[20, 0, 0] + [0]*7,
                biasprm=[0, -20, 0] + [0]*7,
            )

            # Joint visual (motor housing)
            link_body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[self.joint_radius, 0.03, 0],
                axisangle=[1, 0, 0, 1.5708],
                rgba=[0.1, 0.1, 0.1, 1],
                group=1,
                mass=0.2,
            )

            # Tube (if enabled)
            tube_length = 0.0
            if (
                i < len(self.config["genotype_tube"])
                and self.config["genotype_tube"][i] == 1
            ):
                tube_length = self.config["tube_lengths"][i]

                link_body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=[self.tube_radius, tube_length / 2, 0],
                    pos=[0, 0, tube_length / 2],
                    rgba=[0.7, 0.2, 0.2, 1],
                    group=1,
                    mass=0.1,
                )

            # Update chain
            current_parent = link_body
            current_z_offset = tube_length

        # 3. END EFFECTOR
        ee_body = current_parent.add_body(
            name="end_effector",
            pos=[0, 0, current_z_offset],
        )

        ee_body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.01, 0.04, 0],
            pos=[0, 0, 0.04],
            rgba=[0.8, 0.8, 0.8, 1],
            group=1,
            mass=0.05,
        )

        ee_body.add_site(
            name="tcp",
            pos=[0, 0, 0.08],
            rgba=[1, 0, 0, 1],
            size=[0.01, 0.01, 0.01],
        )


# Example Usage
if __name__ == "__main__":
    arm = LynxArm()

    world = TableWorld()
    world.spawn(arm.spec)

    model = world.spec.compile()
    data = mujoco.MjData(model)

    mujoco.viewer.launch(model, data)
