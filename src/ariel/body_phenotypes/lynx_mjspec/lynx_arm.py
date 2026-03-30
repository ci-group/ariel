import math
import mujoco
import numpy as np
import mujoco.viewer
from ariel.body_phenotypes.lynx_mjspec.table import TableWorld

def euler_to_quat(roll, pitch, yaw):
    """Simple ZYX Euler to Quaternion helper."""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy
    ]

class LynxArm:
    """
    A unified MjSpec builder for the Lynxmotion Arm.
    Builds the entire robot in a single kinematic tree.
    """
    def __init__(self, config=None):
        self.spec = mujoco.MjSpec()
        self.spec.compiler.degree = False
        
        # Default configuration matching your sim.yaml
        self.config = config or {
            "num_joints": 6,
            "genotype_tube": [1, 1, 1, 1, 1],
            "genotype_joints": 6,
            "tube_lengths": [0.1, 0.1, 0.1, 0.1, 0.1],
            "rotation_angles": [0.0, -1.57, 0.0, 0.0, 0.0, 0.0],
            "task": "reach"
        }
        
        self.base_height = 0.1
        self.joint_radius = 0.04
        self.tube_radius = 0.03
        
        self._build_robot()

    def _build_robot(self):
        """Constructs the kinematic chain hierarchically."""
        # 1. ROOT / BASE
        base_link = self.spec.worldbody.add_body(name="lynx_base", pos=[0, 0, 0])
        base_link.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.08, self.base_height / 2, 0],
            pos=[0, 0, self.base_height / 2],
            rgba=[0.2, 0.2, 0.2, 1],
            mass=2.0,
            group=1
        )
        
        # Keep track of the current parent body in the chain
        current_parent = base_link
        current_z_offset = self.base_height

        # 2. ITERATIVE CHAIN (Joints and Tubes)
        num_joints = min(self.config["num_joints"], self.config["genotype_joints"])
        
        for i in range(num_joints):
            # Parse rotations from config
            yaw = self.config["rotation_angles"][i] if i < len(self.config["rotation_angles"]) else 0.0
            quat = euler_to_quat(0, 0, yaw)
            
            # Create a new Link Body attached to the previous link
            link_name = f"link_{i+1}"
            link_body = current_parent.add_body(
                name=link_name, 
                pos=[0, 0, current_z_offset], 
                quat=quat
            )
            
            # Alternate joint rotation axes: Y, Z, Y, Z, Y, Z
            # This allows proper 3D reach instead of being constrained to a plane
            joint_axis = [0, 1, 0] if i % 2 == 0 else [0, 0, 1]
            
            # Add Joint
            joint_name = f"joint_{i+1}"
            link_body.add_joint(
                name=joint_name,
                type=mujoco.mjtJoint.mjJNT_HINGE,
                axis=joint_axis,
                range=[-2.8, 2.8],
                limited=True,
                damping=0.1,
                frictionloss=0.01
            )
            
            # Add Actuator for this joint directly to the spec
            self.spec.add_actuator(
                name=f"motor_{i+1}",
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
                target=joint_name,
                ctrlrange=[-2.8, 2.8],
                ctrllimited=True,
                gainprm=[50, 0, 0, 0, 0, 0, 0, 0, 0, 0], # P gain
                biasprm=[0, -50, 0, 0, 0, 0, 0, 0, 0, 0] # D gain
            )
            
            # Add Joint Visuals (The motor housing)
            link_body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[self.joint_radius, 0.03, 0],
                axisangle=[1, 0, 0, 1.5708], # Rotate 90 deg to lay flat
                rgba=[0.1, 0.1, 0.1, 1],
                group=1,
                mass=0.2
            )

            # Check if there is a tube after this joint
            tube_length = 0.0
            if i < len(self.config["genotype_tube"]) and self.config["genotype_tube"][i] == 1:
                tube_length = self.config["tube_lengths"][i]
                
                # Add Tube Geom to the SAME link body
                link_body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=[self.tube_radius, tube_length / 2, 0],
                    pos=[0, 0, tube_length / 2],
                    rgba=[0.7, 0.2, 0.2, 1],
                    group=1,
                    mass=0.1
                )
            
            # Update parent and offset for the next iteration
            current_parent = link_body
            current_z_offset = tube_length  # Next joint starts at the end of this tube

        # 3. END EFFECTOR
        ee_name = "end_effector"
        ee_body = current_parent.add_body(name=ee_name, pos=[0, 0, current_z_offset])
        
        # Add visual end effector pointer
        ee_body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.01, 0.04, 0],
            pos=[0, 0, 0.04],
            rgba=[0.8, 0.8, 0.8, 1],
            group=1,
            mass=0.05
        )
        
        # Add the all-important TCP site for your RL rewards/IK tracking
        ee_body.add_site(
            name="tcp",
            pos=[0, 0, 0.08],
            rgba=[1, 0, 0, 1],
            size=[0.01, 0.01, 0.01]
        )

# Example Usage:
if __name__ == "__main__":   
    # 1. Create the unified arm
    arm = LynxArm()
    
    # 2. Spawn it into your TableWorld
    world = TableWorld()
    world.spawn(arm.spec) # You can just attach the root spec directly now!
    
    # 3. Compile and View
    model = world.spec.compile()
    data = mujoco.MjData(model)
    
    mujoco.viewer.launch(model, data)