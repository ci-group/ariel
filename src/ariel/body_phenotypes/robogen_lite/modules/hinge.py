"""TODO(jmdm): description of script.

Todo:
----
    [ ] ".rotate" as superclass method?
"""

# Third-party libraries
import mujoco
import numpy as np
import quaternion as qnp

# Local libraries
from ariel.body_phenotypes.robogen_lite.config import ModuleFaces, ModuleType
from ariel.body_phenotypes.robogen_lite.modules.module import Module

# Global constants
SHRINK = 0.99

# Type Aliases
type WeightType = float
type DimensionType = tuple[float, float, float]

# --- Robogen Configuration --- #
# Module weights (kg)
STATOR_MASS: WeightType = 0.02  # 20 grams
ROTOR_MASS: WeightType = 0.04  # 40 grams

# Module dimensions (length, width, height) in meters
STATOR_DIMENSIONS: DimensionType = (0.025, 0.03, 0.025)
ROTOR_DIMENSIONS: DimensionType = (0.025, 0.02, 0.025)
# ------------------------------ #


class HingeModule(Module):
    """Hinge module specifications."""

    index: int | None = None
    module_type: ModuleType = ModuleType.HINGE

    def __init__(self, index: int) -> None:
        """Initialize the hinge module.

        Parameters
        ----------
        index
            The index of the hinge module being instantiated
        """
        # Set the index of the module
        self.index = index

        # Create the parent spec.
        spec = mujoco.MjSpec()

        # ========= Hinge =========
        hinge_name = self.module_type.name.lower()
        hinge = spec.worldbody.add_body(
            name=hinge_name,
            mass=STATOR_MASS + ROTOR_MASS,
        )

        # ========= Stator =========
        stator_name = "stator"
        stator = hinge.add_body(
            name=stator_name,
            pos=[0, STATOR_DIMENSIONS[1], 0],
        )
        stator.add_geom(
            name=stator_name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            mass=STATOR_MASS,
            size=np.array(STATOR_DIMENSIONS) * SHRINK,  # z-fighting
            rgba=(223 / 255, 41 / 255, 53 / 255, 1),
        )

        # ========= Rotor =========
        rotor_name = "rotor"
        rotor = hinge.add_body(
            name=rotor_name,
            pos=[0, STATOR_DIMENSIONS[1] * 2 + ROTOR_DIMENSIONS[1], 0],
        )
        rotor.add_geom(
            name=rotor_name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            mass=ROTOR_MASS,
            size=ROTOR_DIMENSIONS,
            rgba=(160 / 255, 24 / 255, 33 / 255, 1),
        )

        # ======== Attachment Points =========
        self.sites = {}
        self.sites[ModuleFaces.FRONT] = rotor.add_site(
            name=f"{hinge_name}-front",
            pos=[0, ROTOR_DIMENSIONS[1], 0],
        )

        # ========= Servo =========
        # Robot actuators — DSS-M15 digital servo (180° range, 1.47 N·m stall,
        # ~0.14 s/60° no-load). PD gains chosen for critical damping at the
        # typical ariel-leg load inertia (~1.5e-4 kg·m² ≈ one brick on a
        # 0.05 m lever); dynprm[0] = 0.05 approximates the servo's internal
        # position-tracking delay; forcerange clips at the servo stall torque.
        kp = 1
        kv = 0.025
        servo_tau = 0.05
        servo_torque_limit = 1.47
        servo_axis = (0, 0, 1)

        servo_name = "servo"
        rotor.add_joint(
            name=servo_name,
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=servo_axis,
            pos=[0, -ROTOR_DIMENSIONS[1], 0],
        )

        # Actuator parameters are defined over a range of 10...
        dynprm = np.zeros(10)
        gainprm = np.zeros(10)
        biasprm = np.zeros(10)

        # ... but only a few of the parameters are actually used
        gainprm[0] = kp
        biasprm[:3] = [0, -kp, -kv]
        dynprm[0] = servo_tau

        # Contact exclusion
        spec.add_exclude(
            bodyname1=stator_name,
            bodyname2=rotor_name,
        )

        # --- Actuator(s) --- #
        # FILTEREXACT: ctrl is low-passed through a first-order filter of time
        # constant dynprm[0] before reaching the PD gain, simulating the servo's
        # internal position-tracking lag.
        dyntype = mujoco.mjtDyn.mjDYN_FILTEREXACT
        gaintype = mujoco.mjtGain.mjGAIN_FIXED
        biastype = mujoco.mjtBias.mjBIAS_AFFINE
        trntype = mujoco.mjtTrn.mjTRN_JOINT
        spec.add_actuator(
            name=servo_name,
            dyntype=dyntype,
            gaintype=gaintype,
            biastype=biastype,
            dynprm=dynprm,
            gainprm=gainprm,
            biasprm=biasprm,
            trntype=trntype,
            target=servo_name,
            ctrlrange=(
                -np.pi / 2,
                np.pi / 2,
            ),  # [-90, 90] degrees (range of 180)
            forcerange=(-servo_torque_limit, servo_torque_limit),
            forcelimited=True,
        )

        # Save model specifications
        self.spec = spec
        self.body = hinge
        self.rotate(angle=0)  # Initialize with no rotation

    def rotate(
        self,
        angle: float,
    ) -> None:
        """
        Rotate the hinge module by a specified angle.

        Parameters
        ----------
        angle
            The angle in degrees to rotate the hinge.
        """
        # Convert angle to quaternion
        quat = qnp.from_euler_angles([
            np.deg2rad(180),
            -np.deg2rad(180 - angle),
            np.deg2rad(0),
        ])
        quat = np.roll(qnp.as_float_array(quat), shift=-1)

        # Set the quaternion for the hinge body
        self.body.quat = np.round(quat, decimals=3)
