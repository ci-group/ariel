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

# Type Aliases
type WeightType = float
type DimensionType = tuple[float, float, float]

# --- Robogen Configuration ---
# Module weights (kg)
BRICK_MASS: WeightType = 0.055  # 55 grams

# Module dimensions (length, width, height) in meters
BRICK_DIMENSIONS: DimensionType = (0.05, 0.05, 0.05)
# ------------------------------


class BrickModule(Module):
    """Brick module specifications."""

    index: int | None = None
    module_type: ModuleType = ModuleType.BRICK

    def __init__(self, index: int) -> None:
        """Initialize the brick module.

        Parameters
        ----------
        index : int
            The index of the brick module being instantiated
        """
        # Set the index of the module
        self.index = index

        # Create the parent spec.
        spec = mujoco.MjSpec()

        # ========= Core =========
        brick_name = "core"
        brick = spec.worldbody.add_body(
            name=brick_name,
        )
        brick.add_geom(
            name=brick_name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            mass=BRICK_MASS,
            size=BRICK_DIMENSIONS,
            pos=[0, BRICK_DIMENSIONS[0], 0],
            rgba=(28 / 255, 119 / 255, 195 / 255, 1),
        )

        # ========= Attachment Points =========
        self.sites = {}
        shift = -1  # mujoco uses xyzw instead of wxyz
        self.sites[ModuleFaces.FRONT] = brick.add_site(
            name=f"{brick_name}-front",
            pos=[0, BRICK_DIMENSIONS[1] * 2, 0],
            quat=np.round(
                np.roll(
                    qnp.as_float_array(
                        qnp.from_euler_angles([
                            np.deg2rad(0),
                            np.deg2rad(180),
                            np.deg2rad(180),
                        ]),
                    ),
                    shift=shift,
                ),
                decimals=3,
            ),
        )
        self.sites[ModuleFaces.LEFT] = brick.add_site(
            name=f"{brick_name}-left",
            pos=[-BRICK_DIMENSIONS[0], BRICK_DIMENSIONS[1], 0],
            quat=np.round(
                np.roll(
                    qnp.as_float_array(
                        qnp.from_euler_angles([
                            np.deg2rad(90),
                            -np.deg2rad(90),
                            -np.deg2rad(90),
                        ]),
                    ),
                    shift=shift,
                ),
                decimals=3,
            ),
        )
        self.sites[ModuleFaces.RIGHT] = brick.add_site(
            name=f"{brick_name}-right",
            pos=[BRICK_DIMENSIONS[0], BRICK_DIMENSIONS[1], 0],
            quat=np.round(
                np.roll(
                    qnp.as_float_array(
                        qnp.from_euler_angles([
                            np.deg2rad(90),
                            np.deg2rad(90),
                            -np.deg2rad(90),
                        ]),
                    ),
                    shift=shift,
                ),
                decimals=3,
            ),
        )
        self.sites[ModuleFaces.TOP] = brick.add_site(
            name=f"{brick_name}-top",
            pos=[0, BRICK_DIMENSIONS[1], BRICK_DIMENSIONS[2]],
            quat=np.round(
                np.roll(
                    qnp.as_float_array(
                        qnp.from_euler_angles([
                            np.deg2rad(0),
                            np.deg2rad(180),
                            np.deg2rad(90),
                        ]),
                    ),
                    shift=shift,
                ),
                decimals=3,
            ),
        )
        self.sites[ModuleFaces.BOTTOM] = brick.add_site(
            name=f"{brick_name}-bottom",
            pos=[0, BRICK_DIMENSIONS[1], -BRICK_DIMENSIONS[2]],
            quat=np.round(
                np.roll(
                    qnp.as_float_array(
                        qnp.from_euler_angles([
                            np.deg2rad(0),
                            np.deg2rad(0),
                            -np.deg2rad(90),
                        ]),
                    ),
                    shift=shift,
                ),
                decimals=3,
            ),
        )

        # Save model specifications
        self.spec = spec
        self.body = brick
        self.rotate(angle=0)  # Initialize with no rotation

    def rotate(
        self,
        angle: float,
    ) -> None:
        """
        Rotate the brick module by a specified angle.

        Parameters
        ----------
        angle : float
            The angle in degrees to rotate the brick.
        """
        # Convert angle to quaternion
        quat = qnp.from_euler_angles([
            np.deg2rad(180),
            -np.deg2rad(180 - angle),
            np.deg2rad(0),
        ])
        quat = np.roll(qnp.as_float_array(quat), shift=-1)

        # Set the quaternion for the brick body
        self.body.quat = np.round(quat, decimals=3)
