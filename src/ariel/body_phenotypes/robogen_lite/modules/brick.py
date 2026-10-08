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
from ariel.parameters.ariel_modules import ArielModulesConfig

# Global functions
ariel_modules_config = ArielModulesConfig()


def mass_from_length(length: float) -> float:
    """Calculate the mass of the brick module.

    Parameters
    ----------
    length
        The total length of the brick module in meters.

    Returns
    -------
    float
        The mass of the brick module in kilograms.
    """
    return (42.65 + (length - 0.075) * 0.44531428571) / 1000


class BrickModule(Module):
    """Brick module specifications."""

    index: int | None = None
    module_type: ModuleType = ModuleType.BRICK

    def __init__(self, index: int, length: float | None = None) -> None:
        """Initialize the brick module.

        Parameters
        ----------
        index
            The index of the brick module being instantiated
        length
            The total length of the brick module in meters.
        """
        # Set the index of the module
        self.index = index

        # Set the length of the module
        self.length = (
            length
            if length is not None
            else ariel_modules_config.BRICK_LENGTH_DEFAULT
        )

        # Check that the length is within the allowed range
        if not (
            ariel_modules_config.BRICK_LENGTH_MIN
            <= self.length
            <= ariel_modules_config.BRICK_LENGTH_MAX
        ):
            msg = (
                "Brick length must be between "
                f"{ariel_modules_config.BRICK_LENGTH_MIN} and "
                f"{ariel_modules_config.BRICK_LENGTH_MAX} meters."
            )
            raise ValueError(msg)

        # Set the fixed dimensions of the module
        width = ariel_modules_config.BRICK_WIDTH
        height = ariel_modules_config.BRICK_HEIGHT

        # MuJoCo box sizes are half-extents
        half_length = self.length / 2
        half_width = width / 2
        half_height = height / 2

        # Create the parent spec.
        spec = mujoco.MjSpec()

        # ========= BRICK =========
        brick_name = self.module_type.name.lower()
        brick = spec.worldbody.add_body(
            name=brick_name,
        )
        brick.add_geom(
            name=brick_name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            mass=mass_from_length(self.length),
            size=[half_width, half_length, half_height],
            pos=[0, half_length, 0],
            rgba=(28 / 255, 119 / 255, 195 / 255, 1),
        )

        # ========= Attachment Points =========
        self.sites = {}
        shift = -1  # mujoco uses xyzw instead of wxyz
        self.sites[ModuleFaces.FRONT] = brick.add_site(
            name=f"{brick_name}-front",
            pos=[0, self.length, 0],
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
            pos=[
                -half_width,
                half_length,
                0,
            ],
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
            pos=[
                half_width,
                half_length,
                0,
            ],
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
            pos=[
                0,
                half_length,
                half_height,
            ],
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
            pos=[
                0,
                half_length,
                -half_height,
            ],
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
