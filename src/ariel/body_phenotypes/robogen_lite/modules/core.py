"""TODO(jmdm): description of script."""

# Third-party libraries
import mujoco
import numpy as np
import quaternion as qnp

# Local libraries
from ariel.body_phenotypes.robogen_lite.config import (
    IDX_OF_CORE,
    ModuleFaces,
    ModuleType,
)
from ariel.body_phenotypes.robogen_lite.modules.module import Module

# Type Aliases
type WeightType = float
type DimensionType = tuple[float, float, float]

# --- Robogen Configuration ---
# Module weights (kg)
CORE_MASS: WeightType = 1

# Module dimensions (length, width, height) in meters
CORE_DIMENSIONS: DimensionType = (0.10, 0.10, 0.10)
# ------------------------------


class CoreModule(Module):
    """Core module specifications."""

    index: int | None = None
    module_type: str = ModuleType.CORE

    def __init__(self, index: int) -> None:
        """
        Initialize the core module.

        Parameters
        ----------
        index : int
            The index of the core module.

        Raises
        ------
        ValueError
            If the index is not the core module index.
        """
        # Check that the index is the core module index
        if index != IDX_OF_CORE:
            msg = f"Core module index must be {IDX_OF_CORE}, but got {index}."
            raise ValueError(msg)

        # Set the index
        self.index = IDX_OF_CORE

        # Create the parent spec.
        spec = mujoco.MjSpec()

        # ========= Core =========
        core_name = "core"
        core = spec.worldbody.add_body(
            name=core_name,
        )
        core.add_geom(
            name=core_name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            mass=CORE_MASS,
            size=CORE_DIMENSIONS,
            pos=[0, CORE_DIMENSIONS[0], 0],
            rgba=(253 / 255, 202 / 255, 64 / 255, 1),
        )

        # ========= Attachment Points =========
        self.sites = {}
        shift = -1  # mujoco uses xyzw instead of wxyz
        self.sites[ModuleFaces.FRONT] = core.add_site(
            name=f"{core_name}-front",
            pos=[0, CORE_DIMENSIONS[1] * 2, -CORE_DIMENSIONS[1] / 2],
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
        self.sites[ModuleFaces.BACK] = core.add_site(
            name=f"{core_name}-back",
            pos=[0, 0, -CORE_DIMENSIONS[1] / 2],
            quat=np.round(
                np.roll(
                    qnp.as_float_array(
                        qnp.from_euler_angles([
                            np.deg2rad(0),
                            np.deg2rad(0),
                            np.deg2rad(0),
                        ]),
                    ),
                    shift=shift,
                ),
                decimals=3,
            ),
        )
        self.sites[ModuleFaces.LEFT] = core.add_site(
            name=f"{core_name}-left",
            pos=[
                -CORE_DIMENSIONS[0],
                CORE_DIMENSIONS[1],
                -CORE_DIMENSIONS[1] / 2,
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
        self.sites[ModuleFaces.RIGHT] = core.add_site(
            name=f"{core_name}-right",
            pos=[
                CORE_DIMENSIONS[0],
                CORE_DIMENSIONS[1],
                -CORE_DIMENSIONS[1] / 2,
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
        self.sites[ModuleFaces.TOP] = core.add_site(
            name=f"{core_name}-top",
            pos=[0, CORE_DIMENSIONS[1], CORE_DIMENSIONS[2]],
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
        self.sites[ModuleFaces.BOTTOM] = core.add_site(
            name=f"{core_name}-bottom",
            pos=[0, CORE_DIMENSIONS[1], -CORE_DIMENSIONS[2]],
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

    def rotate(self, angle: float) -> None:
        """
        Rotate the core module by a specified angle.

        Parameters
        ----------
        angle : float
            The angle in radians to rotate the core.

        Raises
        ------
        AttributeError
            Core module does not support rotation.
        """
        if angle != 0:
            msg = f"Attempted to rotate the core module by: {angle}."
            msg += f"Core ({self.index}) module does not support rotation."
            raise AttributeError(msg)
