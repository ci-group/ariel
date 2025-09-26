"""TODO(jmdm): description of script."""

# Standard library
from pathlib import Path

# Third-party libraries
import mujoco
import numpy as np
import numpy.typing as npt
from rich.console import Console

# Global constants
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = Path(CWD / "__data__")
DATA.mkdir(exist_ok=True)

# Global functions
console = Console()


def compute_geom_bounding_box(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Compute the axis-aligned bounding box (AABB) for all geoms in a MuJoCo model.

    Parameters
    ----------
    model : mujoco.MjModel
        The MuJoCo model to compute the bounding box for.
    data : mujoco.MjData
        The MuJoCo data to compute the bounding box for.

    Returns
    -------
    tuple[npt.NDArray, npt.NDArray]
        The minimum and maximum corners of the bounding box.
    """
    min_geom_size = data.geom_xpos - model.geom_size
    max_geom_size = data.geom_xpos + model.geom_size

    min_corner_id = np.argmin(min_geom_size, axis=0)
    max_corner_id = np.argmax(max_geom_size, axis=0)

    min_geom_rbound = data.geom_xpos - model.geom_rbound.reshape(-1, 1)
    max_geom_rbound = data.geom_xpos + model.geom_rbound.reshape(-1, 1)

    min_corner = np.diagonal(min_geom_rbound[min_corner_id])
    max_corner = np.diagonal(max_geom_rbound[max_corner_id])

    return min_corner, max_corner
