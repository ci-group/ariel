"""TODO(jmdm): description of script.

Notes
-----
    *

References
----------
    [1]

Todo
----
    [ ]

"""

# Standard library
from pathlib import Path

# Third-party libraries
import mujoco
import numpy as np
from pydantic_settings import BaseSettings

# Local libraries
# Global constants
# Global functions
# Warning Control
# Type Checking
# Type Aliases
# Type Aliases
type WeightType = float
type DimensionType = tuple[float, float, float]

# --- DATA SETUP --- #
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__"
DATA.mkdir(exist_ok=True)

# --- RANDOM GENERATOR SETUP --- #
SEED = 42
RNG = np.random.default_rng(SEED)


class MujocoConfig(BaseSettings):
    # --- Compiler --- #
    # https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjscompiler
    # https://mujoco.readthedocs.io/en/2.3.7/XMLreference.html#compiler
    autolimits: bool = True
    balanceinertia: bool = True
    degree: bool = False

    # --- Option --- #
    # https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjoption
    # https://mujoco.readthedocs.io/en/2.3.7/XMLreference.html#option
    timestep: float = 0.02  # seconds
    integrator: int = int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)

    # --- Visual --- #
    offheight: int = 960
    offwidth: int = 1280

    # --- Default Geom --- #
    floor_name: str = "floor"
