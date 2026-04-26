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


class SER0019(BaseSettings):
    # --- Servo Config --- #
    # https://github.com/ci-group/ariel-models/blob/master/v2/servo_specs/SER0019_sevo.pdf
    # Assume 6 V operation
    MIN_ANGLE: float = -np.pi  # radians
    MAX_ANGLE: float = np.pi  # radians
    MAX_SPEED: float = 0.18  # seconds per 60 degrees of rotation
    STALL_TORQUE: float = 13.5  # kg*cm
    MAX_TORQUE_POWER: float = 15  # kg
    DIMENSIONS: DimensionType = (54.5, 20, 47.5)  # mm
    MIN_FREQUENCY: float = 50  # Hz
    MAX_FREQUENCY: float = 330  # Hz


class ArielModulesConfig(BaseSettings):
    # --- Brick Config --- #
    # Module weights (kg)
    BRICK_MASS: WeightType = 0.055  # 55 grams

    # Module dimensions (length, width, height) in meters
    BRICK_DIMENSIONS: DimensionType = (0.05, 0.05, 0.05)
    # ------------------------------ #

    # --- Hinge Config --- #
    ACTUATOR: SER0019 = SER0019()
    # ------------------------------ #
