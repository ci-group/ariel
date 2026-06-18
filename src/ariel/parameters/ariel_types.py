"""ARIEL Types."""

# Standard library
from typing import TypeAlias

# Third-party libraries
import numpy as np
import numpy.typing as npt

# Local libraries
# Global constants
ND_FLOAT_PRECISION = np.float64
ND_INT_PRECISION = np.int32

# Global functions
# Warning Control
# Type Checking
# Type Aliases

# --- NUMERICAL TYPES --- #
Dimension: TypeAlias = tuple[float, float, float]  # length, width, height
Position: TypeAlias = tuple[float, float, float]  # x-pos, y-pos, z-pos
Rotation: TypeAlias = tuple[float, float, float]  # x-axis, y-axis, z-axis

# --- NUMPY DERIVED TYPES --- #
FloatArray: TypeAlias = npt.NDArray[ND_FLOAT_PRECISION]
IntArray: TypeAlias = npt.NDArray[ND_INT_PRECISION]
