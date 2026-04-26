"""ARIEL Types."""

# Standard library
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
type Dimension = tuple[float, float, float]  # length, width, height
type Position = tuple[float, float, float]  # x-pos, y-pos, z-pos
type Rotation = tuple[float, float, float]  # x-axis, y-axis, z-axis

# --- NUMPY DERIVED TYPES --- #
type FloatArray = npt.NDArray[ND_FLOAT_PRECISION]
type IntArray = npt.NDArray[ND_INT_PRECISION]
