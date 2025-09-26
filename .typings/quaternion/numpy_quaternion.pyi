from quaternion import quaternion as quaternion

_eps: float

def slerp_evaluate(*args, **kwargs):
    """Interpolate linearly along the geodesic between two rotors 

    See also `numpy.slerp_vectorized` for a vectorized version of this function, and
    `quaternion.slerp` for the most useful form, which automatically finds the correct
    rotors to interpolate and the relative time to which they must be interpolated."""
def squad_evaluate(*args, **kwargs):
    """Interpolate linearly along the geodesic between two rotors

    See also `numpy.squad_vectorized` for a vectorized version of this function, and
    `quaternion.squad` for the most useful form, which automatically finds the correct
    rotors to interpolate and the relative time to which they must be interpolated."""
