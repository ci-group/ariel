from _typeshed import Incomplete
from quaternion.numba_wrapper import njit as njit

def unflip_rotors(q, axis: int = -1, inplace: bool = False):
    '''Flip signs of quaternions along axis to ensure continuity

    Quaternions form a "double cover" of the rotation group, meaning that if `q`
    represents a rotation, then `-q` represents the same rotation.  This is clear
    from the way a quaternion is used to rotate a vector `v`: the rotated vector is
    `q * v * q.conjugate()`, which is precisely the same as the vector resulting
    from `(-q) * v * (-q).conjugate()`.  Some ways of constructing quaternions
    (such as converting from rotation matrices or other representations) can result
    in unexpected sign choices.  For many applications, this will not be a problem.
    But if, for example, the quaternions need to be interpolated or differentiated,
    the results may be surprising.  This function flips the signs of successive
    quaternions (along some chosen axis, if relevant), so that successive
    quaternions are as close as possible while still representing the same
    rotations.

    Parameters
    ----------
    q : array_like
        Quaternion array to modify
    axis : int, optional
        Axis along which successive quaternions will be compared.  Default value is
        the last axis of the quaternion array.
    inplace : bool, optional
        If True, modify the data in place without creating a copy; if False (the
        default), a new array is created and returned.

    Returns
    -------
    q_out : array_like
        An array of precisely the same shape as the input array, differing only by
        factors of precisely -1 in some elements.

    '''
def slerp(R1, R2, t1, t2, t_out):
    """Spherical linear interpolation of rotors

    This function uses a simpler interface than the more fundamental
    `slerp_evaluate` and `slerp_vectorized` functions.  The latter
    are fast, being implemented at the C level, but take input `tau`
    instead of time.  This function adjusts the time accordingly.

    Parameters
    ----------
    R1 : quaternion
        Quaternion at beginning of interpolation
    R2 : quaternion
        Quaternion at end of interpolation
    t1 : float
        Time corresponding to R1
    t2 : float
        Time corresponding to R2
    t_out : float or array of floats
        Times to which the rotors should be interpolated

    """
def squad(R_in, t_in, t_out, unflip_input_rotors: bool = False):
    '''Spherical "quadrangular" interpolation of rotors with a cubic spline

    This is typically the best way to interpolate rotation timeseries.
    It uses the analog of a cubic spline, except that the interpolant
    is confined to the rotor manifold in a natural way.  Alternative
    methods involving interpolation of other coordinates on the
    rotation group or normalization of interpolated values give bad
    results.  The results from this method are continuous in value and
    first derivative everywhere, including around the sampling
    locations.

    The input `R_in` rotors are assumed to be reasonably continuous (no
    sign flips), and the input `t` arrays are assumed to be sorted.  No
    checking is done for either case, and you may get silently bad
    results if these conditions are violated.  The first dimension of
    `R_in` must have the same size as `t_in`, but may have additional
    axes following.

    This function simplifies the calling, compared to `squad_evaluate`
    (which takes a set of four quaternions forming the edges of the
    "quadrangle", and the normalized time `tau`) and `squad_vectorized`
    (which takes the same arguments, but in array form, and efficiently
    loops over them).

    Parameters
    ----------
    R_in : array of quaternions
        A time-series of rotors (unit quaternions) to be interpolated
    t_in : array of float
        The times corresponding to R_in
    t_out : array of float
        The times to which R_in should be interpolated
    unflip_input_rotors : bool, optional
        If True, this function calls `unflip_rotors` on the input, to
        ensure that the rotors are more continuous than not.  Defaults
        to False.

    '''
@njit
def frame_from_angular_velocity_integrand(rfrak, Omega): ...

class appending_array:
    _a: Incomplete
    n: int
    def __init__(self, shape, dtype=..., initial_array=None) -> None: ...
    def append(self, row) -> None: ...
    @property
    def a(self): ...

def integrate_angular_velocity(Omega, t0, t1, R0=None, tolerance: float = 1e-12):
    """Compute frame with given angular velocity

    Parameters
    ----------
    Omega : tuple or callable
        Angular velocity from which to compute frame.  Can be
          1) a 2-tuple of float arrays (t, v) giving the angular velocity vector at a series of times,
          2) a function of time that returns the 3-vector angular velocity, or
          3) a function of time and orientation (t, R) that returns the 3-vector angular velocity
        In case 1, the angular velocity will be interpolated to the required times.  Note that accuracy
        is poor in case 1.
    t0 : float
        Initial time
    t1 : float
        Final time
    R0 : quaternion, optional
        Initial frame orientation.  Defaults to 1 (the identity orientation).
    tolerance : float, optional
        Absolute tolerance used in integration.  Defaults to 1e-12.

    Returns
    -------
    t : float array
    R : quaternion array

    """
def minimal_rotation(R, t, iterations: int = 2):
    """Adjust frame so that there is no rotation about z' axis

    The output of this function is a frame that rotates the z axis onto the same z' axis as the
    input frame, but with minimal rotation about that axis.  This is done by pre-composing the input
    rotation with a rotation about the z axis through an angle gamma, where

        dgamma/dt = 2*(dR/dt * z * R.conjugate()).w

    This ensures that the angular velocity has no component along the z' axis.

    Note that this condition becomes easier to impose the closer the input rotation is to a
    minimally rotating frame, which means that repeated application of this function improves its
    accuracy.  By default, this function is iterated twice, though a few more iterations may be
    called for.

    Parameters
    ----------
    R : quaternion array
        Time series describing rotation
    t : float array
        Corresponding times at which R is measured
    iterations : int [defaults to 2]
        Repeat the minimization to refine the result

    """
def angular_velocity(R, t):
    """Approximate angular velocity of a rotating frame

    Parameters
    ----------
    R : array_like
        Quaternion-valued function of time evaluated at a set of times.  This
        represents the quaternion that rotates the standard (x,y,z) frame into the
        moving frame at each instant.
    t : array_like
        Times at which `R` is evaluated.

    Returns
    -------
    Omega : array_like
        The angular velocity (three-vector) as a function of time `t`.  A vector
        that is fixed in the moving frame rotates with this angular velocity with
        respect to the inertial frame.

    Notes
    -----
    The angular velocity at each instant is given by 2 * (dR/dt) / R.  This
    function approximates the input `R` using a cubic spline, and differentiates it
    as such.

    """
