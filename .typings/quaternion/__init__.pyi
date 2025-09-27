from .calculus import definite_integral as definite_integral, derivative as derivative, indefinite_integral as indefinite_integral
from .numpy_quaternion import quaternion as quaternion, slerp_evaluate as slerp_evaluate, squad_evaluate as squad_evaluate
from .quaternion_time_series import integrate_angular_velocity as integrate_angular_velocity, slerp as slerp, squad as squad
from _typeshed import Incomplete

__all__ = ['quaternion', 'as_quat_array', 'as_spinor_array', 'as_float_array', 'from_float_array', 'as_vector_part', 'from_vector_part', 'as_rotation_matrix', 'from_rotation_matrix', 'as_rotation_vector', 'from_rotation_vector', 'as_euler_angles', 'from_euler_angles', 'as_spherical_coords', 'from_spherical_coords', 'rotate_vectors', 'allclose', 'rotor_intrinsic_distance', 'rotor_chordal_distance', 'rotation_intrinsic_distance', 'rotation_chordal_distance', 'slerp_evaluate', 'squad_evaluate', 'zero', 'one', 'x', 'y', 'z', 'integrate_angular_velocity', 'squad', 'slerp', 'derivative', 'definite_integral', 'indefinite_integral']

zero: Incomplete
one: Incomplete
x: Incomplete
y: Incomplete
z: Incomplete
rotor_intrinsic_distance: Incomplete
rotor_chordal_distance: Incomplete
rotation_intrinsic_distance: Incomplete
rotation_chordal_distance: Incomplete

def as_float_array(a):
    '''View the quaternion array as an array of floats

    This function is fast (of order 1 microsecond) because no data is
    copied; the returned quantity is just a "view" of the original.

    The output view has one more dimension (of size 4) than the input
    array, but is otherwise the same shape.  The components along
    that last dimension represent the scalar and vector components of
    each quaternion in that order: `w`, `x`, `y`, `z`.

    '''
def as_quat_array(a):
    '''View a float array as an array of quaternions

    The input array must have a final dimension whose size is
    divisible by four (or better yet *is* 4), because successive
    indices in that last dimension will be considered successive
    components of the output quaternion.  Each set of 4 components
    will be interpreted as the scalar and vector components of a
    quaternion in that order: `w`, `x`, `y`, `z`.

    This function is usually fast (of order 1 microsecond) because no
    data is copied; the returned quantity is just a "view" of the
    original.  However, if the input array is not C-contiguous
    (basically, as you increment the index into the last dimension of
    the array, you just move to the neighboring float in memory), the
    data will need to be copied which may be quite slow.  Therefore,
    you should try to ensure that the input array is in that order.
    Slices and transpositions will frequently break that rule.

    We will not convert back from a two-spinor array because there is
    no unique convention for them, so I don\'t want to mess with that.
    Also, we want to discourage users from the slow, memory-copying
    process of swapping columns required for useful definitions of
    the two-spinors.

    '''
def from_float_array(a): ...
def from_vector_part(v, vector_axis: int = -1):
    """Create a quaternion array from an array of the vector parts.

    Essentially, this just inserts a 0 in front of each vector part, and
    re-interprets the result as a quaternion.

    Parameters
    ----------
    v : array_like
        Array of vector parts of quaternions.  When interpreted as a numpy array,
        if the dtype is `quaternion`, the array is returned immediately, and the
        following argument is ignored.  Otherwise, it it must be a float array with
        dimension `vector_axis` of size 3 or 4.
    vector_axis : int, optional
        The axis to interpret as containing the vector components.  The default is
        -1.

    Returns
    -------
    q : array of quaternions
        Quaternions with vector parts corresponding to input vectors.

    """
def as_vector_part(q):
    """Create an array of vector parts from an array of quaternions.

    Parameters
    ----------
    q : quaternion array_like
        Array of quaternions.

    Returns
    -------
    v : array
        Float array of shape `q.shape + (3,)`

    """
def as_spinor_array(a):
    '''View a quaternion array as spinors in two-complex representation

    This function is relatively slow and scales poorly, because memory
    copying is apparently involved -- I think it\'s due to the
    "advanced indexing" required to swap the columns.

    '''
def as_rotation_matrix(q):
    """Convert input quaternion to 3x3 rotation matrix

    For any quaternion `q`, this function returns a matrix `m` such that, for every
    vector `v`, we have

        m @ v.vec == q * v * q.conjugate()

    Here, `@` is the standard python matrix multiplication operator and `v.vec` is
    the 3-vector part of the quaternion `v`.

    Parameters
    ----------
    q : quaternion or array of quaternions
        The quaternion(s) need not be normalized, but must all be nonzero

    Returns
    -------
    m : float array
        Output shape is q.shape+(3,3).  This matrix should multiply (from
        the left) a column vector to produce the rotated column vector.

    Raises
    ------
    ZeroDivisionError
        If any of the input quaternions have norm 0.0.

    """
def from_rotation_matrix(rot, nonorthogonal: bool = True):
    '''Convert input 3x3 rotation matrix to unit quaternion

    For any orthogonal matrix `rot`, this function returns a quaternion `q` such
    that, for every pure-vector quaternion `v`, we have

        q * v * q.conjugate() == rot @ v.vec

    Here, `@` is the standard python matrix multiplication operator and `v.vec` is
    the 3-vector part of the quaternion `v`.  If `rot` is not orthogonal the
    "closest" orthogonal matrix is used; see Notes below.

    Parameters
    ----------
    rot : (..., N, 3, 3) float array
        Each 3x3 matrix represents a rotation by multiplying (from the left) a
        column vector to produce a rotated column vector.  Note that this input may
        actually have ndims>3; it is just assumed that the last two dimensions have
        size 3, representing the matrix.
    nonorthogonal : bool, optional
        If scipy.linalg is available, use the more robust algorithm of Bar-Itzhack.
        Default value is True.

    Returns
    -------
    q : array of quaternions
        Unit quaternions resulting in rotations corresponding to input rotations.
        Output shape is rot.shape[:-2].

    Raises
    ------
    LinAlgError
        If any of the eigenvalue solutions does not converge

    Notes
    -----
    By default, if scipy.linalg is available, this function uses Bar-Itzhack\'s
    algorithm to allow for non-orthogonal matrices.  [J. Guidance, Vol. 23, No. 6,
    p. 1085 <http://dx.doi.org/10.2514/2.4654>] This will almost certainly be quite
    a bit slower than simpler versions, though it will be more robust to numerical
    errors in the rotation matrix.  Also note that Bar-Itzhack uses some pretty
    weird conventions.  The last component of the quaternion appears to represent
    the scalar, and the quaternion itself is conjugated relative to the convention
    used throughout this module.

    If scipy.linalg is not available or if the optional `nonorthogonal` parameter
    is set to `False`, this function falls back to the possibly faster, but less
    robust, algorithm of Markley [J. Guidance, Vol. 31, No. 2, p. 440
    <http://dx.doi.org/10.2514/1.31730>].

    '''
def as_rotation_vector(q):
    """Convert input quaternion to the axis-angle representation

    Note that if any of the input quaternions has norm zero, no error is
    raised, but NaNs will appear in the output.

    Parameters
    ----------
    q : quaternion or array of quaternions
        The quaternion(s) need not be normalized, but must all be nonzero

    Returns
    -------
    rot : float array
        Output shape is q.shape+(3,).  Each vector represents the axis of
        the rotation, with norm proportional to the angle of the rotation in
        radians.

    """
def from_rotation_vector(rot):
    """Convert input 3-vector in axis-angle representation to unit quaternion

    Parameters
    ----------
    rot : (Nx3) float array
        Each vector represents the axis of the rotation, with norm
        proportional to the angle of the rotation in radians.

    Returns
    -------
    q : array of quaternions
        Unit quaternions resulting in rotations corresponding to input
        rotations.  Output shape is rot.shape[:-1].

    """
def as_euler_angles(q):
    '''Open Pandora\'s Box

    If somebody is trying to make you use Euler angles, tell them no, and
    walk away, and go and tell your mum.

    You don\'t want to use Euler angles.  They are awful.  Stay away.  It\'s
    one thing to convert from Euler angles to quaternions; at least you\'re
    moving in the right direction.  But to go the other way?!  It\'s just not
    right.

    Assumes the Euler angles correspond to the quaternion R via

        R = exp(alpha*z/2) * exp(beta*y/2) * exp(gamma*z/2)

    The angles are naturally in radians.

    NOTE: Before opening an issue reporting something "wrong" with this
    function, be sure to read all of the following page, *especially* the
    very last section about opening issues or pull requests.
    <https://github.com/moble/quaternion/wiki/Euler-angles-are-horrible>

    Parameters
    ----------
    q : quaternion or array of quaternions
        The quaternion(s) need not be normalized, but must all be nonzero

    Returns
    -------
    alpha_beta_gamma : float array
        Output shape is q.shape+(3,).  These represent the angles (alpha,
        beta, gamma) in radians, where the normalized input quaternion
        represents `exp(alpha*z/2) * exp(beta*y/2) * exp(gamma*z/2)`.

    Raises
    ------
    AllHell
        ...if you try to actually use Euler angles, when you could have
        been using quaternions like a sensible person.

    '''
def from_euler_angles(alpha_beta_gamma, beta=None, gamma=None):
    '''Improve your life drastically

    Assumes the Euler angles correspond to the quaternion R via

        R = exp(alpha*z/2) * exp(beta*y/2) * exp(gamma*z/2)

    The angles naturally must be in radians for this to make any sense.

    NOTE: Before opening an issue reporting something "wrong" with this
    function, be sure to read all of the following page, *especially* the
    very last section about opening issues or pull requests.
    <https://github.com/moble/quaternion/wiki/Euler-angles-are-horrible>

    Parameters
    ----------
    alpha_beta_gamma : float or array of floats
        This argument may either contain an array with last dimension of
        size 3, where those three elements describe the (alpha, beta, gamma)
        radian values for each rotation; or it may contain just the alpha
        values, in which case the next two arguments must also be given.
    beta : None, float, or array of floats
        If this array is given, it must be able to broadcast against the
        first and third arguments.
    gamma : None, float, or array of floats
        If this array is given, it must be able to broadcast against the
        first and second arguments.

    Returns
    -------
    R : quaternion array
        The shape of this array will be the same as the input, except that
        the last dimension will be removed.

    '''
def as_spherical_coords(q):
    """Return the spherical coordinates corresponding to this quaternion

    Obviously, spherical coordinates do not contain as much information as a
    quaternion, so this function does lose some information.  However, the
    returned spherical coordinates will represent the point(s) on the sphere
    to which the input quaternion(s) rotate the z axis.

    Parameters
    ----------
    q : quaternion or array of quaternions
        The quaternion(s) need not be normalized, but must be nonzero

    Returns
    -------
    vartheta_varphi : float array
        Output shape is q.shape+(2,).  These represent the angles (vartheta,
        varphi) in radians, where the normalized input quaternion represents
        `exp(varphi*z/2) * exp(vartheta*y/2)`, up to an arbitrary inital
        rotation about `z`.

    """
def from_spherical_coords(theta_phi, phi=None):
    """Return the quaternion corresponding to these spherical coordinates

    Assumes the spherical coordinates correspond to the quaternion R via

        R = exp(phi*z/2) * exp(theta*y/2)

    The angles naturally must be in radians for this to make any sense.

    Note that this quaternion rotates `z` onto the point with the given
    spherical coordinates, but also rotates `x` and `y` onto the usual basis
    vectors (theta and phi, respectively) at that point.

    Parameters
    ----------
    theta_phi : float or array of floats
        This argument may either contain an array with last dimension of
        size 2, where those two elements describe the (theta, phi) values in
        radians for each point; or it may contain just the theta values in
        radians, in which case the next argument must also be given.
    phi : None, float, or array of floats
        If this array is given, it must be able to broadcast against the
        first argument.

    Returns
    -------
    R : quaternion array
        If the second argument is not given to this function, the shape
        will be the same as the input shape except for the last dimension,
        which will be removed.  If the second argument is given, this
        output array will have the shape resulting from broadcasting the
        two input arrays against each other.

    """
def rotate_vectors(R, v, axis: int = -1):
    """Rotate vectors by given quaternions

    This function is for the case where each quaternion (possibly the only input
    quaternion) is used to rotate multiple vectors.  If each quaternion is only
    rotating a single vector, it is more efficient to use the standard formula

        vprime = R * v * R.conjugate()

    (Note that `from_vector_part` and `as_vector_part` may be helpful.)

    Parameters
    ----------
    R : quaternion array
        Quaternions by which to rotate the input vectors
    v : float array
        Three-vectors to be rotated.
    axis : int
        Axis of the `v` array to use as the vector dimension.  This
        axis of `v` must have length 3.

    Returns
    -------
    vprime : float array
        The rotated vectors.  This array has shape R.shape+v.shape.

    Notes
    -----
    For simplicity, this function converts the input quaternion(s) to matrix form,
    and rotates the input vector(s) by the usual matrix multiplication.  As noted
    above, if each input quaternion is only used to rotate a single vector, this is
    not the most efficient approach.  The simple formula shown above is faster than
    this function, though it should be noted that the most efficient approach (in
    terms of operation counts) is to use the formula

      v' = v + 2 * r x (s * v + r x v) / m

    where x represents the cross product, s and r are the scalar and vector parts
    of the quaternion, respectively, and m is the sum of the squares of the
    components of the quaternion.  If you are looping over a very large number of
    quaternions, and just rotating a single vector each time, you might want to
    implement that alternative algorithm using numba (or something that doesn't use
    python).

    """
def allclose(a, b, rtol=..., atol: float = 0.0, equal_nan: bool = False, verbose: bool = False):
    """Returns True if two arrays are element-wise equal within a tolerance.

    This function is essentially a wrapper for the `quaternion.isclose`
    function, but returns a single boolean value of True if all elements
    of the output from `quaternion.isclose` are True, and False otherwise.
    This function also adds the option.

    Note that this function has stricter tolerances than the
    `numpy.allclose` function, as well as the additional `verbose` option.

    Parameters
    ----------
    a, b : array_like
        Input arrays to compare.
    rtol : float
        The relative tolerance parameter (see Notes).
    atol : float
        The absolute tolerance parameter (see Notes).
    equal_nan : bool
        Whether to compare NaN's as equal.  If True, NaN's in `a` will be
        considered equal to NaN's in `b` in the output array.
    verbose : bool
        If the return value is False, all the non-close values are printed,
        iterating through the non-close indices in order, displaying the
        array values along with the index, with a separate line for each
        pair of values.

    See Also
    --------
    isclose, numpy.all, numpy.any, numpy.allclose

    Returns
    -------
    allclose : bool
        Returns True if the two arrays are equal within the given
        tolerance; False otherwise.


    Notes
    -----
    If the following equation is element-wise True, then allclose returns
    True.

      absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))

    The above equation is not symmetric in `a` and `b`, so that
    `allclose(a, b)` might be different from `allclose(b, a)` in
    some rare cases.

    """
