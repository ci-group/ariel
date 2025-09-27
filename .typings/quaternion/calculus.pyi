from quaternion.numba_wrapper import jit as jit, njit as njit, xrange as xrange

def fd_derivative(f, t):
    '''Fourth-order finite-differencing with non-uniform time steps

    The formula for this finite difference comes from Eq. (A 5b) of "Derivative formulas and errors for non-uniformly
    spaced points" by M. K. Bowen and Ronald Smith.  As explained in their Eqs. (B 9b) and (B 10b), this is a
    fourth-order formula -- though that\'s a squishy concept with non-uniform time steps.

    TODO: If there are fewer than five points, the function should revert to simpler (lower-order) formulas.

    '''
@njit
def _derivative(f, t, dfdt) -> None: ...
@njit
def _derivative_2d(f, t, dfdt) -> None: ...
@njit
def _derivative_3d(f, t, dfdt) -> None: ...
@njit
def fd_indefinite_integral(f, t): ...
def fd_definite_integral(f, t): ...
def spline_evaluation(f, t, t_out=None, axis=None, spline_degree: int = 3, derivative_order: int = 0, definite_integral_bounds=None):
    """Approximate input data using a spline and evaluate

    Note that this function is somewhat more general than it needs to be, so that it can be reused
    for closely related functions involving derivatives, antiderivatives, and integrals.

    Parameters
    ==========
    f : (..., N, ...) array_like
        Real or complex function values to be interpolated.

    t : (N,) array_like
        An N-D array of increasing real values. The length of f along the interpolation axis must be
        equal to the length of t.  The number of data points must be larger than the spline degree.

    t_out : None or (M,) array_like [defaults to None]
        The new values of `t` on which to evaluate the result.  If None, it is assumed that some
        other feature of the data is needed, like a derivative or antiderivative, which are then
        output using the same `t` values as the input.

    axis : None or int [defaults to None]
        The axis of `f` with length equal to the length of `t`.  If None, this function searches for
        an axis of equal length in reverse order -- that is, starting from the last axis of `f`.
        Note that this feature is helpful when `f` is one-dimensional or will always satisfy that
        criterion, but is dangerous otherwise.  Caveat emptor.

    spline_degree : int [defaults to 3]
        Degree of the interpolating spline. Must be 1 <= spline_degree <= 5.

    derivative_order : int [defaults to 0]
        The order of the derivative to apply to the data.  Note that this may be negative, in which
        case the corresponding antiderivative is returned.

    definite_integral_bounds : None or (2,) array_like [defaults to None]
        If this is not None, the `t_out` and `derivative_order` parameters are ignored, and the
        returned values are just the (first) definite integrals of the splines between these limits,
        along each remaining axis.

    """
def spline_derivative(f, t, derivative_order: int = 1, axis: int = 0): ...
def spline_indefinite_integral(f, t, integral_order: int = 1, axis: int = 0): ...
def spline_definite_integral(f, t, t1=None, t2=None, axis: int = 0): ...
spline = spline_evaluation
derivative = spline_derivative
antiderivative = spline_indefinite_integral
indefinite_integral = spline_indefinite_integral
definite_integral = spline_definite_integral
derivative = fd_derivative
antiderivative = fd_indefinite_integral
indefinite_integral = fd_indefinite_integral
definite_integral = fd_definite_integral
