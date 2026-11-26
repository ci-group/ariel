import scipy
import scipy.optimize as scipy_optimize
from collections.abc import Callable as Callable
from mujoco.sysid._src import parameter as parameter
from typing import Any, Literal

def optimize(initial_params: parameter.ParameterDict, residual_fn: Callable[..., Any], optimizer: Literal['scipy', 'mujoco', 'scipy_parallel_fd'] = 'mujoco', verbose: bool = True, check_conditioning: bool = False, **optimizer_kwargs) -> tuple[parameter.ParameterDict, scipy_optimize.OptimizeResult]:
    '''Run nonlinear least-squares optimization on the residual.

  Args:
    initial_params: Starting parameter values and bounds.
    residual_fn: Callable with signature ``(x, params) -> (residuals, ...)`` as
      returned by :func:`build_residual_fn`.
    optimizer: Backend — ``"mujoco"`` (default), ``"scipy"``, or
      ``"scipy_parallel_fd"`` (scipy with MuJoCo finite-difference Jacobian).
    verbose: If True, log parameter comparison table after optimization.
    check_conditioning: If True, estimate ``cond(J^T J)`` at the starting
      point and emit a warning if it suggests numerical ill-conditioning.
      Costs one extra finite-difference Jacobian.
    **optimizer_kwargs: Forwarded to the backend. Common ones:

      * ``max_iters``: maximum number of optimizer iterations.
      * ``verbose``: per-backend verbosity flag (separate from this
        function\'s ``verbose``).
      * ``loss``: scipy loss function name (scipy backends only).
      * ``x_scale``: per-parameter scaling, forwarded to the backend; see
        :func:`scipy.optimize.least_squares` and
        :func:`mujoco.minimize.least_squares`.

  Returns:
    ``(opt_params, opt_result)`` — the optimized ParameterDict and a
    ``scipy.optimize.OptimizeResult`` with at least ``x``, ``jac``, ``grad``.
  '''
def calculate_intervals(residuals_star, J, alpha: float = 0.05, lambda_zero_thresh: float = 1e-15, v_zero_thresh: float = 1e-08):
    """Calculate confidence intervals from the Jacobian at the optimum.

  Args:
    residuals_star: List of residual arrays at the optimum.
    J: Jacobian matrix at the optimum, shape ``(n_residuals, n_params)``.
    alpha: Significance level for the confidence intervals.
    lambda_zero_thresh: Threshold below which eigenvalues are treated as zero.
    v_zero_thresh: Threshold below which eigenvector elements are treated as
      zero.

  Returns:
    ``(Sigma_X, intervals)`` — the parameter covariance matrix and the
    half-width confidence intervals for each parameter.
  """
