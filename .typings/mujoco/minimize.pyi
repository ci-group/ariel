import abc
import dataclasses
import enum
import numpy as np
from _typeshed import Incomplete
from typing import Callable, Sequence, TextIO

class Verbosity(enum.Enum):
    SILENT = 0
    FINAL = 1
    ITER = 2
    FULLITER = 3

class Status(enum.Enum):
    FACTORIZATION_FAILED = ...
    NO_IMPROVEMENT = ...
    MAX_ITER = ...
    DX_TOL = ...
    G_TOL = ...

_STATUS_MESSAGE: Incomplete

@dataclasses.dataclass(frozen=True)
class IterLog:
    """Log of a single iteration of the non-linear least-squares solver.

  Attributes:
    candidate: Value of the decision variable at the beginning this iteration.
    objective: Value of the objective at the candidate.
    reduction: Reduction of the objective during this iteration.
    regularizer: Value of the regularizer used for this iteration.
    residual: Optional value of the residual at the candidate.
    jacobian: Optional value of the Jacobian at the candidate.
    grad: Optional value of the gradient at the candidate.
    step: Optional change in decision variable during this iteration.
  """
    candidate: np.ndarray
    objective: np.float64
    reduction: np.float64
    regularizer: np.float64
    residual: np.ndarray | None = ...
    jacobian: np.ndarray | None = ...
    grad: np.ndarray | None = ...
    step: np.ndarray | None = ...

class Norm(abc.ABC, metaclass=abc.ABCMeta):
    '''Abstract interface for norm functions, measuring the magnitude of vectors.

  Key Concepts:

  * Norm Value: The value of the norm for a given input vector.
  * Gradient and Hessian: The gradient (first derivative) and Hessian (second
    derivative) of the norm function with respect to the input vector.

  Subclasses Must Implement:

  * `value(self, r: np.ndarray)`: Computes and returns the norm value for the
     input vector `r`.
  * `grad_hess(self, r: np.ndarray, proj: np.ndarray)`: Computes and returns
     both  the gradient and Hessian of the norm at `r`, projected onto `proj`.
     The reason we ask the user to perform the projection themselves is that
     norm Hessians are often large and sparse, and the "sandwich" projection
     operator `proj.T @ hess @ proj` can be computed efficiently by taking the
     specific norm structure into account.
  '''
    @abc.abstractmethod
    def value(self, r: np.ndarray) -> np.float64:
        """Returns the value of the norm at the input vector `y = norm(r)`."""
    @abc.abstractmethod
    def grad_hess(self, r: np.ndarray, proj: np.ndarray):
        """Computes the projected gradient and Hessian of the norm at `r`.

    Args:
        r: A NumPy column vector (nr x 1).
        proj: A pre-computed projection matrix (nr x nx).

    Returns:
        A tuple containing:
            * Projected gradient: proj.T @ (d_norm/d_r).
            * Projected Hessian: proj.T @ (d^2_norm/d_r^2) @ proj.
    """

class Quadratic(Norm):
    """Implementation of the quadratic norm."""
    def value(self, r: np.ndarray):
        """Returns the quadratic norm of `r`."""
    def grad_hess(self, r: np.ndarray, proj: np.ndarray):
        """Computes the projected gradient and Hessian of the quadratic norm at `r`.

    Args:
        r: A NumPy column vector (nr x 1).
        proj: A pre-computed projection matrix (nr x nx).

    Returns:
        A tuple containing:
            * Projected gradient: `proj.T @ r`.
            * Projected Hessian: `proj.T @ proj`.
    """

def least_squares(x0: np.ndarray, residual: Callable[[np.ndarray], np.ndarray], bounds: Sequence[np.ndarray] | None = None, jacobian: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None, norm: Norm = ..., eps: float = ..., mu_min: float = 1e-06, mu_max: float = 100000000.0, mu_factor: float = ..., xtol: float = 1e-08, gtol: float = 1e-08, max_iter: int = 100, verbose: Verbosity | int = ..., output: TextIO | None = None, iter_callback: Callable[[list[IterLog]], None] | None = None, check_derivatives: bool = False) -> tuple[np.ndarray, list[IterLog]]:
    """Nonlinear Least Squares minimization with box bounds.

  Args:
    x0: Initial guess
    residual: Vectorized function returning the residual for 1 or more points.
    bounds: Optional pair of lower and upper bounds on the solution.
    jacobian: Optional function that returns Jacobian of the residual at a given
      point and residual. If not given, `residual` will be finite-differenced.
    norm: Norm object returning norm scalar or its projected gradient and
      Hessian. See Norm class for detailed documentation.
    eps: Perurbation used for automatic finite-differencing.
    mu_min: Minimum value of the regularizer.
    mu_max: Maximum value of the regularizer.
    mu_factor: Factor for increasing or decreasing the regularizer.
    xtol: Termination tolerance on relative step size.
    gtol: Termination tolerance on gradient norm.
    max_iter: Maximum number of iterations.
    verbose: Verbosity level.
    output: Optional file or StringIO to which to print messages.
    iter_callback: Optional iteration callback, takes trace argument.
    check_derivatives: Compare user-defined Jacobian and norm against fin-diff.

  Returns:
    x: best solution found
    trace: sequence of solution iterates.
  """
def jacobian_fd(residual: Callable[[np.ndarray], np.ndarray], x: np.ndarray, r: np.ndarray, eps: np.float64, n_res: int, bounds: list[np.ndarray] | None = None) -> tuple[np.ndarray, int]:
    """Finite-difference Jacobian of a residual function.

  Args:
    residual: vectorized function that returns the residual of a vector array.
    x: point at which to evaluate the Jacobian.
    r: residual at x.
    eps: finite-difference step size.
    n_res: number or residual evaluations so far.
    bounds: optional pair of lower and upper bounds.

  Returns:
    jac: Jacobian of the residual at x.
    n_res: updated number of residual evaluations (add x.size).
  """
def check_jacobian(residual: Callable[[np.ndarray], np.ndarray], x: np.ndarray, r: np.ndarray, jac: np.ndarray, eps: np.float64, n_res: int, bounds: list[np.ndarray] | None = None, output: TextIO | None = None, name: str | None = 'Jacobian') -> int:
    """Check user-provided Jacobian against internal finite-differencing.

  Args:
    residual: vectorized function that returns the residual of a vector array.
    x: point at which the r and jac were evaluated.
    r: residual at x.
    jac: Jacobian at x.
    eps: finite-difference step size.
    n_res: number or residual evaluations so far.
    bounds: optional pair of lower and upper bounds.
    output: Optional file or StringIO to which to print messages.
    name: Optional name of the function being tested.

  Returns:
    n_res: updated number of residual evaluations.
  """
def check_norm(r: np.ndarray, norm: Norm, eps: np.float64, output: TextIO | None = None):
    """Check user-provided norm against internal finite-differencing.

  Args:
    r: residual vector.
    norm: Norm function returning either the norm scalar or its gradient and
      Gauss-Newton Hessian.
    eps: finite-difference step size.
    output: Optional file or StringIO to which to print messages.
  """
