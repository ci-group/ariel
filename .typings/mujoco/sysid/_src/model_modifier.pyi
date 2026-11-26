import mujoco
import numpy as np
from mujoco.sysid._src.parameter import InertiaType as InertiaType, ModifierFn as ModifierFn, Parameter as Parameter, ParameterDict as ParameterDict

def remove_visuals(in_spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Remove visual elements from a Spec."""
def apply_param_modifiers_spec(params: ParameterDict, spec: mujoco.MjSpec) -> mujoco.MjSpec: ...
def apply_param_modifiers(params: ParameterDict, spec: mujoco.MjSpec) -> mujoco.MjModel: ...
def is_position_actuator(actuator) -> bool:
    """Check if an actuator is a position actuator.

  This function works on both model.actuator and spec.actuator objects.

  Args:
    actuator: An actuator object from model or spec.

  Returns:
    True if the actuator is a position actuator.
  """
def get_actuator_pd_gains(model: mujoco.MjModel, actuator_name: str) -> tuple[float, float]:
    """Return the (P, D) gains of a position actuator.

  Args:
    model: MuJoCo model.
    actuator_name: Name of the actuator.
  """
def apply_pgain(spec: mujoco.MjSpec, actuator_name: str, value: float | np.ndarray) -> mujoco.MjSpec:
    """Set the proportional gain for a position actuator.

  Args:
    spec: MuJoCo model specification.
    actuator_name: Name of the actuator.
    value: Proportional gain value.
  """
def apply_dgain(spec: mujoco.MjSpec, actuator_name: str, value: float | np.ndarray) -> mujoco.MjSpec:
    """Set the derivative gain for a position actuator.

  Args:
    spec: MuJoCo model specification.
    actuator_name: Name of the actuator.
    value: Derivative gain value.
  """
def apply_pdgain(spec: mujoco.MjSpec, actuator_name: str, value: np.ndarray) -> mujoco.MjSpec:
    """Set both proportional and derivative gains for a position actuator.

  Args:
    spec: MuJoCo model specification.
    actuator_name: Name of the actuator.
    value: 2-element array ``[P_gain, D_gain]``.
  """
def apply_body_mass_ipos(spec: mujoco.MjSpec, body_name: str, mass: np.ndarray | None = None, ipos: np.ndarray | None = None, rot_inertia_scale: bool = False) -> mujoco.MjSpec:
    """Apply mass and center-of-mass position to a body.

  Args:
    spec: MuJoCo model specification.
    body_name: Name of the body.
    mass: Optional new mass value.
    ipos: Optional new center-of-mass position.
    rot_inertia_scale: If True, scale rotational inertia proportionally to
      mass change.
  """
def scale_body_inertia(spec: mujoco.MjSpec, body_name: str, value: np.ndarray) -> mujoco.MjSpec: ...
def pi_from_theta(theta: np.ndarray) -> np.ndarray:
    """Convert base parameters θ to inertial parameters π.

  Args:
    theta: 10-D array [alpha, d1, d2, d3, s12, s23, s13, t1, t2, t3] where:
      alpha: Scale parameter (log of U[3,3])
      [d1, d2, d3]: Log of diagonal elements
      [s12, s23, s13]: Shear parameters from upper triangle
      [t1, t2, t3]: Translation parameters from last column

  Returns:
    10-D array π = [m, hx, hy, hz, Ixx, Iyy, Izz, Ixy, Iyz, Ixz].
  """
def pseudoinertia_from_pi(pi: np.ndarray) -> np.ndarray:
    """Converts inertial parameters π to a 4x4 pseudoinertia matrix J.

  Args:
    pi: A 10-D array [m, hx, hy, hz, Ixx, Iyy, Izz, Ixy, Iyz, Ixz].

  Returns:
    A 4x4 pseudoinertia matrix J.
  """
def cholesky_decompose_upper(J: np.ndarray) -> np.ndarray:
    """Perform an upper-triangular Cholesky decomposition of J.

  The returned matrix U is such that J = U @ U.T.

  Args:
    J: A 4x4 positive-definite matrix.

  Returns:
    An upper-triangular matrix U.
  """
def theta_from_pseudoinertia(J: np.ndarray) -> np.ndarray:
    """Extract the 10-D vector of base parameters θ from the pseudoinertia J.

  Args:
    J: A 4x4 pseudoinertia.

  Returns:
    A 10-D array θ = [alpha, d1, d2, d3, s12, s23, s13, t1, t2, t3] where:
      alpha: Scale parameter (log of U[3,3])
      [d1, d2, d3]: Log of diagonal elements
      [s12, s23, s13]: Shear parameters from upper triangle
      [t1, t2, t3]: Translation parameters from last column
  """
def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix from a length-3 vector."""
def inertia_to_fullinertia(q: np.ndarray, inertia: np.ndarray) -> np.ndarray:
    """Convert diagonal inertia with quaternion to full inertia matrix."""
def pi_from_body(spec: mujoco.MjSpec, body_name: str) -> np.ndarray:
    """Extracts the 10-D vector of inertial parameters π from a MuJoCo body.

  Args:
    spec: MuJoCo model specification object.
    body_name: Name of the body to extract parameters from.

  Returns:
    A 10-D numpy array π = [m, hx, hy, hz, Ixx, Iyy, Izz, Ixy, Iyz, Ixz] where:
      m: Mass of the body
      [hx, hy, hz]: First moment of mass (m * com, where com is center of mass)
      [Ixx, Iyy, Izz, Ixy, Iyz, Ixz]: Rotational inertia about the origin of the
        body-fixed reference frame.
  """
def theta_inertia_from_body(spec: mujoco.MjSpec, body_name: str) -> np.ndarray:
    """Extract base parameters θ from a body's inertial properties."""
def apply_body_theta_inertia(spec: mujoco.MjSpec, body_name: str, theta: np.ndarray) -> mujoco.MjSpec:
    """Apply base-parameter inertia θ to a body in the spec.

  Args:
    spec: MuJoCo model specification.
    body_name: Name of the body.
    theta: 10-element array [alpha, d1, d2, d3, s12, s23, s13, t1, t2, t3].
  """
def apply_body_inertia(spec: mujoco.MjSpec, name: str, param: Parameter):
    """Apply inertia parameters to a body based on the parameter type.

  Args:
    spec: MuJoCo model specification.
    name: Name of the body.
    param: Parameter with an ``inertia_type`` attribute.
  """
def body_inertia_param(spec: mujoco.MjSpec, model: mujoco.MjModel, body_name: str, inertia_type: InertiaType = ..., scale_rot_inertia: bool = False, mass_bound_mult: np.ndarray | None = None, ipos_bound_off: np.ndarray | None = None, stretch_bound_mult: np.ndarray | None = None, shear_bound_off: np.ndarray | None = None, param_name: str | None = None, modifier: ModifierFn | None = None) -> Parameter:
    '''Creates Parameter objects for the inertia of a body in a simplified manner.

  Args:
    spec: MuJoCo model specification object.
    model: The MuJoCo model.
    body_name: Name of the body to create the parameter for.
    inertia_type: The type of inertia parameterization to use.
    scale_rot_inertia: Whether to scale the original inertia when mass changes,
      ignored with pseudo inertia.
    mass_bound_mult: Multiplicative bounds for the mass parameter.
    ipos_bound_off: Additive bounds for the ipos parameter.
    stretch_bound_mult: Multiplicative bounds for the stretch parameters in the
      pseudo-inertia parameterization.
    shear_bound_off: Additive bounds for the shear parameters in the
      pseudo-inertia parameterization.
    param_name: Optional name for the parameter.  Defaults to
      ``"{body_name}_inertia"``.
    modifier: Optional custom modifier callback.  If None, the default
      :func:`apply_body_inertia` modifier is registered on the Parameter.

  Returns:
    A Parameter configured for the body\'s inertia.
  '''
