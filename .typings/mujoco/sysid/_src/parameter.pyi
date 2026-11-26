import mujoco
import numpy as np
import numpy.typing as npt
import pathlib
from _typeshed import Incomplete
from enum import Enum
from typing import TypeAlias
from typing_extensions import Self

Fore: Incomplete
Style: Incomplete
ModifierFn: TypeAlias

class InertiaType(Enum):
    Mass = 0
    MassIpos = 1
    Pseudo = 2

class Parameter:
    """A single (possibly multi-dimensional) parameter for system identification.

  A Parameter holds a current ``value``, a ``nominal`` baseline, and box
  bounds (``min_value``, ``max_value``).  An optional ``modifier`` callback
  is invoked during model compilation to apply the parameter to an MjSpec.

  Args:
    name: Human-readable identifier (must be unique within a ParameterDict).
    nominal: Nominal (initial) value; scalar or array-like.
    min_value: Lower bound, same shape as *nominal*.
    max_value: Upper bound, same shape as *nominal*.
    frozen: If True the parameter is excluded from optimization.
    modifier: Optional callback ``(MjSpec, Parameter) -> None`` that writes the
      parameter into a spec during model compilation.
  """
    inertia_type: InertiaType | None
    scale_rot_inertia: bool
    name: Incomplete
    nominal: Incomplete
    min_value: Incomplete
    max_value: Incomplete
    value: Incomplete
    frozen: Incomplete
    modifier: Incomplete
    def __init__(self, name: str, nominal: float | npt.ArrayLike, min_value: float | npt.ArrayLike, max_value: float | npt.ArrayLike, frozen: bool = False, modifier: ModifierFn | None = None) -> None: ...
    @property
    def size(self) -> int: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    def apply_modifier(self, spec: mujoco.MjSpec) -> None:
        """Apply this parameter's modifier callback to *spec*, if one is set."""
    def as_vector(self) -> np.ndarray:
        """Return the current value as a flat 1-D array."""
    def as_nominal_vector(self) -> np.ndarray:
        """Return the nominal value as a flat 1-D array."""
    def update_from_vector(self, vector: np.ndarray) -> None:
        """Update the current value from a flat vector.

    Args:
      vector: Flat array of length ``self.size``.
    """
    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(lower, upper)`` bound arrays, each flat 1-D."""
    def reset(self) -> None:
        """Reset the current value to nominal."""
    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample a random value uniformly within bounds.

    Args:
      rng: Optional numpy random generator. Uses default if None.
    """
    def move_off_bound(self, fraction: float = 0.05, at_bound_tol: float = 0.001) -> None:
        """Shift values within ``at_bound_tol * (hi - lo)`` of a bound to
    ``lo + fraction*(hi-lo)`` (or symmetric for the upper bound). Interior
    components are unchanged."""
    def __deepcopy__(self, memo): ...

class ParameterDict:
    """An ordered collection of :class:`Parameter` objects.

  Behaves like a ``dict[str, Parameter]`` with convenience methods for
  vectorized access (``as_vector`` / ``update_from_vector``), serialization,
  and tabular comparison of parameter estimates.

  Frozen parameters are silently skipped by vector/bounds methods so that the
  decision-variable dimension seen by optimizers matches only the free params.
  """
    parameters: Incomplete
    def __init__(self, parameters: dict[str, Parameter] | None = None) -> None: ...
    def __getitem__(self, key: str) -> Parameter: ...
    def __setitem__(self, key: str, value: Parameter) -> None: ...
    def __contains__(self, key: str) -> bool: ...
    def __len__(self) -> int: ...
    def copy(self) -> Self:
        """Return a deep copy of this ParameterDict."""
    def add(self, param: Parameter) -> None:
        """Add a Parameter, keyed by its ``name``."""
    def update(self, pdict: Self) -> None: ...
    def keys(self) -> list[str]: ...
    def values(self) -> list[Parameter]: ...
    def items(self) -> list[tuple[str, Parameter]]: ...
    @property
    def size(self) -> int:
        """Get the total size of all non-frozen parameters."""
    def as_vector(self, include_frozen: bool = False) -> np.ndarray:
        """Convert all non-frozen parameters to a flat vector."""
    def as_nominal_vector(self, include_frozen: bool = False) -> np.ndarray:
        """Get the nominal values of parameters as a flat array."""
    def update_from_vector(self, vector: np.ndarray) -> None:
        """Update all non-frozen parameters from a flat vector.

    Args:
      vector: Flat array whose length equals the total size of non-frozen
        parameters.
    """
    def move_off_bounds(self, fraction: float = 0.05, at_bound_tol: float = 0.001) -> Self:
        """Call :meth:`Parameter.move_off_bound` on every non-frozen parameter,
    returning ``self`` so calls can be chained before
    :func:`optimize`."""
    def save_to_disk(self, path: str | pathlib.Path) -> None:
        """Save the parameter dictionary to disk (schema and data).

    Args:
      path: Path where the data will be saved.
    """
    @classmethod
    def load_from_disk(cls, path: str | pathlib.Path) -> ParameterDict:
        """Load parameter dictionary from disk (schema and data).

    Args:
      path: Path to the saved data.

    Returns:
      A new ParameterDict object.
    """
    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the bounds for all non-frozen parameters."""
    def reset(self) -> None:
        """Reset all parameters to their nominal values."""
    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample parameter values within bounds for non-frozen parameters.

    Args:
      rng: Optional numpy random generator. Uses default if None.
    """
    def randomize(self, rng: np.random.Generator | None = None) -> None:
        """Randomize parameter values for non-frozen parameters.

    Args:
      rng: Optional numpy random generator. Uses default if None.
    """
    def compare_parameters(self, init_params: np.ndarray, predicted_params: np.ndarray, measured_params: np.ndarray | None = None, sig_digits: int = 4) -> str:
        """Compare true and predicted parameter values.

    Args:
      init_params: Initial parameter values as a flat array.
      predicted_params: Predicted parameter values as a flat array.
      measured_params: True parameter values as a flat array.
      sig_digits: Number of significant digits to display.

    Returns:
      A formatted string with parameter comparison table.
    """
    def get_non_frozen_parameter_names(self) -> list[str]:
        """Get the names of all non-frozen parameters, expanding multi-dimensional ones."""
    def get_parameter_info(self) -> str:
        """Get information about all parameters in the dictionary.

    Returns:
      A formatted string with parameter information.
    """
