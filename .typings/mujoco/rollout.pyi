import mujoco
from _typeshed import Incomplete
from collections.abc import Sequence
from mujoco import _rollout as _rollout
from numpy import typing as npt

class Rollout:
    """Rollout object containing a thread pool for parallel rollouts."""
    nthread: Incomplete
    rollout_: Incomplete
    def __init__(self, *, nthread: int | None = None) -> None:
        """Construct a rollout object containing a thread pool for parallel rollouts.

    Args:
      nthread: Number of threads in pool.
        If zero, this pool is not started and rollouts run on the calling thread.
    """
    def __enter__(self): ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: types.TracebackType | None) -> None: ...
    def close(self) -> None: ...
    def rollout(self, model: mujoco.MjModel | Sequence[mujoco.MjModel], data: mujoco.MjData | Sequence[mujoco.MjData], initial_state: npt.ArrayLike, control: npt.ArrayLike | None = None, *, control_spec: int = ..., skip_checks: bool = False, nstep: int | None = None, initial_warmstart: npt.ArrayLike | None = None, state: npt.ArrayLike | None = None, sensordata: npt.ArrayLike | None = None, chunk_size: int | None = None):
        """Rolls out open-loop trajectories from initial states, get subsequent state and sensor values.

    Python wrapper for rollout.cc, see documentation therein.
    Infers nbatch and nstep.
    Tiles inputs with singleton dimensions.
    Allocates outputs if none are given.

    Args:
      model: An instance or length nbatch sequence of MjModel with the same size signature.
      data: Associated mjData instance or sequence of instances with length nthread.
      initial_state: Array of initial states from which to roll out trajectories.
        ([nbatch or 1] x nstate)
      control: Open-loop controls array to apply during the rollouts.
        ([nbatch or 1] x [nstep or 1] x ncontrol)
      control_spec: mjtState specification of control vectors.
      skip_checks: Whether to skip internal shape and type checks.
      nstep: Number of steps in rollouts (inferred if unspecified).
      initial_warmstart: Initial qfrc_warmstart array (optional).
        ([nbatch or 1] x nv)
      state: State output array (optional).
        (nbatch x nstep x nstate)
      sensordata: Sensor data output array (optional).
        (nbatch x nstep x nsensordata)
      chunk_size: Determines threadpool chunk size. If unspecified,
                  chunk_size = max(1, nbatch / (nthread * 10))

    Returns:
      state:
        State output array, (nbatch x nstep x nstate).
      sensordata:
        Sensor data output array, (nbatch x nstep x nsensordata).

    Raises:
      RuntimeError: rollout requested after thread pool shutdown.
      ValueError: bad shapes or sizes.
    """

persistent_rollout: Incomplete

def shutdown_persistent_pool() -> None:
    """Shutdown the persistent thread pool that is optionally created by rollout.

  This is called automatically interpreter shutdown, but can also be called manually.
  """
def rollout(model: mujoco.MjModel | Sequence[mujoco.MjModel], data: mujoco.MjData | Sequence[mujoco.MjData], initial_state: npt.ArrayLike, control: npt.ArrayLike | None = None, *, control_spec: int = ..., skip_checks: bool = False, nstep: int | None = None, initial_warmstart: npt.ArrayLike | None = None, state: npt.ArrayLike | None = None, sensordata: npt.ArrayLike | None = None, chunk_size: int | None = None, persistent_pool: bool = False):
    """Rolls out open-loop trajectories from initial states, get subsequent states and sensor values.

  Python wrapper for rollout.cc, see documentation therein.
  Infers nbatch and nstep.
  Tiles inputs with singleton dimensions.
  Allocates outputs if none are given.

  Args:
    model: An instance or length nbatch sequence of MjModel with the same size signature.
    data: Associated mjData instance or sequence of instances with length nthread.
    initial_state: Array of initial states from which to roll out trajectories.
      ([nbatch or 1] x nstate)
    control: Open-loop controls array to apply during the rollouts.
      ([nbatch or 1] x [nstep or 1] x ncontrol)
    control_spec: mjtState specification of control vectors.
    skip_checks: Whether to skip internal shape and type checks.
    nstep: Number of steps in rollouts (inferred if unspecified).
    initial_warmstart: Initial qfrc_warmstart array (optional).
      ([nbatch or 1] x nv)
    state: State output array (optional).
      (nbatch x nstep x nstate)
    sensordata: Sensor data output array (optional).
      (nbatch x nstep x nsensordata)
    chunk_size: Determines threadpool chunk size. If unspecified,
                chunk_size = max(1, nbatch / (nthread * 10))
    persistent_pool: Determines if a persistent thread pool is created or reused.

  Returns:
    state:
      State output array, (nbatch x nstep x nstate).
    sensordata:
      Sensor data output array, (nbatch x nstep x nsensordata).

  Raises:
    ValueError: bad shapes or sizes.
  """
def _check_must_be_numeric(**kwargs) -> None: ...
def _check_number_of_dimensions(ndim, **kwargs) -> None: ...
def _check_trailing_dimension(dim, **kwargs) -> None: ...
def _ensure_2d(arg): ...
def _ensure_3d(arg): ...
def _infer_dimension(dim, value, **kwargs):
    """Infers dimension `dim` given guess `value` from set of arrays.

  Args:
    dim: Dimension to be inferred.
    value: Initial guess of inferred value (1: unknown).
    **kwargs: List of arrays which should all have the same size (or 1) along
      dimension dim.

  Returns:
    Inferred dimension.

  Raises:
    ValueError: If mismatch between array shapes or initial guess.
  """
def _tile_if_required(array, dim0, dim1: Incomplete | None = None): ...
