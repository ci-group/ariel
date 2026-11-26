import dataclasses
import mujoco
import numpy as np
import pathlib
from _typeshed import Incomplete
from collections.abc import Sequence
from mujoco.sysid._src import timeseries as timeseries

@dataclasses.dataclass(frozen=True)
class SystemTrajectory:
    """Encapsulates a trajectory rolled out from a system.

  Attributes:
    model: MuJoCo model used to simulate the trajectory.
    control: A TimeSeries instance containing control signals.
    sensordata: A TimeSeries instance containing sensor data.
    initial_state: Initial state of the simulation. Shape (n_state,).
    state: Simulation states over time. Shape (n_steps, n_state). Optional for
      real robot trajectories.
  """
    model: mujoco.MjModel
    control: timeseries.TimeSeries
    sensordata: timeseries.TimeSeries
    initial_state: np.ndarray
    state: timeseries.TimeSeries | None
    def replace(self, **kwargs) -> SystemTrajectory:
        """Return a copy with the specified fields replaced."""
    def get_sensordata_slice(self, sensor: str = 'joint_pos') -> np.ndarray:
        '''Extract contiguous sensor columns by type.

    Args:
      sensor: One of ``"joint_pos"``, ``"joint_vel"``, or ``"joint_torque"``.

    Returns:
      2-D array of shape ``(n_steps, total_sensor_dim)``.
    '''
    @property
    def sensordim(self) -> int:
        """Total number of scalar sensor outputs in the model."""
    def __len__(self) -> int:
        """Number of time steps in the trajectory."""
    def save_to_disk(self, path: pathlib.Path) -> None:
        """Save this trajectory to disk as a compressed NumPy archive."""
    @classmethod
    def load_from_disk(cls, path: pathlib.Path, model: mujoco.MjModel, allow_missing_sensors: bool = False) -> SystemTrajectory:
        """Load a trajectory from a compressed NumPy archive."""
    def check_compatible(self, allow_missing_sensors: bool = False) -> None:
        """Validate that data dimensions match the model.

    Checks sensor, control, state, and initial-state dimensions.

    Args:
      allow_missing_sensors: If True, a sensor dimension mismatch is logged as a
        warning instead of raising.
    """
    def split(self, chunk_size: int) -> list[SystemTrajectory]:
        """Split into consecutive non-overlapping chunks of *chunk_size* steps.

    Incomplete trailing steps are discarded.  Requires ``state`` to be set
    (needed to extract the initial state for each chunk).

    Args:
      chunk_size: Number of time steps per chunk.

    Returns:
      A list of SystemTrajectory chunks.
    """
    def render(self, height: int = 240, width: int = 320, camera: str | int = -1, fps: int = 30) -> list[np.ndarray]:
        """Render this trajectory to a list of RGB frames.

    Requires ``state`` to be set.  Delegates to
    :func:`~mujoco_sysid._src.plotting.render_rollout`.

    Args:
      height: Height of the rendered frames.
      width: Width of the rendered frames.
      camera: Camera index or name.
      fps: Frames per second.

    Returns:
      A list of RGB frames as numpy arrays.
    """

def create_initial_state(model: mujoco.MjModel, qpos: np.ndarray, qvel: np.ndarray | None = None, act: np.ndarray | None = None, qpos_names: Sequence[str] | None = None, qvel_names: Sequence[str] | None = None, act_names: Sequence[str] | None = None) -> np.ndarray:
    """Build a ``mjSTATE_FULLPHYSICS`` initial-state vector from components.

  Args:
    model: MuJoCo model.
    qpos: Joint positions, shape ``(nq,)``.
    qvel: Joint velocities, shape ``(nv,)``.  Defaults to zero.
    act: Actuator activations, shape ``(na,)``.  Defaults to zero.
    qpos_names: Names to map elements of qpos to specific MuJoCo states.
      If None, assumes qpos is in MuJoCo's order.
    qvel_names: Names to map elements of qvel to specific MuJoCo states.
      If None, assumes qvel is in MuJoCo's order.
    act_names: Actuator names to map elements of act to specific MuJoCo
      actuators. If None, assumes act is in MuJoCo's order.

  Returns:
    Flat state vector suitable for ``mujoco.rollout``.
  """

class ModelSequences:
    """A model spec paired with one or more measured trajectory sequences.

  Groups a single ``MjSpec`` (the model to be identified) with the
  corresponding measured data so that the residual pipeline can iterate
  over all sequences for that model.

  Args:
    name: Identifier for this model group (used for file-naming on save).
    spec: MjSpec that will be recompiled with candidate parameters.
    sequence_name: Name(s) identifying each measured sequence.
    initial_state: Initial state(s) for each sequence.
    control: Measured control TimeSeries for each sequence.
    sensordata: Measured sensor TimeSeries for each sequence.
    allow_missing_sensors: Passed through to
      :meth:`SystemTrajectory.check_compatible`.
  """
    name: Incomplete
    spec: Incomplete
    allow_missing_sensors: Incomplete
    gt_model: Incomplete
    sequence_name: list[str]
    initial_state: list[np.ndarray]
    control: list[timeseries.TimeSeries]
    sensordata: list[timeseries.TimeSeries]
    measured_rollout: list[SystemTrajectory]
    def __init__(self, name: str, spec: mujoco.MjSpec, sequence_name: str | Sequence[str], initial_state: np.ndarray | Sequence[np.ndarray], control: timeseries.TimeSeries | Sequence[timeseries.TimeSeries], sensordata: timeseries.TimeSeries | Sequence[timeseries.TimeSeries], allow_missing_sensors: bool = False) -> None: ...
    def __getitem__(self, key): ...

def timeseries2array(control_signal: timeseries.TimeSeries | Sequence[timeseries.TimeSeries]) -> tuple[np.ndarray, np.ndarray]:
    """Convert control TimeSeries to stacked arrays, dropping the last step.

  Args:
    control_signal: Control TimeSeries or sequence of TimeSeries.

  Returns:
    ``(control_array, control_times)`` with the last time step removed.
  """
def sequence2array(initial_states: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
    """Stack a sequence of initial-state vectors into a single array.

  Args:
    initial_states: Single state array or sequence of state arrays.
  """
def arrays2traj(models: mujoco.MjModel | Sequence[mujoco.MjModel], initial_states: np.ndarray | Sequence[np.ndarray], control: np.ndarray, control_times: np.ndarray, state: np.ndarray, sensordata: np.ndarray, signal_mapping: timeseries.SignalMappingType, state_mapping: timeseries.SignalMappingType, ctrl_mapping: timeseries.SignalMappingType) -> Sequence[SystemTrajectory]:
    """Convert raw rollout arrays into a list of SystemTrajectory objects.

  Args:
    models: Single model or sequence of models (one per batch element).
    initial_states: Initial state array(s).
    control: Control array, shape ``(nbatch, nsteps, nu)``.
    control_times: Control timestamps, shape ``(nbatch, nsteps)``.
    state: State array, shape ``(nbatch, nsteps, nstate)``.
    sensordata: Sensor data array, shape ``(nbatch, nsteps, nsensordata)``.
    signal_mapping: Signal mapping for sensor data.
    state_mapping: Signal mapping for state data.
    ctrl_mapping: Signal mapping for control data.
  """
def sysid_rollout(models: mujoco.MjModel | Sequence[mujoco.MjModel], datas: mujoco.MjData | Sequence[mujoco.MjData], control_signal: Sequence[timeseries.TimeSeries] | timeseries.TimeSeries, initial_states: np.ndarray | Sequence[np.ndarray], rollout_signal_mapping: timeseries.SignalMappingType | None = None, rollout_state_mapping: timeseries.SignalMappingType | None = None, ctrl_mapping: timeseries.SignalMappingType | None = None) -> Sequence[SystemTrajectory]:
    """Rollout trajectories in parallel for the given models and controls.

  Args:
    models: MuJoCo model or sequence of models.
    datas: MuJoCo data or sequence of data.
    control_signal: Control signals as TimeSeries or sequence of TimeSeries.
    initial_states: Initial states of the simulation. Shape (n_state,) or
      (n_batch, n_state).
    rollout_signal_mapping: Optional signal mapping for sensordata.
    rollout_state_mapping: Optional signal mapping for state.
    ctrl_mapping: Optional signal mapping for controls.

  Returns:
    Sequence of SystemTrajectory instances containing the simulation results.
  """
