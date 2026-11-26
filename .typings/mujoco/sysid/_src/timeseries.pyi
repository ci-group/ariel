import mujoco
import numpy as np
import pathlib
from _typeshed import Incomplete
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

class SignalType(Enum):
    MjSensor = 0
    CustomObs = 1
    MjStateQPos = 2
    MjStateQVel = 3
    MjStateAct = 4
    MjCtrl = 5
SignalMappingType: TypeAlias = dict[str, tuple[SignalType, np.ndarray]]

def encode_signal_mapping(signal_mapping: SignalMappingType) -> str:
    """Serialize a signal mapping to a JSON string.

  This avoids storing object arrays (which require ``allow_pickle=True`` to
  load) on disk. The mapping is plain data: enum values and integer indices.
  """
def decode_signal_mapping(encoded: str) -> SignalMappingType:
    """Inverse of :func:`encode_signal_mapping`."""

InterpolationMethod: Incomplete

@dataclass(frozen=True)
class TimeSeries:
    """A utility class for working with time-series data.

  Attributes:
    times: 1D array of timestamps.
    data: Array of signal data. The first axis corresponds to time.
    signal_mapping: Dict of tuples that maps the signal type and its signal
      fields in data
  """
    times: np.ndarray
    data: np.ndarray
    signal_mapping: SignalMappingType | None = ...
    @staticmethod
    def compute_all_sensor_mapping(model: mujoco.MjModel) -> SignalMappingType:
        """Computes mapping for all sensors in the model."""
    @staticmethod
    def compute_all_control_mapping(model: mujoco.MjModel) -> SignalMappingType:
        """Computes mapping for all controls (actuators) in the model."""
    @staticmethod
    def compute_all_state_mappings(model: mujoco.MjModel) -> tuple[SignalMappingType, SignalMappingType, SignalMappingType, SignalMappingType]:
        """Computes mappings for all state components (qpos, qvel, act) + ctrl."""
    @classmethod
    def from_custom_map(cls, times: np.ndarray, data: np.ndarray, signals: Sequence[str | tuple[str, int, SignalType]]) -> TimeSeries:
        """Construct a TimeSeries from custom data with explicit signal definitions.

    Use this when you have custom signal types (e.g., from a custom
    modify_residual function) that are not auto-resolved from a MuJoCo model.
    You must explicitly specify the signal names, widths, and types.

    Args:
      times: 1-D timestamp array of length N.
      data: 2-D array of shape ``(N, D)``.
      signals: Defines the layout of the columns in ``data``.

    Returns:
      A TimeSeries object with the constructed signal mapping.
    """
    @classmethod
    def from_names(cls, times: np.ndarray, data: np.ndarray, model: mujoco.MjModel, names: Sequence[str | tuple[str, SignalType]] | None = None) -> TimeSeries:
        """Construct a TimeSeries for observations from the model.

    This method automatically resolves signal names (sensors, qpos, qvel, act)
    from
    the MuJoCo model, determining their types and data layout. Use this for
    standard
    observation signals that are defined in the model.

    Args:
      times: 1-D timestamps of length N.
      data: 2-D array of shape (N, D).
      model: MuJoCo model used to auto-resolve signal names and types.
      names: Signal names to map. Can be strings or (name, SignalType) tuples.
        If None, maps ALL model sensors in sensor address order.

    Warning:
      When names=None, data columns MUST match the model's sensor layout
      (i.e., data[:, i] corresponds to model.sensordata[i] during simulation).
      If your data is in a different order, pass explicit names.

    Raises:
      ValueError: If MjCtrl signals are passed (use from_control_names).
    """
    @classmethod
    def from_control_names(cls, times: np.ndarray, data: np.ndarray, model: mujoco.MjModel, names: Sequence[str | tuple[str, SignalType]] | None = None) -> TimeSeries:
        """Construct a TimeSeries for control signals from the model.

    This method automatically resolves control/actuator names from the MuJoCo
    model,
    determining their layout. Use this for control signals (MjCtrl type).

    Args:
      times: 1-D timestamps of length N.
      data: 2-D array of shape (N, model.nu).
      model: MuJoCo model used to auto-resolve actuator names.
      names: Actuator names to map. If None, maps ALL actuators in order.

    Warning:
      When names=None, data columns MUST match actuator order in the model
      (i.e., data[:, i] corresponds to actuator i). Pass explicit names
      if your data is in a different order.
    """
    def get_indices(self, obs_name: str) -> tuple[SignalType, np.ndarray]:
        """Look up the signal type and column indices for a named observation.

    Args:
      obs_name: Name of the observation signal.
    """
    @classmethod
    def create(cls, times: np.ndarray, data: np.ndarray, signal_mapping: dict[str, tuple[SignalType, np.ndarray | list[int] | int]]) -> TimeSeries:
        """Construct a TimeSeries, normalizing index entries to ``np.ndarray``.

    Args:
      times: 1-D timestamp array.
      data: Data array with first axis corresponding to time.
      signal_mapping: Dict mapping signal names to ``(type, indices)`` tuples.
        Index entries are coerced to ``np.ndarray``.
    """
    @classmethod
    def slice_by_name(cls, ts: TimeSeries, enabled_sensors: list[str]) -> TimeSeries:
        """Return a new TimeSeries containing only the named signals.

    Columns are re-indexed so the resulting ``signal_mapping`` has contiguous
    indices starting from 0.

    Args:
      ts: The source TimeSeries.
      enabled_sensors: Names of signals to keep.
    """
    def __post_init__(self) -> None:
        """Validate the time series data after initialization.

    Raises:
      ValueError: If times is not 1D, if lengths don't match, if times
        is not strictly increasing, or if arrays are empty.
    """
    def __len__(self) -> int: ...
    def save_to_disk(self, path: str | pathlib.Path) -> None:
        """Save the time series data to disk.

    Args:
      path: Path where the data will be saved.
    """
    def save_to_csv(self, path: str | pathlib.Path) -> None:
        """Save the time series data to a CSV file.

    Args:
      path: Path where the CSV file will be written.
    """
    @classmethod
    def load_from_disk(cls, path: str | pathlib.Path) -> TimeSeries:
        """Load time series data from disk.

    Args:
      path: Path to the saved data.

    Returns:
      A new TimeSeries object.
    """
    def interpolate(self, t: float | np.ndarray, method: InterpolationMethod = 'linear') -> np.ndarray:
        """Interpolate data at specified time(s).

    This is the core interpolation function used by both get() and resample().

    Args:
      t: Time point(s) at which to interpolate data.
      method: Interpolation method to use.

    Returns:
      Interpolated data values.
    """
    def get(self, t: float | np.ndarray, method: InterpolationMethod = 'linear') -> tuple[np.ndarray, np.ndarray]:
        """Get interpolated data at specified time(s).

    This method is useful for querying data at specific timestamps without
    creating a new TimeSeries object.

    Args:
      t: Time point(s) at which to get data.
      method: Interpolation method to use.

    Returns:
      Tuple of (times, interpolated_data).
    """
    def resample(self, new_times: np.ndarray | None = None, target_dt: float | None = None, method: InterpolationMethod = 'linear') -> TimeSeries:
        """Resample the time series to new timestamps or a specific time interval.

    This method creates a new TimeSeries object with data interpolated at the
    specified timestamps.

    Args:
      new_times: Optional array of new timestamps. If provided, target_dt is
        ignored.
      target_dt: Optional time interval for regular resampling. Only used if
        new_times is None.
      method: Interpolation method to use.

    Returns:
      A new TimeSeries object with resampled data.

    Raises:
      ValueError: If neither new_times nor target_dt is provided, or if
        new_times is not strictly increasing.
    """
    def remove_from_beginning(self, time_to_remove_s: float) -> TimeSeries:
        """Remove time from the beginning of the time series.

    Args:
      time_to_remove_s: Time to remove from the beginning of the time series.

    Returns:
      A new TimeSeries object with the specified time removed.
    """
    def dt_statistics(self) -> dict[str, float]:
        """Calculate statistics about the time intervals.

    Returns:
      Dictionary with mean, median, std, min, and max of time intervals.

    Raises:
      ValueError: If there are fewer than two timestamps.
    """
