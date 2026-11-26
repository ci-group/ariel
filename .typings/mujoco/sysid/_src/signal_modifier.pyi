import mujoco
import numpy as np
from collections.abc import Mapping
from mujoco.sysid._src import parameter as parameter, timeseries as timeseries

def get_sensor_indices(model: mujoco.MjModel, sensor_name: str | list[str], sort: bool = False) -> list[int]:
    """Get sensor indices from a sensor configuration dictionary.

  Args:
    model: MuJoCo model containing the sensors.
    sensor_name: sensor name or list of names to return the indices for.
    sort: If True, sort the indices before returning.

  Returns:
    A list of sensor indices.
  """
def apply_bias(ts: timeseries.TimeSeries, sensor_name: str, bias: parameter.Parameter) -> timeseries.TimeSeries:
    """Apply a bias to a sensor in a timeseries.

  Args:
    ts: Input timeseries.
    sensor_name: Name of the sensor to modify.
    bias: Parameter whose ``.value`` is added to the sensor columns.
  """
def apply_gain(ts: timeseries.TimeSeries, sensor_name: str, gain: parameter.Parameter) -> timeseries.TimeSeries:
    """Apply a gain to a sensor in a timeseries.

  Args:
    ts: Input timeseries.
    sensor_name: Name of the sensor to modify.
    gain: Parameter whose ``.value`` multiplies the sensor columns.
  """
def apply_delay(ts: timeseries.TimeSeries, sensor_name: str, delay: parameter.Parameter) -> timeseries.TimeSeries:
    """Apply a delay to a sensor in a timeseries.

  Args:
    ts: Input timeseries.
    sensor_name: Name of the sensor to delay.
    delay: Parameter whose ``.value`` is the delay in seconds.
  """
def apply_time_window(ts: timeseries.TimeSeries, min_t: float, max_t: float) -> timeseries.TimeSeries:
    """Select a subset of a timeseries within a time window.

  The selected timestamps, plus the max delay, can be sampled from ts_sample.

  Args:
    ts: The timeseries to window.
    min_t: The minimum time.
    max_t: The maximum time.

  Returns:
    A new timeseries with the timestamps windowed.
  """
def apply_delayed_ts_window(ts: timeseries.TimeSeries, ts_delayed: timeseries.TimeSeries, min_delay: float, max_delay: float) -> timeseries.TimeSeries:
    """Window a timeseries to fit within the bounds of a delayed series.

  Included timestamps must lay within the bounds of a timeseries that may be
  delayed between min_delay and max_delay.

  Args:
    ts: The timeseries to window.
    ts_delayed: The timeseries to use as the bounds.
    min_delay: The minimum delay. May be negative.
    max_delay: The maximum delay.

  Returns:
    A new timeseries with the timestamps windowed.
  """
def apply_resample_and_delay(ts: timeseries.TimeSeries, times: np.ndarray, default_delay: float, sensor_delays: dict[str, float] | None = None, predicted_data: bool = True) -> timeseries.TimeSeries:
    """Resample a timeseries and apply per-sensor delays.

  Args:
    ts: Input timeseries to resample.
    times: Target timestamps.
    default_delay: Default delay applied to all columns.
    sensor_delays: Optional per-sensor delay overrides.
    predicted_data: If True, negate delays (shift predicted to match measured).
  """
def prepare_sensor_weights(sensor_weights: Mapping[str, float] | np.ndarray, n_sensors: int, model: mujoco.MjModel) -> np.ndarray:
    """Prepare sensor weights array from a dict or numpy array.

  Args:
    sensor_weights: Mapping from sensor name to weight, or a flat array.
    n_sensors: Total number of sensor columns.
    model: MuJoCo model for resolving sensor names to indices.
  """
def weighted_diff(predicted_data: np.ndarray, measured_data: np.ndarray, model: mujoco.MjModel | None = None, sensor_weights: Mapping[str, float] | np.ndarray | None = None) -> np.ndarray:
    """Compute the weighted difference between measured and predicted data.

  Calculates ``measured_data - predicted_data``, optionally scaled by sensor
  weights.

  Args:
    predicted_data: The predicted data, of shape (n_timesteps, n_sensors).
    measured_data: The measured data, of shape (n_timesteps, n_sensors).
    model: Optional mujoco model. This argument is required if sensor_weights
      is not None.
    sensor_weights: An optional dict mapping sensor name to weight. Unspecified
      sensors are assumed to have a weight of 1.

  Returns:
    A numpy array of the weighted difference.
  """
def normalize_residual(residual: np.ndarray, measured_data: np.ndarray) -> np.ndarray:
    """Normalize the residual by the standard deviation of the measured data.

  Args:
    residual: Residual array, shape ``(n_timesteps, n_sensors)``.
    measured_data: Measured data array, same shape as *residual*.
  """
