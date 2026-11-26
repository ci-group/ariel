import mujoco
import numpy as np
from _typeshed import Incomplete
from collections.abc import Mapping
from mujoco.sysid._src import parameter as parameter, signal_modifier as signal_modifier, timeseries as timeseries

class SignalTransform:
    '''Declarative signal transformation replacing boilerplate modify_residual callbacks.

  Usage::

    transform = SignalTransform()
    transform.delay("*_pos", params["delay_pos"])
    transform.delay("*_torque", params["delay_torque"])
    transform.gain("*_torque", params["torque_scale"], target="predicted")
    transform.enable_sensors(cfg.sensors_enabled)

  The ``apply`` method has the same signature as ``ModifyResidualFn`` and can
  be passed directly to ``build_residual_fn(signal_transform=transform)``.
  '''
    normalize: Incomplete
    def __init__(self, normalize: bool = True) -> None: ...
    def delay(self, pattern: str, param: parameter.Parameter) -> None:
        """Register a delay for sensors matching *pattern* (fnmatch).

    Args:
      pattern: fnmatch pattern matched against sensor names.
      param: Parameter whose ``.value`` is the delay in seconds.
    """
    def gain(self, pattern: str, param: parameter.Parameter, target: str = 'both') -> None:
        '''Register a multiplicative gain for sensors matching *pattern*.

    Args:
      pattern: fnmatch pattern matched against sensor names.
      param: Parameter whose ``.value`` is the gain factor.
      target: One of ``"predicted"``, ``"measured"``, or ``"both"``.
    '''
    def bias(self, pattern: str, param: parameter.Parameter, target: str = 'both') -> None:
        '''Register an additive bias for sensors matching *pattern*.

    Args:
      pattern: fnmatch pattern matched against sensor names.
      param: Parameter whose ``.value`` is the additive bias.
      target: One of ``"predicted"``, ``"measured"``, or ``"both"``.
    '''
    def enable_sensors(self, sensor_names: list[str]) -> None:
        """Only include these sensors in the returned residual/timeseries.

    Args:
      sensor_names: Sensor names to keep in the output.
    """
    def set_sensor_weights(self, weights: Mapping[str, float]) -> None:
        """Set per-sensor weights for the weighted diff.

    Args:
      weights: Mapping from sensor name to weight.
    """
    def apply(self, params: parameter.ParameterDict, sensordata_predicted: timeseries.TimeSeries, sensordata_measured: timeseries.TimeSeries, model: mujoco.MjModel, return_pred_all: bool, state: np.ndarray | None = None, sensor_weights: Mapping[str, float] | None = None) -> tuple[np.ndarray, timeseries.TimeSeries, timeseries.TimeSeries]:
        """Apply all registered transforms and compute the residual.

    Signature matches :data:`ModifyResidualFn` so this method can be passed
    directly as ``modify_residual`` to :func:`model_residual`.

    Pipeline: window measured data, resample + delay predicted data, apply
    gains/biases, weighted diff, normalise, slice to enabled sensors.

    Args:
      params: Current parameter dictionary.
      sensordata_predicted: Predicted sensor TimeSeries from rollout.
      sensordata_measured: Measured sensor TimeSeries (ground truth).
      model: MuJoCo model for sensor metadata lookup.
      return_pred_all: Whether to return all predicted signals.
      state: Unused; part of the ModifyResidualFn signature.
      sensor_weights: Per-sensor weighting for the residual.

    Returns:
      ``(residual_array, predicted_ts, measured_ts)`` — the residual matrix
      and the (possibly sliced) predicted/measured TimeSeries.
    """
