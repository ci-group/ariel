import mujoco
import numpy as np
from collections.abc import Callable, Mapping, Sequence
from mujoco.sysid._src import model_modifier as model_modifier, parameter as parameter, signal_modifier as signal_modifier, timeseries as timeseries
from mujoco.sysid._src.trajectory import ModelSequences as ModelSequences, SystemTrajectory as SystemTrajectory, sysid_rollout as sysid_rollout
from typing import TypeAlias

BuildModelFn: TypeAlias
CustomRolloutFn: TypeAlias = Callable[..., Sequence[SystemTrajectory]]
ModifyResidualFn: TypeAlias

def construct_ts_from_defaults(state_ts: timeseries.TimeSeries, pred_sensordata: timeseries.TimeSeries, measured_sensordata: timeseries.TimeSeries, enabled_observations: Sequence[tuple[str, timeseries.SignalType]] | None = None):
    """Assemble predicted observations to match the measured signal layout.

  For each enabled observation, copies the predicted values from either
  ``pred_sensordata`` (for MjSensor signals) or ``state_ts`` (for state
  signals like qpos/qvel/act) into a new array whose columns align with
  the measured data.

  Args:
    state_ts: Predicted state TimeSeries (time column already stripped).
    pred_sensordata: Raw predicted sensor TimeSeries from rollout.
    measured_sensordata: Measured sensor TimeSeries (defines the target layout).
    enabled_observations: Subset of observations to include.  If None, all
      observations in ``measured_sensordata`` are used.

  Returns:
    A ``(measured, predicted)`` tuple of TimeSeries with matching signal
    mappings, sliced to the enabled observations.
  """
def model_residual(x: np.ndarray, params: parameter.ParameterDict, build_model: Callable[[parameter.ParameterDict], mujoco.MjModel], traj_measured: Sequence[SystemTrajectory] | SystemTrajectory, modify_residual: ModifyResidualFn | None = None, custom_rollout: CustomRolloutFn | None = None, n_threads: int = ..., return_pred_all: bool = False, resample_true: bool = True, sensor_weights: Mapping[str, float] | None = None, enabled_observations: Sequence[tuple[str, timeseries.SignalType]] = ()):
    """Compute residuals for a single model against measured trajectories.

  Builds the model from *x*, rolls out each trajectory, and computes the
  weighted difference between predicted and measured sensor data.

  Args:
    x: Decision variable vector (flat, or 2-D for batched finite-difference).
    params: Parameter dictionary — updated in-place from *x*.
    build_model: ``(ParameterDict) -> MjModel`` factory.
    traj_measured: Ground-truth trajectory or sequence of trajectories.
    modify_residual: Optional custom residual callback (replaces the default
      resampling / differencing logic).
    custom_rollout: Optional replacement for :func:`sysid_rollout`.
    n_threads: Number of ``MjData`` scratch objects for parallel rollout.
    return_pred_all: If True, return full predicted/measured TimeSeries.
    resample_true: Whether to resample the measured data at simulation timesteps
      (ignored when *modify_residual* is provided).
    sensor_weights: Per-sensor weights for the weighted diff.
    enabled_observations: Subset of ``(name, SignalType)`` pairs to include.

  Returns:
    A 3-tuple ``(residuals, pred_sensordatas, measured_sensordatas)``.
  """
def build_residual_fn(**captured_kwargs):
    """Create a residual closure with pre-bound keyword arguments.

  Returns a function ``fn(x, params, **overrides)`` that calls
  :func:`residual` with the captured kwargs merged in.  This is the
  recommended way to construct the callable passed to :func:`optimize`.

  Example::

      residual_fn = build_residual_fn(
          models_sequences=seqs,
          signal_transform=transform,
      )
      opt_params, result = optimize(params, residual_fn)
  """
def residual(x: np.ndarray, params: parameter.ParameterDict, models_sequences: list[ModelSequences], build_model: BuildModelFn = ..., modify_residual: ModifyResidualFn | None = None, custom_rollout: CustomRolloutFn | None = None, n_threads: int = ..., return_pred_all: bool = False, resample_true: bool = True, sensor_weights: Mapping[str, float] | None = None, enabled_observations: Sequence[tuple[str, timeseries.SignalType]] = ()):
    """Top-level residual: iterate over all model-sequence groups.

  Calls :func:`model_residual` for every measured rollout in every
  :class:`ModelSequences` entry and collects the results.

  Args:
    x: Current parameter vector.
    params: Parameter dictionary with bounds and metadata.
    models_sequences: List of model-sequence groups.
    build_model: Callable to compile a model from parameters.
    modify_residual: Optional post-processing for the residual.
    custom_rollout: Optional custom rollout function.
    n_threads: Number of threads for parallel rollout.
    return_pred_all: Whether to return all predicted signals.
    resample_true: Whether to resample ground-truth to predicted times.
    sensor_weights: Per-sensor weighting for the residual.
    enabled_observations: Additional observation signals to include.

  Returns:
    A 3-tuple ``(residuals, preds, records)`` — lists with one entry per
    measured rollout across all groups.
  """
