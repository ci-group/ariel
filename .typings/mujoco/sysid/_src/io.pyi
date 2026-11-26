import os
import scipy.optimize as scipy_optimize
from collections.abc import Sequence
from mujoco.sysid._src import parameter as parameter
from mujoco.sysid._src.optimize import calculate_intervals as calculate_intervals
from mujoco.sysid._src.trajectory import ModelSequences as ModelSequences

def save_results(experiment_results_folder: str | os.PathLike[str], models_sequences: Sequence[ModelSequences], initial_params: parameter.ParameterDict, opt_params: parameter.ParameterDict, opt_result: scipy_optimize.OptimizeResult, residual_fn):
    """Save optimization results and confidence intervals to disk.

  Args:
    experiment_results_folder: Directory where results are written.
    models_sequences: Model/sequence groups; identified XMLs are saved here.
    initial_params: Parameters before optimization.
    opt_params: Parameters after optimization.
    opt_result: Scipy OptimizeResult from the optimizer.
    residual_fn: Residual function used to compute confidence intervals.
  """
