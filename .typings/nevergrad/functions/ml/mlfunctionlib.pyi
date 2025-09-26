import numpy as np
import typing as tp
from ..base import ExperimentFunction as ExperimentFunction
from _typeshed import Incomplete
from nevergrad.parametrization import parameter as p

class MLTuning(ExperimentFunction):
    '''Class for generating ML hyperparameter tuning problems.
    We propose different possible regressors and different dimensionalities.
    In each case, Nevergrad will optimize the parameters of a scikit learning.

    Parameters
    ----------
    regressor: str
        type of function we can use for doing the regression. Can be "mlp", "decision_tree", "decision_tree_depth",
        "keras_dense_nn", "any".
        "any" means that the regressor has one more parameter which is a discrete choice among sklearn possibilities.
    data_dimension: int
        dimension of the data we generate. None if not an artificial dataset.
    dataset: str
        type of dataset; can be diabetes, boston, artificial, artificialcoos, artificialsquare.
    overfitter: bool
        if we want the evaluation to be the same as during the optimization run. This means that instead
        of train/valid/error, we have train/valid/valid. This is for research purpose, when we want to check if an algorithm
        is particularly good or particularly bad because it fails to minimize the validation loss or because it overfits.

    '''
    def _ml_parametrization(self, depth: int, criterion: str, min_samples_split: float, solver: str, activation: str, alpha: float, learning_rate: str, regressor: str, noise_free: bool) -> float: ...
    regressor: Incomplete
    data_dimension: Incomplete
    dataset: Incomplete
    overfitter: Incomplete
    name: Incomplete
    num_data: int
    _cross_val_num: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_train_cv: list[tp.Any]
    X_valid_cv: list[tp.Any]
    y_train_cv: list[tp.Any]
    y_valid_cv: list[tp.Any]
    _evalparams: Incomplete
    def __init__(self, regressor: str, data_dimension: int | None = None, dataset: str = 'artificial', overfitter: bool = False) -> None: ...
    def evaluation_function(self, *recommendations: p.Parameter) -> float: ...
    def make_dataset(self, data_dimension: int | None, dataset: str) -> None: ...
