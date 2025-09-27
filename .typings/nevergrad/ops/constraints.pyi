import nevergrad.common.typing as tp
from _typeshed import Incomplete
from nevergrad import callbacks as callbacks, optimizers as optimizers
from nevergrad.common import errors as errors
from nevergrad.parametrization import core as core

class Constraint(core.Operator):
    """Operator for applying a constraint on a Parameter

    Parameters
    ----------
    func: function
        the constraint function, taking the same arguments that the function to optimize.
        This constraint function must return a float (or a list/tuple/array of floats),
        positive if the constraint is not satisfied, null or negative otherwise.
    optimizer: str
        name of the optimizer to use for solving the constraint
    budget: int
        the budget to use for applying the constraint

    Example
    -------
    >>> constrained_parameter = Constraint(constraint_function)(parameter)
    >>> constrained_parameter.value  # value after trying to satisfy the constraint
    """
    _LAYER_LEVEL: Incomplete
    _func: Incomplete
    _opt_cls: Incomplete
    _budget: Incomplete
    _cache: tp.Any
    def __init__(self, func: tp.Callable[..., tp.Loss], optimizer: str = 'NGOpt', budget: int = 100) -> None: ...
    def _layered_del_value(self) -> None: ...
    def apply_constraint(self, parameter: core.Parameter) -> core.Parameter:
        """Find a new parameter that better satisfies the constraint"""
    def function(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Loss: ...
    def parameter(self) -> core.Parameter:
        """Returns a constraint-free parameter, for the optimization process"""
    def stopping_criterion(self, optimizer: tp.Any) -> bool:
        """Checks whether a solution was found
        This is used as stopping criterio callback
        """
    def _layered_get_value(self) -> tp.Any: ...
