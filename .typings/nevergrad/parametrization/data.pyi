import nevergrad.common.typing as tp
import numpy as np
from . import _layering as _layering, core as core, utils as utils
from .container import Dict as Dict
from _typeshed import Incomplete
from nevergrad.common import errors as errors

D = tp.TypeVar('D', bound='Data')
P = tp.TypeVar('P', bound=core.Parameter)

def _param_string(parameters: Dict) -> str:
    """Hacky helper for nice-visualizatioon"""

class Data(core.Parameter):
    """Array/scalar parameter

    Parameters
    ----------
    init: np.ndarray, or None
        initial value of the array (defaults to 0, with a provided shape)
    shape: tuple of ints, or None
        shape of the array, to be provided iff init is not provided
    lower: array, float or None
        minimum value
    upper: array, float or None
        maximum value
    mutable_sigma: bool
        whether the mutation standard deviation must mutate as well (for mutation based algorithms)

    Note
    ----
    - More specific behaviors can be obtained throught the following methods:
      set_bounds, set_mutation, set_integer_casting
    - if both lower and upper bounds are provided, sigma will be adapted so that the range spans 6 sigma.
      Also, if init is not provided, it will be set to the middle value.
    """
    _value: Incomplete
    parameters: Incomplete
    def __init__(self, *, init: tp.Optional[tp.ArrayLike] = None, shape: tp.Optional[tp.Tuple[int, ...]] = None, lower: tp.BoundValue = None, upper: tp.BoundValue = None, mutable_sigma: bool = False) -> None: ...
    @property
    def bounds(self) -> tp.Tuple[tp.Optional[np.ndarray], tp.Optional[np.ndarray]]:
        """Estimate of the bounds (None if unbounded)

        Note
        ----
        This may be inaccurate (WIP)
        """
    @property
    def dimension(self) -> int: ...
    def _get_name(self) -> str: ...
    @property
    def sigma(self) -> Data:
        """Value for the standard deviation used to mutate the parameter"""
    def _layered_sample(self) -> D: ...
    value: Incomplete
    def set_bounds(self, lower: tp.BoundValue = None, upper: tp.BoundValue = None, method: str = 'bouncing', full_range_sampling: tp.Optional[bool] = None) -> D:
        '''Bounds all real values into [lower, upper] using a provided method

        Parameters
        ----------
        lower: array, float or None
            minimum value
        upper: array, float or None
            maximum value
        method: str
            One of the following choices:

            - "bouncing": bounce on border (at most once). This is a variant of clipping,
               avoiding bounds over-samping (default).
            - "clipping": clips the values inside the bounds. This is efficient but leads
              to over-sampling on the bounds.
            - "constraint": adds a constraint (see register_cheap_constraint) which leads to rejecting mutations
              reaching beyond the bounds. This avoids oversampling the boundaries, but can be inefficient in large
              dimension.
            - "arctan": maps the space [lower, upper] to to all [-inf, inf] using arctan transform. This is efficient
              but it completely reshapes the space (a mutation in the center of the space will be larger than a mutation
              close to the bounds), and reaching the bounds is equivalent to reaching the infinity.
            - "tanh": same as "arctan", but with a "tanh" transform. "tanh" saturating much faster than "arctan", it can lead
              to unexpected behaviors.
        full_range_sampling: Optional bool
            Changes the default behavior of the "sample" method (aka creating a child and mutating it from the current instance)
            or the sampling optimizers, to creating a child with a value sampled uniformly (or log-uniformly) within
            the while range of the bounds. The "sample" method is used by some algorithms to create an initial population.
            This is activated by default if both bounds are provided.

        Notes
        -----
        - "tanh" reaches the boundaries really quickly, while "arctan" is much softer
        - only "clipping" accepts partial bounds (None values)
        '''
    def set_mutation(self, sigma: tp.Optional[tp.Union[float, core.Parameter]] = None, exponent: tp.Optional[float] = None) -> D:
        '''Output will be cast to integer(s) through deterministic rounding.

        Parameters
        ----------
        sigma: Array/Log or float
            The standard deviation of the mutation. If a Parameter is provided, it will replace the current
            value. If a float is provided, it will either replace a previous float value, or update the value
            of the Parameter.
        exponent: float
            exponent for the logarithmic mode. With the default sigma=1, using exponent=2 will perform
            x2 or /2 "on average" on the value at each mutation.

        Returns
        -------
        self
        '''
    def set_integer_casting(self) -> D:
        """Output will be cast to integer(s) through deterministic rounding.

        Returns
        -------
        self

        Note
        ----
        Using integer casting makes the parameter discrete which can make the optimization more
        difficult. It is especially ill-advised to use this with a range smaller than 10, or
        a sigma lower than 1. In those cases, you should rather use a TransitionChoice instead.
        """
    @property
    def integer(self) -> bool: ...
    def _internal_set_standardized_data(self, data: np.ndarray, reference: D) -> None: ...
    def _internal_get_standardized_data(self, reference: D) -> np.ndarray: ...
    def _to_reduced_space(self, value: tp.Optional[np.ndarray] = None) -> np.ndarray:
        """Converts array with appropriate shapes to reduced (uncentered) space
        by applying log scaling and sigma scaling
        """
    def _layered_recombine(self, *others: D) -> None: ...
    def copy(self) -> D: ...
    def _layered_set_value(self, value: np.ndarray) -> None: ...
    def _layered_get_value(self) -> np.ndarray: ...
    def _new_with_data_layer(self, name: str, *args: tp.Any, **kwargs: tp.Any) -> D: ...
    def __mod__(self, module: tp.Any) -> D: ...
    def __rpow__(self, base: float) -> D: ...
    def __add__(self, offset: tp.Any) -> D: ...
    def __sub__(self, offset: tp.Any) -> D: ...
    def __radd__(self, offset: tp.Any) -> D: ...
    def __mul__(self, value: tp.Any) -> D: ...
    def __rmul__(self, value: tp.Any) -> D: ...
    def __truediv__(self, value: tp.Any) -> D: ...
    def __rtruediv__(self, value: tp.Any) -> D: ...
    def __pow__(self, power: float) -> D: ...
    def __neg__(self) -> D: ...

def _fix_legacy(parameter: Data) -> None:
    '''Ugly hack for keeping the legacy behaviors with considers bounds always after the exponent
    and can still sample "before" the exponent (log-uniform).
    '''

class Array(Data):
    value: core.ValueProperty[tp.ArrayLike, np.ndarray]

class Scalar(Data):
    """Parameter representing a scalar.

    Parameters
    ----------
    init: optional float
        initial value of the scalar (defaults to 0.0 if both bounds are not provided)
    lower: optional float
        minimum value if any
    upper: optional float
        maximum value if any
    mutable_sigma: bool
        whether the mutation standard deviation must mutate as well (for mutation based algorithms)

    Notes
    -----
    - by default, this is an unbounded scalar with Gaussian mutations.
    - if both lower and upper bounds are provided, sigma will be adapted so that the range spans 6 sigma.
      Also, if init is not provided, it will be set to the middle value.
    - More specific behaviors can be obtained throught the following methods:
      :code:`set_bounds`, :code:`set_mutation`, :code:`set_integer_casting`
    """
    value: core.ValueProperty[float, float]
    def __init__(self, init: tp.Optional[float] = None, *, lower: tp.Optional[float] = None, upper: tp.Optional[float] = None, mutable_sigma: bool = True) -> None: ...

class Log(Scalar):
    """Parameter representing a positive variable, mutated by Gaussian mutation in log-scale.

    Parameters
    ----------
    init: float or None
        initial value of the variable. If not provided, it is set to the middle of lower and upper in log space
    exponent: float or None
        exponent for the log mutation: an exponent of 2.0 will lead to mutations by factors between around 0.5 and 2.0
        By default, it is set to either 2.0, or if the parameter is completely bounded to a factor so that bounds are
        at 3 sigma in the transformed space.
    lower: float or None
        minimum value if any (> 0)
    upper: float or None
        maximum value if any
    mutable_sigma: bool
        whether the mutation standard deviation must mutate as well (for mutation based algorithms)

    Note
    ----
    This class is only a wrapper over :code:`Scalar`.
    """
    def __init__(self, *, init: tp.Optional[float] = None, exponent: tp.Optional[float] = None, lower: tp.Optional[float] = None, upper: tp.Optional[float] = None, mutable_sigma: bool = False) -> None: ...
