import contextlib
import nevergrad.common.typing as tp
import numpy as np
from . import _datalayers as _datalayers, _layering as _layering, container as container, core as core, data as pdata, transforms as trans
from _typeshed import Incomplete

def flatten(parameter: core.Parameter, with_containers: bool = True, order: int = 0) -> tp.List[tp.Tuple[str, core.Parameter]]:
    '''List all the instances involved as parameter (not as subparameter/
    endogeneous parameter)

    Parameter
    ---------
    parameter: Parameter
        the parameter to inspect
    with_containers: bool
        returns all the sub-parameter if True, otherwise only non-pure containers (i.e. only Data and Choice)
    order: int
        order of model/internal parameters to extract. With 0, no model/internal parameters is
        extracted, with 1, only 1st order are extracted, with 2, so model/internal parameters and
        their own model/internal parameters etc. Order 1 subparameters/endogeneous parameters
        include sigma for instance.


    Returns
    -------
    list
        a list of all (name, parameter) implied in this parameter, i.e all choices, items of dict
        and tuples etc (except if only_data=True). Names have a format "<index>.<key>" for a tuple
        containing dicts containing data for instance. Supbaramaters have # in their names.
        The parameters are sorted in the same way they would appear in the standard data.

    '''
def list_data(parameter: core.Parameter) -> tp.List[pdata.Data]:
    """List all the Data instances involved as parameter (not as subparameter/
    endogeneous parameter) in the order they are defined in the standardized data.

    Parameter
    ---------
    parameter: Parameter
        the parameter to inspect
    """

class ParameterInfo(tp.NamedTuple):
    """Information about a parameter

    Attributes
    ----------
    deterministic: bool
        whether the function equipped with its instrumentation is deterministic.
        Can be false if the function is not deterministic or if the instrumentation
        contains a softmax.
    continuous: bool
        whether the domain is entirely continuous.
    ordered: bool
        whether all domains and subdomains are ordered.
    arity: int
        number of options for discrete parameters (-1 if continuous)
    """
    deterministic: bool
    continuous: bool
    ordered: bool
    arity: int

def analyze(parameter: core.Parameter) -> ParameterInfo:
    """Analyzes a parameter to provide useful information about it"""
@contextlib.contextmanager
def deterministic_sampling(parameter: core.Parameter) -> tp.Iterator[None]:
    '''Temporarily change the behavior of a Parameter to become deterministic

    Parameters
    ----------
    parameter: Parameter
        the parameter which must behave deterministically during the "with" context
    '''
def _fully_bounded_layers(data: pdata.Data) -> tp.List[_datalayers.BoundLayer]:
    """Extract fully bounded layers of a Data parameter"""

class Normalizer:
    """Hacky way to sample in the space defined by the parametrization.
    Given an vector of values between 0 and 1,
    the transform method samples in the bounds if provided,
    or using the provided function otherwise.
    This is used for samplers.
    Code of parametrization and/or this helper should definitely be
    updated to make it simpler and more robust
    """
    reference: Incomplete
    _ref_arrays: Incomplete
    working: bool
    _only_sampling: Incomplete
    unbounded_transform: Incomplete
    fully_bounded: Incomplete
    def __init__(self, reference: core.Parameter, unbounded_transform: tp.Optional[trans.Transform] = None, only_sampling: bool = False) -> None: ...
    def _warn(self) -> None: ...
    def backward(self, x: tp.ArrayLike) -> np.ndarray:
        """Transform from [0, 1] to standardized space"""
    def forward(self, x: tp.ArrayLike) -> np.ndarray:
        """Transform from standardized space to [0, 1]"""
    def _apply(self, x: tp.ArrayLike, forward: bool = True) -> np.ndarray: ...
    def _apply_unsafe(self, x: np.ndarray, forward: bool = True) -> None: ...
