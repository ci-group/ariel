import nevergrad.common.typing as tp
import numpy as np
from . import core as core, utils as utils
from _typeshed import Incomplete

D = tp.TypeVar('D', bound='Container')

class Container(core.Parameter):
    """Parameter which can hold other parameters.
    This abstract implementation is based on a dictionary.

    Parameters
    ----------
    **parameters: Any
        the objects or Parameter which will provide values for the dict

    Note
    ----
    This is the base structure for all container Parameters, and it is
    used to hold the internal/model parameters for all Parameter classes.
    """
    _subobjects: Incomplete
    _content: tp.Dict[tp.Any, core.Parameter]
    _sizes: tp.Optional[tp.Dict[str, int]]
    _ignore_in_repr: tp.Dict[str, str]
    def __init__(self, **parameters: tp.Any) -> None: ...
    @property
    def dimension(self) -> int: ...
    def _sanity_check(self, parameters: tp.List[core.Parameter]) -> None:
        """Check that all parameters are different"""
    def __getitem__(self, name: tp.Any) -> core.Parameter: ...
    def __setitem__(self, name: tp.Any, value: tp.Any) -> None: ...
    def __len__(self) -> int: ...
    def _get_parameters_str(self) -> str: ...
    def _get_name(self) -> str: ...
    def get_value_hash(self) -> tp.Hashable: ...
    def _internal_get_standardized_data(self, reference: D) -> np.ndarray: ...
    def _internal_set_standardized_data(self, data: np.ndarray, reference: D) -> None: ...
    def _layered_sample(self) -> D: ...
    def _layered_recombine(self, *others: D) -> None: ...

class Dict(Container):
    """Dictionary-valued parameter. This Parameter can contain other Parameters,
    its value is a dict, with keys the ones provided as input, and corresponding values are
    either directly the provided values if they are not Parameter instances, or the value of those
    Parameters. It also implements a getter to access the Parameters directly if need be.

    Parameters
    ----------
    **parameters: Any
        the objects or Parameter which will provide values for the dict

    Note
    ----
    This is the base structure for all container Parameters, and it is
    used to hold the internal/model parameters for all Parameter classes.
    """
    value: core.ValueProperty[tp.Dict[str, tp.Any], tp.Dict[str, tp.Any]]
    def __iter__(self) -> tp.Iterator[str]: ...
    def keys(self) -> tp.KeysView[str]: ...
    def items(self) -> tp.ItemsView[str, core.Parameter]: ...
    def values(self) -> tp.ValuesView[core.Parameter]: ...
    def _layered_get_value(self) -> tp.Dict[str, tp.Any]: ...
    def _layered_set_value(self, value: tp.Dict[str, tp.Any]) -> None: ...
    def _get_parameters_str(self) -> str: ...

class Tuple(Container):
    """Tuple-valued parameter. This Parameter can contain other Parameters,
    its value is tuple which values are either directly the provided values
    if they are not Parameter instances, or the value of those Parameters.
    It also implements a getter to access the Parameters directly if need be.

    Parameters
    ----------
    **parameters: Any
        the objects or Parameter which will provide values for the tuple

    Note
    ----
    This is the base structure for all container Parameters, and it is
    used to hold the subparameters for all Parameter classes.
    """
    def __init__(self, *parameters: tp.Any) -> None: ...
    def _get_parameters_str(self) -> str: ...
    def __iter__(self) -> tp.Iterator[core.Parameter]: ...
    value: core.ValueProperty[tp.Tuple[tp.Any, ...], tp.Tuple[tp.Any, ...]]
    def _layered_get_value(self) -> tp.Tuple[tp.Any, ...]: ...
    def _layered_set_value(self, value: tp.Tuple[tp.Any, ...]) -> None: ...

class Instrumentation(Tuple):
    '''Container of parameters available through `args` and `kwargs` attributes.
    The parameter provided as input are used to provide values for
    an `arg` tuple and a `kwargs` dict.
    `value` attribue returns `(args, kwargs)`, but each can be independantly
    accessed through the `args` and `kwargs` properties.

    Parameters
    ----------
    *args
         values or Parameters to be used to fill the tuple of args
    *kwargs
         values or Parameters to be used to fill the dict of kwargs

    Note
    ----
    When used in conjonction with the "minimize" method of an optimizer,
    functions call use `func(*param.args, **param.kwargs)` instead of
    `func(param.value)`. This is for simplifying the parametrization of
    multiparameter functions.
    '''
    def __init__(self, *args: tp.Any, **kwargs: tp.Any) -> None: ...
    @property
    def args(self) -> tp.Tuple[tp.Any, ...]: ...
    @property
    def kwargs(self) -> tp.Dict[str, tp.Any]: ...
    value: core.ValueProperty[tp.ArgsKwargs, tp.ArgsKwargs]
