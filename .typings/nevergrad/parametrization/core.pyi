import nevergrad.common.typing as tp
import numpy as np
from . import utils as utils
from ._layering import Layered as Layered, ValueProperty as ValueProperty
from _typeshed import Incomplete
from nevergrad.common import errors as errors

P = tp.TypeVar('P', bound='Parameter')

def default_congruence(x: tp.Any) -> tp.Any: ...

class Parameter(Layered):
    """Class providing the core functionality of a parameter, aka
    value, internal/model parameters, mutation, recombination
    and additional features such as shared random state,
    constraint check, hashes, generation and naming.
    The value field should sent to the function to optimize.

    Example
    -------
    >>> ng.p.Array(shape=(2,)).value
    array([0., 0.])
    """
    _LAYER_LEVEL: Incomplete
    value: ValueProperty[tp.Any, tp.Any]
    tabu_congruence: tp.Any
    neural: bool
    has_constraints: bool
    enforce_determinism: bool
    real_world: bool
    hptuning: bool
    tabu_fails: int
    _subobjects: Incomplete
    tabu_length: int
    tabu_set: tp.Set[tp.Any]
    tabu_list: tp.List[tp.Any]
    tabu_index: int
    parents_uids: tp.List[str]
    heritage: tp.Dict[tp.Hashable, tp.Any]
    loss: tp.Optional[float]
    _losses: tp.Optional[np.ndarray]
    _dimension: tp.Optional[int]
    _random_state: tp.Optional[np.random.RandomState]
    _generation: int
    _constraint_checkers: tp.List[tp.Callable[[tp.Any], tp.Union[bool, float]]]
    _name: tp.Optional[str]
    _frozen: bool
    _meta: tp.Dict[tp.Hashable, tp.Any]
    function: Incomplete
    def __init__(self) -> None: ...
    @property
    def losses(self) -> np.ndarray:
        """Possibly multiobjective losses which were told
        to the optimizer along this parameter.
        In case of mono-objective loss, losses is the array containing this loss as sole element

        Note
        ----
        This API is highly experimental
        """
    @property
    def args(self) -> tp.Tuple[tp.Any, ...]:
        """Value of the positional arguments.
        Used to input value in a function as `func(*param.args, **param.kwargs)`
        Use `parameter.Instrumentation` to set `args` and `kwargs` with full freedom.
        """
    @property
    def kwargs(self) -> tp.Dict[str, tp.Any]:
        """Value of the keyword arguments.
        Used to input value in a function as `func(*param.args, **param.kwargs)`
        Use `parameter.Instrumentation` to set `args` and `kwargs` with full freedom.
        """
    @property
    def dimension(self) -> int:
        """Dimension of the standardized space for this parameter
        i.e size of the vector returned by get_standardized_data(reference=...)
        """
    def mutate(self) -> None:
        """Mutate parameters of the instance, and then its value"""
    def _layered_mutate(self) -> None: ...
    def sample(self) -> P:
        '''Sample a new instance of the parameter.
        This usually means spawning a child and mutating it.
        This function should be used in optimizers when creating an initial population,
        and parameter.heritage["lineage"] is reset to parameter.uid instead of its parent\'s
        '''
    def recombine(self, *others: P) -> None:
        """Update value and parameters of this instance by combining it with
        other instances.

        Parameters
        ----------
        *others: Parameter
            other instances of the same type than this instance.
        """
    def get_standardized_data(self, *, reference: P) -> np.ndarray:
        '''Get the standardized data representing the value of the instance as an array in the optimization space.
        In this standardized space, a mutation is typically centered and reduced (sigma=1) Gaussian noise.
        The data only represent the value of this instance, not the parameters (eg.: mutable sigma), hence it does not
        fully represent the state of the instance. Also, in stochastic cases, the value can be non-deterministically
        deduced from the data (eg.: categorical variable, for which data includes sampling weights for each value)

        Parameters
        ----------
        reference: Parameter
            the reference instance for representation in the standardized data space. This keyword parameter is
            mandatory to make the code clearer.
            If you use "self", this method will always return a zero vector.

        Returns
        -------
        np.ndarray
            the representation of the value in the optimization space

        Note
        ----
        - Operations between different standardized data should only be performed if each array was produced
          by the same reference in the exact same state (no mutation)
        - to make the code more explicit, the "reference" parameter is enforced as a keyword-only parameter.
        '''
    def _internal_get_standardized_data(self, reference: P) -> np.ndarray: ...
    def set_standardized_data(self, data: tp.ArrayLike, *, reference: tp.Optional[P] = None) -> P:
        '''Updates the value of the provided reference (or self) using the standardized data.

        Parameters
        ----------
        np.ndarray
            the representation of the value in the optimization space
        reference: Parameter
            the reference point for representing the data ("self", if not provided)

        Returns
        -------
        Parameter
            self (modified)

        Note
        ----
        To make the code more explicit, the "reference" is enforced
        as keyword-only parameters.
        '''
    def _internal_set_standardized_data(self, data: np.ndarray, reference: P) -> None: ...
    @property
    def generation(self) -> int:
        """Generation of the parameter (children are current generation + 1)"""
    def get_value_hash(self) -> tp.Hashable:
        """Hashable object representing the current value of the instance"""
    def __repr__(self) -> str: ...
    def __bool__(self) -> bool: ...
    def can_skip_constraints(self, ref: tp.Optional[P] = None) -> bool: ...
    def satisfies_constraints(self, ref: tp.Optional[P] = None, no_tabu: bool = False) -> bool:
        """Whether the instance satisfies the constraints added through
        the `register_cheap_constraint` method

        Parameters
        ----------
        ref
            parameter of the optimization algorithm, if we want to check the tabu list.
        no_tabu
            if we want to ignore the Tabu system.
        Returns
        -------
        bool
            True iff the constraint is satisfied
        """
    def specify_tabu_length(self, tabu_length: int) -> None: ...
    def register_cheap_constraint(self, func: tp.Union[tp.Callable[[tp.Any], bool], tp.Callable[[tp.Any], float]], as_layer: bool = False) -> None:
        '''Registers a new constraint on the parameter values.

        Parameters
        ----------
        func: Callable
            function which, given the value of the instance, returns whether it satisfies the constraints (if output = bool),
            or a float which is >= 0 if the constraint is satisfied.

        Note
        ----
        - this is only for checking after mutation/recombination/etc if the value still satisfy the constraints.
          The constraint is not used in those processes.
        - constraints should be fast to compute.
        - this function has an additional "as_layer" parameter which is experimental for now, and can have unexpected
          behavior
        '''
    @property
    def random_state(self) -> np.random.RandomState:
        """Random state the instrumentation and the optimizers pull from.
        It can be seeded/replaced.
        """
    @random_state.setter
    def random_state(self, random_state: tp.Optional[np.random.RandomState]) -> None: ...
    def _set_random_state(self, random_state: tp.Optional[np.random.RandomState]) -> None: ...
    def spawn_child(self, new_value: tp.Optional[tp.Any] = None) -> P:
        """Creates a new instance which shares the same random generator than its parent,
        is sampled from the same data, and mutates independently from the parentp.
        If a new value is provided, it will be set to the new instance

        Parameters
        ----------
        new_value: anything (optional)
            if provided, it will update the new instance value (cannot be used at the same time as new_data).

        Returns
        -------
        Parameter
            a new instance of the same class, with same content/internal-model parameters/...
            Optionally, a new value will be set after creation
        """
    def copy(self) -> P:
        """Creates a full copy of the parameter (with new unique uid).
        Use spawn_child instead to make sure to add the parenthood information.
        """
    def _set_parenthood(self, parent: tp.Optional['Parameter']) -> None:
        """Sets the parenthood information to Parameter and subparameters."""
    def freeze(self) -> None:
        """Prevents the parameter from changing value again (through value, mutate etc...)"""
    def _check_frozen(self) -> None: ...

class Constant(Parameter):
    """Parameter-like object for simplifying management of constant parameters:
    mutation/recombination do nothing, value cannot be changed, standardize data is an empty array,
    child is the same instance.

    Parameter
    ---------
    value: Any
        the value that this parameter will always provide
    """
    _value: Incomplete
    def __init__(self, value: tp.Any) -> None: ...
    def _get_name(self) -> str: ...
    def get_value_hash(self) -> tp.Hashable: ...
    def _layered_get_value(self) -> tp.Any: ...
    def _layered_set_value(self, value: tp.Any) -> None: ...
    def _layered_sample(self) -> P: ...
    def get_standardized_data(self, *, reference: tp.Optional[P] = None) -> np.ndarray: ...
    value: Incomplete
    def spawn_child(self, new_value: tp.Optional[tp.Any] = None) -> P: ...
    def recombine(self, *others: P) -> None: ...
    def mutate(self) -> None: ...

def as_parameter(param: tp.Any) -> Parameter:
    """Returns a Parameter from anything:
    either the input if it is already a parameter, or a Constant if not
    This is convenient for iterating over Parameter and other objects alike
    """

class MultiobjectiveReference(Constant):
    def __init__(self, parameter: tp.Optional[Parameter] = None) -> None: ...

class Operator(Layered):
    """Layer object that can be used as an operator on a Parameter"""
    _LAYER_LEVEL: Incomplete
    def __call__(self, parameter: Parameter) -> Parameter:
        """Applies the operator on a Parameter to create a new Parameter"""
