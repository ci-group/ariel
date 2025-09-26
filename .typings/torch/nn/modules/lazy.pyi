from _typeshed import Incomplete
from typing import Any, Protocol

__all__ = ['LazyModuleMixin']

class _LazyProtocol(Protocol):
    """This class is used to avoid errors with mypy checks for the attributes in a mixin.

    https://mypy.readthedocs.io/en/latest/more_types.html#mixin-classes
    """
    def _register_load_state_dict_pre_hook(self, hook): ...
    def register_forward_pre_hook(self, hook, *, prepend: bool = False, with_kwargs: bool = False): ...
    def _lazy_load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs): ...
    def _get_name(self): ...
    def _infer_parameters(self, module, input): ...
    @property
    def _parameters(self): ...
    @property
    def _buffers(self): ...
    @property
    def _non_persistent_buffers_set(self): ...
    @property
    def _load_hook(self): ...
    @property
    def _initialize_hook(self): ...

class LazyModuleMixin:
    '''A mixin for modules that lazily initialize parameters, also known as "lazy modules".

    .. warning:
        Lazy modules are an experimental new feature under active development,
        and their API is likely to change.

    Modules that lazily initialize parameters, or "lazy modules",
    derive the shapes of their parameters from the first input(s)
    to their forward method. Until that first forward they contain
    :class:`torch.nn.UninitializedParameter` s that should not be accessed
    or used, and afterward they contain regular :class:`torch.nn.Parameter` s.
    Lazy modules are convenient since they don\'t require computing some
    module arguments, like the :attr:`in_features` argument of a
    typical :class:`torch.nn.Linear`.

    After construction, networks with lazy modules should first
    be converted to the desired dtype and placed on the expected device.
    This is because lazy modules only perform shape inference so the usual dtype
    and device placement behavior applies.
    The lazy modules should then perform "dry runs" to initialize all the components in the module.
    These "dry runs" send inputs of the correct size, dtype, and device through
    the network and to each one of its lazy modules. After this the network can be used as usual.

    >>> # xdoctest: +SKIP
    >>> class LazyMLP(torch.nn.Module):
    ...     def __init__(self) -> None:
    ...         super().__init__()
    ...         self.fc1 = torch.nn.LazyLinear(10)
    ...         self.relu1 = torch.nn.ReLU()
    ...         self.fc2 = torch.nn.LazyLinear(1)
    ...         self.relu2 = torch.nn.ReLU()
    ...
    ...     def forward(self, input):
    ...         x = self.relu1(self.fc1(input))
    ...         y = self.relu2(self.fc2(x))
    ...         return y
    >>> # constructs a network with lazy modules
    >>> lazy_mlp = LazyMLP()
    >>> # transforms the network\'s device and dtype
    >>> # NOTE: these transforms can and should be applied after construction and before any \'dry runs\'
    >>> lazy_mlp = lazy_mlp.cuda()
    >>> lazy_mlp
    LazyMLP( (fc1): LazyLinear(in_features=0, out_features=10, bias=True)
      (relu1): ReLU()
      (fc2): LazyLinear(in_features=0, out_features=1, bias=True)
      (relu2): ReLU()
    )
    >>> # performs a dry run to initialize the network\'s lazy modules
    >>> lazy_mlp(torch.ones(10, 10).cuda())
    >>> # after initialization, LazyLinear modules become regular Linear modules
    >>> lazy_mlp
    LazyMLP(
      (fc1): Linear(in_features=10, out_features=10, bias=True)
      (relu1): ReLU()
      (fc2): Linear(in_features=10, out_features=1, bias=True)
      (relu2): ReLU()
    )
    >>> # attaches an optimizer, since parameters can now be used as usual
    >>> optim = torch.optim.SGD(lazy_mlp.parameters(), lr=0.01)

    A final caveat when using lazy modules is that the order of initialization of a network\'s
    parameters may change, since the lazy modules are always initialized after other modules.
    For example, if the LazyMLP class defined above had a :class:`torch.nn.LazyLinear` module
    first and then a regular :class:`torch.nn.Linear` second, the second module would be
    initialized on construction and the first module would be initialized during the first dry run.
    This can cause the parameters of a network using lazy modules to be initialized differently
    than the parameters of a network without lazy modules as the order of parameter initializations,
    which often depends on a stateful random number generator, is different.
    Check :doc:`/notes/randomness` for more details.

    Lazy modules can be serialized with a state dict like other modules. For example:

    >>> lazy_mlp = LazyMLP()
    >>> # The state dict shows the uninitialized parameters
    >>> lazy_mlp.state_dict()
    OrderedDict({\'fc1.weight\': <UninitializedParameter>,
                 \'fc1.bias\': <UninitializedParameter>,
                 \'fc2.weight\': <UninitializedParameter>,
                 \'fc2.bias\': <UninitializedParameter>})

    Lazy modules can load regular :class:`torch.nn.Parameter` s (i.e. you can serialize/deserialize
    initialized LazyModules and they will remain initialized)


    >>> full_mlp = LazyMLP()
    >>> # Dry run to initialize another module
    >>> full_mlp.forward(torch.ones(10, 1))
    >>> # Load an initialized state into a lazy module
    >>> lazy_mlp.load_state_dict(full_mlp.state_dict())
    >>> # The state dict now holds valid values
    >>> lazy_mlp.state_dict()
    OrderedDict([(\'fc1.weight\',
                  tensor([[-0.3837],
                          [ 0.0907],
                          [ 0.6708],
                          [-0.5223],
                          [-0.9028],
                          [ 0.2851],
                          [-0.4537],
                          [ 0.6813],
                          [ 0.5766],
                          [-0.8678]])),
                 (\'fc1.bias\',
                  tensor([-1.8832e+25,  4.5636e-41, -1.8832e+25,  4.5636e-41, -6.1598e-30,
                           4.5637e-41, -1.8788e+22,  4.5636e-41, -2.0042e-31,  4.5637e-41])),
                 (\'fc2.weight\',
                  tensor([[ 0.1320,  0.2938,  0.0679,  0.2793,  0.1088, -0.1795, -0.2301,  0.2807,
                            0.2479,  0.1091]])),
                 (\'fc2.bias\', tensor([0.0019]))])

    Note, however, that the loaded parameters will not be replaced when doing a "dry run" if they are initialized
    when the state is loaded. This prevents using initialized modules in different contexts.
    '''
    cls_to_become: type[Any] | None
    _load_hook: Incomplete
    _initialize_hook: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def _save_to_state_dict(self, destination, prefix, keep_vars): ...
    def _lazy_load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """load_state_dict pre-hook function for lazy buffers and parameters.

        The purpose of this hook is to adjust the current state and/or
        ``state_dict`` being loaded so that a module instance serialized in
        both un/initialized state can be deserialized onto both un/initialized
        module instance.
        See comment in ``torch.nn.Module._register_load_state_dict_pre_hook``
        for the details of the hook specification.
        """
    def initialize_parameters(self, *args, **kwargs):
        """Initialize parameters according to the input batch properties.

        This adds an interface to isolate parameter initialization from the
        forward pass when doing parameter shape inference.
        """
    def has_uninitialized_params(self):
        """Check if a module has parameters that are not initialized."""
    def _infer_parameters(self, module, args, kwargs=None):
        """Infers the size and initializes the parameters according to the provided input batch.

        Given a module that contains parameters that were declared inferrable
        using :class:`torch.nn.parameter.ParameterMode.Infer`, runs a forward pass
        in the complete module using the provided input to initialize all the parameters
        as needed.
        The module is set into evaluation mode before running the forward pass in order
        to avoid saving statistics or calculating gradients
        """
    def _replicate_for_data_parallel(self): ...
