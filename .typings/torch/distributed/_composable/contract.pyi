import torch.nn as nn
from _typeshed import Incomplete
from torch.distributed._composable_state import _State as _State
from torch.distributed.utils import _get_root_modules as _get_root_modules
from typing import Callable, Generic, Protocol
from typing_extensions import Concatenate, ParamSpec, TypeVar

_T = TypeVar('_T', covariant=True)
_P = ParamSpec('_P')

def generate_state_key(string: str = '__composable_api_state_key'): ...

STATE_KEY: Incomplete
REGISTRY_KEY: Incomplete

class RegistryItem: ...
_TState = TypeVar('_TState', bound='_State', covariant=True)
_M = TypeVar('_M', nn.Module, list[nn.Module])

class _ContractFn(Generic[_P, _T, _TState], Protocol):
    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _T: ...
    def state(self, module: nn.Module) -> _TState: ...

def contract(state_cls: type[_TState] = ...) -> Callable[[Callable[Concatenate[_M, _P], _M]], _ContractFn[Concatenate[_M, _P], _M, _TState]]:
    '''
    Decorate a function as a composable distributed API, where the first
    argument of the function must be an :class:`nn.Module` instance or sequence
    of :class:`nn.Module` instances.

    The decorator verifies that the decorated function does not modify
    fully-qualified names (FQNs) for parameters, buffers, or modules. The
    decorated function can return different module instances than the input
    modules; the FQN invariant will be enforced following the input order.

    When a function ``func`` is decorated by ``@contract()``, a
    ``.state(module: nn.Module)`` method will be installed to the decorated
    function. Then you can retrieve and modify the state on a module by calling
    ``func.state(module)``.

    Example::
        >>> # xdoctest: +SKIP
        >>> import torch.nn as nn
        >>>
        >>> class MyModel(nn.Module):
        >>>     def __init__(self) -> None:
        >>>         super().__init__()
        >>>         self.l1 = nn.Linear(10, 10)
        >>>         self.l2 = nn.Linear(10, 10)
        >>>
        >>>     def forward(self, x):
        >>>         return self.l2(self.l1(x))
        >>>
        >>> @contract()
        >>> def my_feature(module: nn.Module) -> nn.Module:
        >>>     my_feature.state(module).some_state = "any value"
        >>>     return module
        >>>
        >>> model = MyModel()
        >>> my_feature(model.l1)
        >>> assert my_feature.state(model.l1).some_state == "any value"
        >>> my_feature(model.l2)
        >>> model(torch.randn(2, 10)).sum().backward()
    '''
def _get_registry(module: nn.Module) -> dict[str, RegistryItem] | None:
    """
    Get an ``OrderedDict`` of composable APIs that have been applied to the
    ``module``, indexed by the API name. If no API has been applied, then this
    returns ``None``.
    """
