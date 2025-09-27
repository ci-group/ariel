import torch.nn as nn
import weakref

class _State: ...

_module_state_mapping: weakref.WeakKeyDictionary[nn.Module, weakref.ReferenceType[_State]]

def _insert_module_state(module: nn.Module, state: _State) -> None: ...
def _get_module_state(module: nn.Module) -> _State | None:
    """
    Return the ``_State`` in ``model``.

    Given a ``module``, this API finds out if the module is also a ``_State``
    instance or if the module is managed by a composable API. If the module
    is also a ``_State``, ``module`` will be casted to ``_State` and returned.
    If it is managed by a composable API, the corresponding ``_State`` will
    be returned.
    """
