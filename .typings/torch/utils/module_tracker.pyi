import types
import weakref
from _typeshed import Incomplete
from torch.utils.hooks import RemovableHandle

__all__ = ['ModuleTracker']

class ModuleTracker:
    '''
    ``ModuleTracker`` is a context manager that tracks the nn.Module hierarchy during execution
    so that other system can query which Module is currently being executed (or its backward is being
    executed).

    You can access the ``parents`` attribute on this context manager to get the set of all the
    Modules currently being executed via their fqn (fully qualified name, also used as the key within
    the state_dict).
    You can access the ``is_bw`` attribute to know if you are currently running in backward or not.

    Note that ``parents`` is never empty and always contains the "Global" key. The ``is_bw`` flag
    will remain ``True`` after the forward until another Module is executed. If you need it to be
    more accurate, please submit an issue requesting this. Adding a map from fqn to the module instance
    is possible but not done yet, please submit an issue requesting this if you need it.

    Example usage

    .. code-block:: python

        mod = torch.nn.Linear(2, 2)

        with ModuleTracker() as tracker:
            # Access anything during the forward pass
            def my_linear(m1, m2, bias):
                print(f"Current modules: {tracker.parents}")
                return torch.mm(m1, m2.t()) + bias
            torch.nn.functional.linear = my_linear

            mod(torch.rand(2, 2))

    '''
    parents: set[str]
    _known_modules: weakref.WeakKeyDictionary
    _seen_modules: weakref.WeakSet
    _has_callback: bool
    _hooks: list[RemovableHandle]
    def __init__(self) -> None: ...
    def _maybe_set_engine_callback(self) -> None: ...
    @property
    def is_bw(self):
        """
        A boolean marking if this is currently running during the backward pass or not
        """
    def _get_mod_name(self, mod): ...
    def _get_append_fn(self, name, is_bw): ...
    def _get_pop_fn(self, name, is_bw): ...
    def _fw_pre_hook(self, mod, input) -> None: ...
    def _fw_post_hook(self, mod, input, output) -> None: ...
    _fw_pre_handle: Incomplete
    _fw_post_handle: Incomplete
    def __enter__(self): ...
    def __exit__(self, *args) -> None: ...
