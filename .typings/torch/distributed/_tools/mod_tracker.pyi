import types
import weakref
from _typeshed import Incomplete
from typing import Callable

__all__ = ['ModTracker']

class ModTracker:
    '''
    ``ModTracker`` is a context manager that tracks the nn.Module hierarchy during execution
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

        with ModTracker() as tracker:
            # Access anything during the forward pass
            def my_linear(m1, m2, bias):
                print(f"Current modules: {tracker.parents}")
                return torch.mm(m1, m2.t()) + bias

            torch.nn.functional.linear = my_linear

            mod(torch.rand(2, 2))

    '''
    parents: set[str]
    _active_module_cnt: Incomplete
    _known_modules: weakref.WeakKeyDictionary
    _seen_modules: weakref.WeakSet
    _has_callback: bool
    _post_bw_callbacks_to_enqueue: list[Callable]
    _user_pre_fw_hook: Incomplete
    _user_post_fw_hook: Incomplete
    _user_pre_bw_hook: Incomplete
    _user_post_bw_hook: Incomplete
    def __init__(self) -> None: ...
    def _maybe_set_engine_callback(self) -> None: ...
    @property
    def is_bw(self):
        """
        A boolean marking if this is currently running during the backward pass or not
        """
    def get_known_fqn(self, mod):
        """
        Return the fqn for the given module if it is known to the ``ModTracker``, otherwise ``None``.
        """
    def register_user_hooks(self, pre_fw_hook: Callable | None = None, post_fw_hook: Callable | None = None, pre_bw_hook: Callable | None = None, post_bw_hook: Callable | None = None):
        """
        Registers user-specified hooks to be called before/after the forward/backward pass for each
        module tracked by the ``ModTracker``. One or more can be ``None``.
        Args:
            pre_fw_hook (Callable, optional): A hook to be called before the forward pass for the
                module. It should have the following signature:
                pre_fw_hook (module, input) -> None
            post_fw_hook (Callable, optional): A hook to be called after the forward pass for the
                module. It should have the following signature:
                post_fw_hook (module, input, output) -> None
            pre_bw_hook (Callable, optional): A multi-grad hook to be called on all the outputs of
                the module that require gradients. It should have the following signature:
                pre_bw_hook (module, grad_output) -> None
            post_bw_hook (Callable, optional): A multi-grad hook to be called on all the inputs of
                the module that require gradients. It should have the following signature:
                post_bw_hook (module, grad_input) -> None
        Raises:
            AssertionError: If a new hook is provided when one is already registered.
        Note:
            If the module is not alive during the backward pass, the pre_bw_hook and post_bw_hook will
            will receive None as the module argument.
            The module fqn will be present in the ``parents`` attribute when each of the hooks is called.
            Hooks are intended to be used as markers only not to modify the inputs/outputs.
        """
    def clear_user_hooks(self) -> None:
        """
        Clears the user specified hooks registered with ``register_user_hooks``
        """
    def _get_mod_name(self, mod): ...
    def _get_append_fn(self, w_mod, name, is_bw): ...
    def _get_pop_fn(self, w_mod, name, is_bw): ...
    def _fw_pre_hook(self, mod, input) -> None: ...
    def _fw_post_hook(self, mod, input, output) -> None: ...
    _fw_pre_handle: Incomplete
    _fw_post_handle: Incomplete
    def __enter__(self): ...
    def __exit__(self, *args) -> None: ...
