from _typeshed import Incomplete
from typing import Any

__all__ = ['RemovableHandle', 'unserializable_hook', 'warn_if_has_hooks', 'BackwardHook']

class RemovableHandle:
    """
    A handle which provides the capability to remove a hook.

    Args:
        hooks_dict (dict): A dictionary of hooks, indexed by hook ``id``.
        extra_dict (Union[dict, List[dict]]): An additional dictionary or list of
            dictionaries whose keys will be deleted when the same keys are
            removed from ``hooks_dict``.
    """
    id: int
    next_id: int
    hooks_dict_ref: Incomplete
    extra_dict_ref: tuple
    def __init__(self, hooks_dict: Any, *, extra_dict: Any = None) -> None: ...
    def remove(self) -> None: ...
    def __getstate__(self): ...
    def __setstate__(self, state) -> None: ...
    def __enter__(self) -> RemovableHandle: ...
    def __exit__(self, type: Any, value: Any, tb: Any) -> None: ...

def unserializable_hook(f):
    """
    Mark a function as an unserializable hook with this decorator.

    This suppresses warnings that would otherwise arise if you attempt
    to serialize a tensor that has a hook.
    """
def warn_if_has_hooks(tensor) -> None: ...

class BackwardHook:
    """
    A wrapper class to implement nn.Module backward hooks.

    It handles:
      - Ignoring non-Tensor inputs and replacing them by None before calling the user hook
      - Generating the proper Node to capture a set of Tensor's gradients
      - Linking the gradients captures for the outputs with the gradients captured for the input
      - Calling the user hook once both output and input gradients are available
    """
    user_hooks: Incomplete
    user_pre_hooks: Incomplete
    module: Incomplete
    grad_outputs: Incomplete
    n_outputs: int
    output_tensors_index: Incomplete
    n_inputs: int
    input_tensors_index: Incomplete
    def __init__(self, module, user_hooks, user_pre_hooks) -> None: ...
    def _pack_with_none(self, indices, values, size): ...
    def _unpack_none(self, indices, values): ...
    def _set_user_hook(self, grad_fn): ...
    def _apply_on_tensors(self, fn, args): ...
    def setup_input_hook(self, args): ...
    def setup_output_hook(self, args): ...
