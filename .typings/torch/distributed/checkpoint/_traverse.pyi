from collections.abc import MutableMapping
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from typing import Callable, TypeVar

__all__ = ['traverse_state_dict', 'set_element', 'get_element', 'print_tensor']

PATH_ITEM = str | int
OBJ_PATH = tuple[PATH_ITEM, ...]
T = TypeVar('T')
STATE_DICT_ITEM = object
CONTAINER_TYPE = MutableMapping[PATH_ITEM, STATE_DICT_ITEM]

def traverse_state_dict(state_dict: STATE_DICT_TYPE, visitor: Callable[[OBJ_PATH, STATE_DICT_ITEM], None], keep_traversing: Callable[[STATE_DICT_ITEM], bool] = ...) -> None:
    """
    Invoke ``visitor`` for each value recursively in ``state_dict``.
    Mapping will be traversed and ``visitor`` will be applied to the leaf elements.
    ``visitor`` will only be applied to elements in a list or a tuple, if the
    container contains tensors or mappings.
    """
def set_element(root_dict: STATE_DICT_TYPE, path: OBJ_PATH, value: STATE_DICT_ITEM) -> None:
    """Set ``value`` in ``root_dict`` along the ``path`` object path."""
def get_element(root_dict: STATE_DICT_TYPE, path: OBJ_PATH, default_value: T | None = None) -> T | None:
    """Retrieve the value at ``path``from ``root_dict``, returning ``default_value`` if not found."""
def print_tensor(path: OBJ_PATH, value: STATE_DICT_ITEM, print_fun: Callable[[str], None] = ...) -> None:
    """
    Use this callback with traverse_state_dict to print its content.

    By default the content is printed using the builtin ``print`` but this can
    be change by passing a different ``print_fun` callable.
    """
