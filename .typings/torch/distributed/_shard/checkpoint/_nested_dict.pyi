from . import _version as _version
from ._traverse import OBJ_PATH as OBJ_PATH, STATE_DICT_ITEM as STATE_DICT_ITEM, set_element as set_element, traverse_state_dict as traverse_state_dict, traverse_state_dict_v_2_3 as traverse_state_dict_v_2_3
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE as STATE_DICT_TYPE

FLATTEN_MAPPING = dict[str, OBJ_PATH]

def flatten_state_dict(state_dict: STATE_DICT_TYPE) -> tuple[STATE_DICT_TYPE, FLATTEN_MAPPING]:
    """
    Flatten ``state_dict`` made of nested dicts and lists into a top level dictionary.

    Use ``unflatten_state_dict`` to revert this process.
    Returns:
        A tuple with the flatten state_dict and a mapping from original to new state_dict.
    N.B. The new keys are derived from the object paths, joined by dot.
        For example: ``{ 'a': {'b':...}}`` results in the key `a.b`.
    """
def unflatten_state_dict(state_dict: STATE_DICT_TYPE, mapping: FLATTEN_MAPPING) -> STATE_DICT_TYPE:
    """Restore the original nested state_dict according to ``mapping`` and the flattened ``state_dict``."""
