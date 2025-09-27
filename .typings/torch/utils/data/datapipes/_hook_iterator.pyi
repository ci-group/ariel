from enum import Enum

class _SnapshotState(Enum):
    """
    These are the snapshotting-related states that IterDataPipes can be in.

    `NotStarted` - allows you to restore a snapshot and create an iterator with reset
    `Restored` - cannot restore again, allows you to create an iterator without resetting the DataPipe
    `Iterating` - can restore, will reset if you create a new iterator
    """
    NotStarted = 0
    Restored = 1
    Iterating = 2

def _simplify_obj_name(obj) -> str:
    """Simplify the display strings of objects for the purpose of rendering within DataPipe error messages."""
def _strip_datapipe_from_name(name: str) -> str: ...
def _generate_input_args_string(obj):
    """Generate a string for the input arguments of an object."""
def _generate_iterdatapipe_msg(datapipe, simplify_dp_name: bool = False): ...
def _gen_invalid_iterdatapipe_msg(datapipe): ...

_feedback_msg: str

def _check_iterator_valid(datapipe, iterator_id, next_method_exists: bool = False) -> None:
    """
    Given an instance of a DataPipe and an iterator ID, check if the IDs match, and if not, raises an exception.

    In the case of ChildDataPipe, the ID gets compared to the one stored in `main_datapipe` as well.
    """
def _set_datapipe_valid_iterator_id(datapipe):
    """Given a DataPipe, updates its valid iterator ID and reset the DataPipe."""
def hook_iterator(namespace):
    """
    Define a hook that is applied to all `__iter__` of metaclass `_DataPipeMeta`.

    This is done for the purpose of profiling and checking if an iterator is still valid.
    """
