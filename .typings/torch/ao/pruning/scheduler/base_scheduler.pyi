from _typeshed import Incomplete

__all__ = ['BaseScheduler']

class BaseScheduler:
    sparsifier: Incomplete
    base_sl: Incomplete
    last_epoch: Incomplete
    _step_count: int
    verbose: Incomplete
    _get_sl_called_within_step: bool
    def __init__(self, sparsifier, last_epoch: int = -1, verbose: bool = False) -> None: ...
    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the sparsifier.
        """
    def load_state_dict(self, state_dict) -> None:
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
    def get_last_sl(self):
        """Return last computed sparsity level by current scheduler."""
    def get_sl(self) -> None: ...
    def print_sl(self, is_verbose, group, sl, epoch=None) -> None:
        """Display the current sparsity level."""
    def __repr__(self) -> str: ...
    o: Incomplete
    _last_sl: Incomplete
    def step(self, epoch=None): ...
    def _make_sure_a_list(self, var):
        """Utility that extends it to the same length as the .groups, ensuring it is a list"""
