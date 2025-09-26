import abc
from _typeshed import Incomplete

__all__ = ['BaseDataScheduler']

class BaseDataScheduler(metaclass=abc.ABCMeta):
    """
    The BaseDataScheduler is the abstract scheduler class specifically for the
    BaseDataSparsifier class. This class controls a specific hyperparameter of
    the sparsifier class and varies it across the training process (or across time).

    Args:
        data_sparsifier (instance of BaseDataSparsifier)
            Implemented class data sparsifier class wherein the update_mask is implemented
        schedule_param (str)
            A specific hyperparameter of the passed sparsifier that needs to be scheduled/varied
        last_epoch (int, default=-1)
            This is specifically is passed when training needs to be resumed from a particular
            point.
        verbose (bool, default=False)
            Verbosity of the BaseDataScheduler

    The *get_hyperparam()* function needs to be implemented by the user.
    """
    data_sparsifier: Incomplete
    schedule_param: Incomplete
    base_param: Incomplete
    last_epoch: Incomplete
    _step_count: int
    verbose: Incomplete
    _get_sp_called_within_step: bool
    def __init__(self, data_sparsifier, schedule_param: str, last_epoch: int = -1, verbose: bool = False) -> None: ...
    @abc.abstractmethod
    def get_schedule_param(self):
        """
        Abstract method that needs to be implemented by the child class.
        The expected return type should is a dictionary of name to schedule_param value
        The returned values will be updated in sparsifier when the scheduler step() function
        is called.

        Example:
            >>> def get_schedule_param(self):
            ...     new_param = {}
            ...     for name in self.sparsifier.data_groups.keys():
            ...         new_param[name] = (
            ...             self.sparsifier.data_groups[name][self.schedule_param] * 0.5
            ...         )
            ...     return new_param

        When the step() function is called, the value in self.sparsifier.data_groups[name][self.schedule_param]
        would be halved
        """
    def __repr__(self) -> str: ...
    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the sparsifier.

        Note:
            The scheduler class does not track the state of the data_sparsifier.
            Make sure to store the state of the sparsifier before storing the
            state of the scheduler
        """
    def load_state_dict(self, state_dict) -> None:
        """Loads the schedulers state.

        Note:
            Remember to restore the state of the data_sparsifier before the scheduler.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
    def get_last_param(self): ...
    o: Incomplete
    _last_param: Incomplete
    def step(self): ...
