import abc
import torch
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any, NamedTuple

__all__ = ['JoinHook', 'Joinable', 'Join']

class JoinHook:
    """
    This defines a join hook, which provides two entry points in the join context manager.

    Entry points : a main hook, which is called repeatedly while there exists a non-joined
    process, and a post-hook, which is called once all processes have joined.

    To implement a join hook for the generic join context manager, define a
    class that inherits from :class:`JoinHook` and override ``main_hook()`` and
    ``post_hook()`` as appropriate.
    """
    def main_hook(self) -> None:
        """Call this hook while there exists a non-joined process to shadow collective communications in a training iteration.

        Training iteration i.e., in one forward pass, backward pass, and optimizer step.
        """
    def post_hook(self, is_last_joiner: bool) -> None:
        """
        Call hook after all processes have joined.

        It is passed an additional ``bool`` argument ``is_last_joiner``, which indicates if the rank is one of the last to join.

        Arguments:
            is_last_joiner (bool): ``True`` if the rank is one of the last to
                join; ``False`` otherwise.
        """

class Joinable(ABC, metaclass=abc.ABCMeta):
    """
    This defines an abstract base class for joinable classes.

    A joinable class
    (inheriting from :class:`Joinable`) should implement :meth:`join_hook`,
    which returns a :class:`JoinHook` instance, in addition to
    :meth:`join_device` and :meth:`join_process_group` that return device and
    process group information, respectively.
    """
    _join_config: Incomplete
    @abstractmethod
    def __init__(self): ...
    @abstractmethod
    def join_hook(self, **kwargs) -> JoinHook:
        """
        Return a :class:`JoinHook` instance for the given :class:`Joinable`.

        Arguments:
            kwargs (dict): a :class:`dict` containing any keyword arguments
                to modify the behavior of the join hook at run time; all
                :class:`Joinable` instances sharing the same join context
                manager are forwarded the same value for ``kwargs``.
        """
    @property
    @abstractmethod
    def join_device(self) -> torch.device:
        """Return the device from which to perform collective communications needed by the join context manager."""
    @property
    @abstractmethod
    def join_process_group(self) -> Any:
        """Returns the process group for the collective communications needed by the join context manager itself."""

class _JoinConfig(NamedTuple):
    """This includes all fields needed from a :class:`Joinable` instance for the join context manager side."""
    enable: bool
    throw_on_early_termination: bool
    is_first_joinable: bool
    @staticmethod
    def construct_disabled_join_config():
        """Return a :class:`_JoinConfig` instance indicating that join-related logic should be disabled.

        e.g. if the caller is not in a join context manager.
        """

class Join:
    '''
    This class defines the generic join context manager, which allows custom hooks to be called after a process joins.

    These hooks should shadow the
    collective communications of non-joined processes to prevent hanging and
    erroring and to ensure algorithmic correctness. Refer to :class:`JoinHook`
    for details about the hook definition.

    .. warning::
        The context manager requires each participating :class:`Joinable` to
        call the method :meth:`notify_join_context()` before its own per-
        iteration collective communications to ensure correctness.

    .. warning::
        The context manager requires that all ``process_group`` attributes in
        the :class:`JoinHook` objects are the same. If there are multiple
        :class:`JoinHook` objects, then the ``device`` of the first is used.
        The process group and device information is used for checking for non-
        joined processes and for notifying processes to throw an exception if
        ``throw_on_early_termination`` is enabled, both of which using an all-
        reduce.

    Arguments:
        joinables (List[Joinable]): a list of the participating
            :class:`Joinable` s; their hooks are iterated over in the given
            order.

        enable (bool): a flag enabling uneven input detection; setting to
            ``False`` disables the context manager\'s functionality and should
            only be set when the user knows the inputs will not be uneven
            (default: ``True``).

        throw_on_early_termination (bool): a flag controlling whether to throw an
            exception upon detecting uneven inputs (default: ``False``).

    Example::

        >>> import os
        >>> import torch
        >>> import torch.distributed as dist
        >>> import torch.multiprocessing as mp
        >>> # xdoctest: +SKIP
        >>> import torch.nn.parallel.DistributedDataParallel as DDP
        >>> import torch.distributed.optim.ZeroRedundancyOptimizer as ZeRO
        >>> from torch.distributed.algorithms.join import Join
        >>>
        >>> # On each spawned worker
        >>> def worker(rank):
        >>>     dist.init_process_group("nccl", rank=rank, world_size=2)
        >>>     model = DDP(torch.nn.Linear(1, 1).to(rank), device_ids=[rank])
        >>>     optim = ZeRO(model.parameters(), torch.optim.Adam, lr=0.01)
        >>>     # Rank 1 gets one more input than rank 0
        >>>     inputs = [torch.tensor([1.]).to(rank) for _ in range(10 + rank)]
        >>>     with Join([model, optim]):
        >>>         for input in inputs:
        >>>             loss = model(input).sum()
        >>>             loss.backward()
        >>>             optim.step()
        >>>     # All ranks reach here without hanging/erroring
    '''
    _joinables: Incomplete
    _join_hooks: Incomplete
    _enable: Incomplete
    _throw_on_early_termination: Incomplete
    def __init__(self, joinables: list[Joinable], enable: bool = True, throw_on_early_termination: bool = False, **kwargs) -> None: ...
    def _set_joinable_configs(self) -> None:
        """Set the :class:`_JoinConfig` of each participating :class:`Joinable`."""
    _process_group: Incomplete
    _rank: Incomplete
    _device: Incomplete
    def _extract_dist_info(self) -> None:
        """
        Extract the process group and device information from the joinables.

        If there are multiple joinables, then the context manager uses the
        first specified device.

        Preconditions:
            ``self._joinables`` is not ``None`` and is non-empty.

        Raises:
            ValueError
                If there are multiple conflicting ``process_group`` attributes
                among the ``Joinable`` objects.
        """
    def __enter__(self) -> None: ...
    def __exit__(self, type: type[BaseException] | None, value: BaseException | None, traceback: TracebackType | None):
        """
        Repeatedly runs the main hooks until all processes join; then, runs the post-hooks.

        Raises:
            RuntimeError
                If ``throw_on_early_termination=True``.
        """
    def _get_num_nonjoined_procs(self):
        """Return the number of non-joined processes by shadowing an all-reduce in the non-joined processes."""
    def _notify_procs_to_terminate(self) -> None:
        """Schedule an all-reduce to notify non-joined processes to terminate.

        Also raise a ``RuntimeError`` indicating that the current process has exhausted its inputs.
        """
    @staticmethod
    def notify_join_context(joinable: Joinable):
        """
        Notifies the join context manager that the calling process has not yet joined.

        Then, if ``throw_on_early_termination=True``, checks if uneven inputs have been detected
        (i.e. if one process has already joined) and throws an exception if so.

        This method should be called from a :class:`Joinable` object before
        its per-iteration collective communications. For example, this should
        be called at the beginning of the forward pass in
        :class:`DistributedDataParallel`.

        Only the first :class:`Joinable` object passed into the context
        manager performs the collective communications in this method, and
        for the others, this method is vacuous.

        Arguments:
            joinable (Joinable): the :class:`Joinable` object calling this
                method.

        Returns:
            An async work handle for the all-reduce meant to notify the context
            manager that the process has not yet joined if ``joinable`` is the
            first one passed into the context manager; ``None`` otherwise.
        """
