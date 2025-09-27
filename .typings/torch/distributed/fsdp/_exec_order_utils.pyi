import torch.distributed as dist
import torch.nn as nn
from _typeshed import Incomplete
from enum import Enum
from torch.distributed.fsdp._common_utils import _FSDPState as _FSDPState, _get_param_to_fqns as _get_param_to_fqns
from torch.distributed.fsdp._flat_param import FlatParamHandle as FlatParamHandle

class _ExecOrderWarnStatus(Enum):
    """Used internally for execution order validation."""
    NONE = ...
    WARNING = ...
    WARNED = ...

class _ExecOrderData:
    """
    This contains the data structures to track the execution order. We track
    the pre-forward order on the *first* iteration for forward prefetching
    (which thus assumes static graph) and the post-forward order on *every*
    iteration for backward prefetching (which thus does not assume static
    graph but may be provide an incorrect order).
    """
    handles_pre_forward_order: list[FlatParamHandle]
    handles_post_forward_order: list[FlatParamHandle | None]
    _iter: int
    _backward_prefetch_limit: Incomplete
    _forward_prefetch_limit: Incomplete
    _checking_order: bool
    process_group: dist.ProcessGroup | None
    world_size: int | None
    all_handles: list[FlatParamHandle]
    param_to_fqn: dict[nn.Parameter, list[str]]
    current_order_index: int
    warn_status: Incomplete
    def __init__(self, debug_level: dist.DebugLevel, backward_prefetch_limit: int, forward_prefetch_limit: int) -> None: ...
    rank: Incomplete
    def init(self, state: _FSDPState, root_module: nn.Module, process_group: dist.ProcessGroup) -> None:
        """
        Initializes the data structures needed for checking the forward order.
        This should be called after a root FSDP instance has been set during
        lazy initialization.
        """
    @property
    def is_first_iter(self) -> bool: ...
    def get_handle_to_backward_prefetch(self, current_handle: FlatParamHandle) -> FlatParamHandle | None:
        """
        Returns a :class:`list` of the handles keys of the handles to backward
        prefetch given the current handles key. If there are no valid handles
        keys to prefetch, then this returns an empty :class:`list`.
        """
    def get_handle_to_forward_prefetch(self, current_handle: FlatParamHandle) -> FlatParamHandle | None:
        """
        Returns a :class:`list` of the handles keys of the handles to forward
        prefetch given the current handles key. If there are no valid handles
        keys to prefetch, then this returns an empty :class:`list`.
        """
    def record_post_forward(self, handle: FlatParamHandle | None) -> None:
        """
        Records ``handles`` in the post-forward order, where ``handles`` should
        be a group of handles used in the same module's forward. If ``handles``
        is empty, then it is omitted.

        Unlike :meth:`record_pre_forward`, this records the order *every*
        iteration with the expectation that the recorded order is reset in
        :meth:`next_iter`.
        """
    def record_pre_forward(self, handle: FlatParamHandle | None, is_training: bool) -> None:
        """
        Records ``handles`` in the pre-forward order, where ``handles`` should
        be a group of handles used in the same module's forward. If ``handles``
        is empty, then it is omitted.

        On the first iteration, this checks the execution order across ranks.
        See :meth:`_check_order` for details.
        """
    def _check_order(self, handle: FlatParamHandle, is_training: bool) -> None:
        """
        Checks the forward execution order as long as ``is_training`` is
        ``True`` since checking in eval mode is not supported. This only checks
        if the distributed debug level is DETAIL.

        - On the first iteration, this uses all-gathers to check that all ranks
        are all-gathering the same handles and hence ``FlatParameter`` s,
        raising an error if not.
        - On subsequent iterations, this checks that each rank is locally
        consistent with its own forward order from the first iteration, issuing
        a warning if not. This issues a warning on the first deviating
        iteration and stops warning thereafter.
        """
    def _get_handle_indices(self, handle: FlatParamHandle) -> tuple[int | None, ...]:
        """
        Returns the handle indices (i.e. indices into ``self.all_handles``)
        corresponding to the handles in ``handle``. An entry in the
        returned tuple is ``None`` if the handle is invalid.
        """
    def _get_names_from_handle_indices(self, handle_indices: tuple[int, ...]) -> list[list[str]]:
        """
        Returns a list of FQNs for each handle in ``handle_indices``. If a
        handle index is invalid, then its FQNs are omitted from the returned
        list.
        """
    def _get_names_from_handles(self, handle: FlatParamHandle) -> list[list[str]]:
        """
        Returns a list of FQNs for each handle in ``handles_key``. If a handle
        is invalid, then its FQNs are omitted from the returned list.
        """
    def next_iter(self) -> None:
        """
        Advances the internal data structures per iteration. This should be
        called in the post-backward callback since that marks the true end of
        an iteration.
        """
