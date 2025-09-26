import abc
import threading
import torch.distributed as dist
from .api import RendezvousHandler, RendezvousInfo, RendezvousParameters, RendezvousStoreInfo
from .utils import _PeriodicTimer
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from torch.distributed import Store
from torch.distributed.elastic.events import NodeState
from typing import Any, Callable

__all__ = ['RendezvousBackend', 'RendezvousTimeout', 'RendezvousSettings', 'DynamicRendezvousHandler', 'create_handler']

Token = Any

class RendezvousBackend(ABC, metaclass=abc.ABCMeta):
    """Represent a backend that holds the rendezvous state."""
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the backend."""
    @abstractmethod
    def get_state(self) -> tuple[bytes, Token] | None:
        """Get the rendezvous state.

        Returns:
            A tuple of the encoded rendezvous state and its fencing token or
            ``None`` if no state is found in the backend.

        Raises:
            RendezvousConnectionError:
                The connection to the backend has failed.
            RendezvousStateError:
                The rendezvous state is corrupt.
        """
    @abstractmethod
    def set_state(self, state: bytes, token: Token | None = None) -> tuple[bytes, Token, bool] | None:
        """Set the rendezvous state.

        The new rendezvous state is set conditionally:

          - If the specified ``token`` matches the fencing token stored in the
            backend, the state will be updated. The new state will be returned
            to the caller along with its fencing token.
          - If the specified ``token`` does not match the fencing token stored
            in the backend, the state won't be updated; instead the existing
            state along with its fencing token will be returned to the caller.
          - If the specified ``token`` is ``None``, the new state will be set
            only if there is no existing state in the backend. Either the new
            state or the existing state along with its fencing token will be
            returned to the caller.

        Args:
            state:
                The encoded rendezvous state.
            token:
                An optional fencing token that was retrieved by a previous call
                to :py:meth:`get_state` or ``set_state()``.

        Returns:
            A tuple of the serialized rendezvous state, its fencing token, and
            a boolean value indicating whether our set attempt succeeded.

        Raises:
            RendezvousConnectionError:
                The connection to the backend has failed.
            RendezvousStateError:
                The rendezvous state is corrupt.
        """

class RendezvousTimeout:
    """Hold the timeout configuration of a rendezvous.

    Args:
        join:
            The time within which the rendezvous is expected to complete.
        last_call:
            An additional wait amount before completing the rendezvous once the
            rendezvous has the minimum number of required participants.
        close:
            The time within which the rendezvous is expected to close after a
            call to :py:meth:`RendezvousHandler.set_closed` or
            :py:meth:`RendezvousHandler.shutdown`.
        heartbeat:
            The time within which a keep-alive heartbeat is expected to
            complete.
    """
    _ZERO: Incomplete
    _DEFAULT_TIMEOUTS: Incomplete
    _join: timedelta
    _last_call: timedelta
    _close: timedelta
    _heartbeat: timedelta
    def __init__(self, join: timedelta | None = None, last_call: timedelta | None = None, close: timedelta | None = None, heartbeat: timedelta | None = None) -> None: ...
    @property
    def join(self) -> timedelta:
        """Get the join timeout."""
    @property
    def last_call(self) -> timedelta:
        """Get the last call timeout."""
    @property
    def close(self) -> timedelta:
        """Get the close timeout."""
    @property
    def heartbeat(self) -> timedelta:
        """Get the keep-alive heartbeat timeout."""
    def _set_timeouts(self, **timeouts: timedelta | None): ...

@dataclass(repr=False, eq=False, frozen=True)
class RendezvousSettings:
    """Hold the settings of the rendezvous.

    Attributes:
        run_id:
            The run id of the rendezvous.
        min_nodes:
            The minimum number of nodes to admit to the rendezvous.
        max_nodes:
            The maximum number of nodes to admit to the rendezvous.
        timeout:
            The timeout configuration of the rendezvous.
        keep_alive_interval:
            The amount of time a node waits before sending a heartbeat to keep
            it alive in the rendezvous.
        keep_alive_max_attempt:
            The maximum number of failed heartbeat attempts after which a node
            is considered dead.
    """
    run_id: str
    min_nodes: int
    max_nodes: int
    timeout: RendezvousTimeout
    keep_alive_interval: timedelta
    keep_alive_max_attempt: int

@dataclass(eq=True, order=True, frozen=True)
class _NodeDesc:
    """Describe a node in the rendezvous.

    Attributes:
        addr:
            The FQDN of the node or user specified local node address.
        pid:
            The id of the process in which the rendezvous handler runs.
        local_id:
            A process-wide unique id.
    """
    addr: str
    pid: int
    local_id: int
    def __repr__(self) -> str: ...

class _NodeDescGenerator:
    """Generate node descriptors.

    A node descriptor is a combination of an FQDN, a process id, and an auto-
    incremented integer that uniquely identifies a node in the rendezvous.
    """
    _lock: threading.Lock
    _local_id: int
    def __init__(self) -> None: ...
    def generate(self, local_addr: str | None = None) -> _NodeDesc: ...

class _RendezvousState:
    """Hold the state of a rendezvous.

    Attributes:
        round:
            The current round of the rendezvous.
        complete:
            A boolean value indicating whether the current round of the
            rendezvous is complete.
        deadline:
            The time at which the current round of the rendezvous will be
            considered complete if it is still waiting for nodes to join.
        closed:
            A boolean value indicating whether the rendezvous is closed.
        participants:
            A dictionary of the participants and their corresponding ranks.
        wait_list:
            A set of nodes that are waiting to participate in the next round of
            the rendezvous.
        redundancy_list:
            A set of nodes that are redundant in the current round and can join
            the next rendezvous without triggering re-rendezvous.
        last_heartbeats:
            A dictionary containing each node's last heartbeat time.
    """
    round: int
    complete: bool
    deadline: datetime | None
    closed: bool
    participants: dict[_NodeDesc, int]
    wait_list: set[_NodeDesc]
    redundancy_list: set[_NodeDesc]
    last_heartbeats: dict[_NodeDesc, datetime]
    def __init__(self) -> None: ...

class _RendezvousStateHolder(ABC, metaclass=abc.ABCMeta):
    """Hold the shared rendezvous state synced with other nodes."""
    @property
    @abstractmethod
    def state(self) -> _RendezvousState:
        """Get the local state."""
    @abstractmethod
    def sync(self) -> bool | None:
        """Read or writes the latest state.

        Returns:
            A boolean value indicating whether the local state, in case marked
            as dirty, was successfully synced with other nodes.
        """
    @abstractmethod
    def mark_dirty(self) -> None:
        """Mark the local state as dirty."""

class _BackendRendezvousStateHolder(_RendezvousStateHolder):
    """Hold the rendezvous state synced with other nodes via a backend.

    Args:
        backend:
            The rendezvous backend to use.
        settings:
            The rendezvous settings.
        cache_duration:
            The amount of time, in seconds, to cache the last rendezvous state
            before requesting it from the backend again.
    """
    _backend: RendezvousBackend
    _state: _RendezvousState
    _settings: RendezvousSettings
    _cache_duration: int
    _token: Token
    _dirty: bool
    _last_sync_time: float
    _dead_nodes: list[_NodeDesc]
    def __init__(self, backend: RendezvousBackend, settings: RendezvousSettings, cache_duration: int = 1) -> None: ...
    def _record(self, message: str, node_state: NodeState = ...): ...
    @property
    def state(self) -> _RendezvousState:
        """See base class."""
    def sync(self) -> bool | None:
        """See base class."""
    def _sanitize(self) -> None: ...
    def mark_dirty(self) -> None:
        """See base class.

        If the local rendezvous state is dirty, the next sync call will try to
        write the changes back to the backend. However this attempt might fail
        if another node, which had the same state, also made changes and wrote
        them before us.
        """

class _Action(Enum):
    """Specifies the possible actions based on the state of the rendezvous."""
    KEEP_ALIVE = 1
    ADD_TO_PARTICIPANTS = 2
    ADD_TO_WAIT_LIST = 3
    ADD_TO_REDUNDANCY_LIST = 4
    REMOVE_FROM_PARTICIPANTS = 5
    REMOVE_FROM_WAIT_LIST = 6
    REMOVE_FROM_REDUNDANCY_LIST = 7
    MARK_RENDEZVOUS_COMPLETE = 8
    MARK_RENDEZVOUS_CLOSED = 9
    SYNC = 10
    ERROR_CLOSED = 11
    ERROR_TIMEOUT = 12
    FINISH = 13

class _RendezvousContext:
    """Holds the context of the rendezvous.

    Attributes:
        node:
            The node descriptor associated with the current rendezvous handler
            instance.
        state:
            The current state of the rendezvous.
        settings:
            The rendezvous settings.
    """
    node: _NodeDesc
    state: _RendezvousState
    settings: RendezvousSettings
    def __init__(self, node: _NodeDesc, state: _RendezvousState, settings: RendezvousSettings) -> None: ...

class _RendezvousOpExecutor(ABC, metaclass=abc.ABCMeta):
    """Execute rendezvous operations."""
    @abstractmethod
    def run(self, state_handler: Callable[[_RendezvousContext, float], _Action], deadline: float, update_deadline: Callable[[timedelta], float] | None = None) -> None:
        """Execute a rendezvous operation.

        An operation is run inside a state machine and is expected to transition
        the rendezvous from one state to another.

        Args:
            state_handler:
                A callable that is expected to return the next state transition
                action based on the current state of the rendezvous.
            deadline:
                The time, in seconds, at which the operation will be considered
                timed-out.
            update_deadline:
                Function to generate a new operation deadline if the current
                node may participate in the next rendezvous.
        """

class _DistributedRendezvousOpExecutor(_RendezvousOpExecutor):
    """Execute rendezvous operations using a shared state.

    Args:
        node:
            The node descriptor associated with the current rendezvous handler
            instance.
        state_holder:
            The ``RendezvousStateHolder`` to use to sync the rendezvous state
            with other nodes.
        settings:
            The rendezvous settings.
    """
    _node: _NodeDesc
    _state: _RendezvousState
    _state_holder: _RendezvousStateHolder
    _settings: RendezvousSettings
    def __init__(self, node: _NodeDesc, state_holder: _RendezvousStateHolder, settings: RendezvousSettings) -> None: ...
    def _record(self, message: str, node_state: NodeState = ...) -> None: ...
    def run(self, state_handler: Callable[[_RendezvousContext, float], _Action], deadline: float, update_deadline: Callable[[timedelta], float] | None = None) -> None:
        """See base class."""
    def _keep_alive(self) -> None: ...
    def _add_to_participants(self) -> None: ...
    def _add_to_wait_list(self) -> None: ...
    def _add_to_redundancy_list(self) -> None: ...
    def _remove_from_participants(self) -> None: ...
    def _remove_from_wait_list(self) -> None: ...
    def _remove_from_redundancy_list(self) -> None: ...
    def _mark_rendezvous_complete(self) -> None: ...
    def _mark_rendezvous_closed(self) -> None: ...

class _RendezvousExitOp:
    """Represent a rendezvous exit operation."""
    def __call__(self, ctx: _RendezvousContext, deadline: float) -> _Action: ...

class _RendezvousJoinOp:
    """Represent a rendezvous join operation."""
    def __call__(self, ctx: _RendezvousContext, deadline: float) -> _Action: ...

class _RendezvousCloseOp:
    """Represent a rendezvous close operation."""
    def __call__(self, ctx: _RendezvousContext, deadline: float) -> _Action: ...

class _RendezvousKeepAliveOp:
    """Represent a rendezvous keep-alive update operation."""
    def __call__(self, ctx: _RendezvousContext, deadline: float) -> _Action: ...

class DynamicRendezvousHandler(RendezvousHandler):
    """Represent a handler that sets up a rendezvous among a set of nodes."""
    _node_desc_generator: Incomplete
    _this_node: _NodeDesc
    _settings: RendezvousSettings
    _backend_name: str
    _store: Store
    _state_holder: _RendezvousStateHolder
    _op_executor: _RendezvousOpExecutor
    _heartbeat_lock: threading.Lock
    _keep_alive_timer: _PeriodicTimer | None
    @classmethod
    def from_backend(cls, run_id: str, store: Store, backend: RendezvousBackend, min_nodes: int, max_nodes: int, local_addr: str | None = None, timeout: RendezvousTimeout | None = None, keep_alive_interval: int = 5, keep_alive_max_attempt: int = 3):
        """Create a new :py:class:`DynamicRendezvousHandler`.

        Args:
            run_id:
                The run id of the rendezvous.
            store:
                The C10d store to return as part of the rendezvous.
            backend:
                The backend to use to hold the rendezvous state.
            min_nodes:
                The minimum number of nodes to admit to the rendezvous.
            max_nodes:
                The maximum number of nodes to admit to the rendezvous.
            local_addr:
                The local node address.
            timeout:
                The timeout configuration of the rendezvous.
            keep_alive_interval:
                The amount of time a node waits before sending a heartbeat to keep
                it alive in the rendezvous.
            keep_alive_max_attempt:
                The maximum number of failed heartbeat attempts after which a node
                is considered dead.
        """
    _shared_tcp_store_server: dist.Store | None
    _bootstrap_store_info: RendezvousStoreInfo | None
    def __init__(self, node: _NodeDesc, settings: RendezvousSettings, backend_name: str, store: Store, state_holder: _RendezvousStateHolder) -> None: ...
    def _record(self, message: str, node_state: NodeState = ..., rank: int | None = None) -> None: ...
    def _create_tcp_store_server(self, master_addr, master_port) -> dist.TCPStore: ...
    @property
    def settings(self) -> RendezvousSettings:
        """Get the settings of the rendezvous."""
    def get_backend(self) -> str:
        """See base class."""
    @property
    def use_agent_store(self) -> bool:
        """See base class."""
    def next_rendezvous(self) -> RendezvousInfo:
        """See base class."""
    def is_closed(self) -> bool:
        """See base class."""
    def set_closed(self) -> None:
        """See base class."""
    def num_nodes_waiting(self) -> int:
        """See base class."""
    def get_run_id(self) -> str:
        """See base class."""
    def shutdown(self) -> bool:
        """See base class."""
    def _close(self) -> None: ...
    @staticmethod
    def _keep_alive_weak(weak_self) -> None: ...
    def _keep_alive(self) -> None: ...
    def _start_heartbeats(self) -> None: ...
    def _stop_heartbeats(self) -> None: ...
    def _get_world(self) -> tuple[int, int]: ...
    def _wrap_store(self, store: Store) -> Store: ...
    def _get_store(self) -> Store: ...
    def _get_deadline(self, timeout: timedelta) -> float: ...

def create_handler(store: Store, backend: RendezvousBackend, params: RendezvousParameters) -> DynamicRendezvousHandler:
    """Create a new :py:class:`DynamicRendezvousHandler` from the specified parameters.

    Args:
        store:
            The C10d store to return as part of the rendezvous.
        backend:
            The backend to use to hold the rendezvous state.

    +-------------------+------------------------------------------------------+
    | Parameter         | Description                                          |
    +===================+======================================================+
    | join_timeout      | The total time, in seconds, within which the         |
    |                   | rendezvous is expected to complete. Defaults to 600  |
    |                   | seconds.                                             |
    +-------------------+------------------------------------------------------+
    | last_call_timeout | An additional wait amount, in seconds, before        |
    |                   | completing the rendezvous once the minimum number of |
    |                   | nodes has been reached. Defaults to 30 seconds.      |
    +-------------------+------------------------------------------------------+
    | close_timeout     | The time, in seconds, within which the rendezvous is |
    |                   | expected to close after a call to                    |
    |                   | :py:meth:`RendezvousHandler.set_closed` or           |
    |                   | :py:meth:`RendezvousHandler.shutdown`. Defaults to   |
    |                   | 30 seconds.                                          |
    +-------------------+------------------------------------------------------+
    | heartbeat         | The time, in seconds, within which a keep-alive      |
    |                   | heartbeat is expected to complete                    |
    +-------------------+------------------------------------------------------+
    """
