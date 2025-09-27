from dataclasses import dataclass, field
from enum import Enum

__all__ = ['EventSource', 'Event', 'NodeState', 'RdzvEvent']

EventMetadataValue = str | int | float | bool | None

class EventSource(str, Enum):
    """Known identifiers of the event producers."""
    AGENT = 'AGENT'
    WORKER = 'WORKER'

@dataclass
class Event:
    """
    The class represents the generic event that occurs during the torchelastic job execution.

    The event can be any kind of meaningful action.

    Args:
        name: event name.
        source: the event producer, e.g. agent or worker
        timestamp: timestamp in milliseconds when event occurred.
        metadata: additional data that is associated with the event.
    """
    name: str
    source: EventSource
    timestamp: int = ...
    metadata: dict[str, EventMetadataValue] = field(default_factory=dict)
    def __str__(self) -> str: ...
    @staticmethod
    def deserialize(data: str | Event) -> Event: ...
    def serialize(self) -> str: ...

class NodeState(str, Enum):
    """The states that a node can be in rendezvous."""
    INIT = 'INIT'
    RUNNING = 'RUNNING'
    SUCCEEDED = 'SUCCEEDED'
    FAILED = 'FAILED'

@dataclass
class RdzvEvent:
    """
    Dataclass to represent any rendezvous event.

    Args:
        name: Event name. (E.g. Current action being performed)
        run_id: The run id of the rendezvous
        message: The message describing the event
        hostname: Hostname of the node
        pid: The process id of the node
        node_state: The state of the node (INIT, RUNNING, SUCCEEDED, FAILED)
        master_endpoint: The master endpoint for the rendezvous store, if known
        rank: The rank of the node, if known
        local_id: The local_id of the node, if defined in dynamic_rendezvous.py
        error_trace: Error stack trace, if this is an error event.
    """
    name: str
    run_id: str
    message: str
    hostname: str
    pid: int
    node_state: NodeState
    master_endpoint: str = ...
    rank: int | None = ...
    local_id: int | None = ...
    error_trace: str = ...
    def __str__(self) -> str: ...
    @staticmethod
    def deserialize(data: str | RdzvEvent) -> RdzvEvent: ...
    def serialize(self) -> str: ...
