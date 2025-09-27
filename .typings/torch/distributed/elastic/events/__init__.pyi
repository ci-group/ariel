import logging
from .api import Event as Event, EventMetadataValue as EventMetadataValue, EventSource as EventSource, NodeState as NodeState, RdzvEvent as RdzvEvent
from torch.distributed.elastic.events.handlers import get_logging_handler as get_logging_handler

_events_loggers: dict[str, logging.Logger]

def _get_or_create_logger(destination: str = 'null') -> logging.Logger:
    """
    Construct python logger based on the destination type or extends if provided.

    Available destination could be found in ``handlers.py`` file.
    The constructed logger does not propagate messages to the upper level loggers,
    e.g. root logger. This makes sure that a single event can be processed once.

    Args:
        destination: The string representation of the event handler.
            Available handlers found in ``handlers`` module
    """
def record(event: Event, destination: str = 'null') -> None: ...
def record_rdzv_event(event: RdzvEvent) -> None: ...
def construct_and_record_rdzv_event(run_id: str, message: str, node_state: NodeState, name: str = '', hostname: str = '', pid: int | None = None, master_endpoint: str = '', local_id: int | None = None, rank: int | None = None) -> None:
    '''
    Initialize rendezvous event object and record its operations.

    Args:
        run_id (str): The run id of the rendezvous.
        message (str): The message describing the event.
        node_state (NodeState): The state of the node (INIT, RUNNING, SUCCEEDED, FAILED).
        name (str): Event name. (E.g. Current action being performed).
        hostname (str): Hostname of the node.
        pid (Optional[int]): The process id of the node.
        master_endpoint (str): The master endpoint for the rendezvous store, if known.
        local_id (Optional[int]):  The local_id of the node, if defined in dynamic_rendezvous.py
        rank (Optional[int]): The rank of the node, if known.
    Returns:
        None
    Example:
        >>> # See DynamicRendezvousHandler class
        >>> def _record(
        ...     self,
        ...     message: str,
        ...     node_state: NodeState = NodeState.RUNNING,
        ...     rank: Optional[int] = None,
        ... ) -> None:
        ...     construct_and_record_rdzv_event(
        ...         name=f"{self.__class__.__name__}.{get_method_name()}",
        ...         run_id=self._settings.run_id,
        ...         message=message,
        ...         node_state=node_state,
        ...         hostname=self._this_node.addr,
        ...         pid=self._this_node.pid,
        ...         local_id=self._this_node.local_id,
        ...         rank=rank,
        ...     )
    '''
