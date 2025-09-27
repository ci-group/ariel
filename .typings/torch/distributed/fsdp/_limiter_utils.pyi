import collections
import torch

class _FreeEventQueue:
    """
    This tracks all pending frees corresponding to inflight all-gathers. The
    queueing pattern is iterative enqueues with a single dequeue per iteration
    once the limit ``_max_num_inflight_all_gathers`` is reached.
    """
    _queue: collections.deque[torch.Event]
    _max_num_inflight_all_gathers: int
    def __init__(self) -> None: ...
    def enqueue(self, free_event: torch.Event) -> None:
        """Enqueues a free event."""
    def dequeue_if_needed(self) -> torch.Event | None:
        """Dequeues a single event if the limit is reached."""
    def _dequeue(self) -> torch.Event | None:
        """Dequeues a free event if possible."""
