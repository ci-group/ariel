import multiprocessing.queues
from _typeshed import Incomplete

class ConnectionWrapper:
    """Proxy class for _multiprocessing.Connection which uses ForkingPickler for object serialization."""
    conn: Incomplete
    def __init__(self, conn) -> None: ...
    def send(self, obj) -> None: ...
    def recv(self): ...
    def __getattr__(self, name): ...

class Queue(multiprocessing.queues.Queue):
    _reader: ConnectionWrapper
    _writer: ConnectionWrapper
    _send: Incomplete
    _recv: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...

class SimpleQueue(multiprocessing.queues.SimpleQueue):
    _reader: ConnectionWrapper
    _writer: ConnectionWrapper
    def _make_methods(self) -> None: ...
