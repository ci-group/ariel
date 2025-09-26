import datetime
from _typeshed import Incomplete
from torch.distributed import Store as Store

def cas_delay() -> None: ...

class EtcdStore(Store):
    """
    Implement a c10 Store interface by piggybacking on the rendezvous etcd instance.

    This is the store object returned by ``EtcdRendezvous``.
    """
    client: Incomplete
    prefix: Incomplete
    def __init__(self, etcd_client, etcd_store_prefix, timeout: datetime.timedelta | None = None) -> None: ...
    def set(self, key, value) -> None:
        """
        Write a key/value pair into ``EtcdStore``.

        Both key and value may be either Python ``str`` or ``bytes``.
        """
    def get(self, key) -> bytes:
        """
        Get a value by key, possibly doing a blocking wait.

        If key is not immediately present, will do a blocking wait
        for at most ``timeout`` duration or until the key is published.


        Returns:
            value ``(bytes)``

        Raises:
            LookupError - If key still not published after timeout
        """
    def add(self, key, num: int) -> int:
        """
        Atomically increment a value by an integer amount.

        The integer is represented as a string using base 10. If key is not present,
        a default value of ``0`` will be assumed.

        Returns:
             the new (incremented) value


        """
    def wait(self, keys, override_timeout: datetime.timedelta | None = None):
        """
        Wait until all of the keys are published, or until timeout.

        Raises:
            LookupError - if timeout occurs
        """
    def check(self, keys) -> bool:
        """Check if all of the keys are immediately present (without waiting)."""
    def _encode(self, value) -> str: ...
    def _decode(self, value) -> bytes: ...
    def _try_wait_get(self, b64_keys, override_timeout=None): ...
