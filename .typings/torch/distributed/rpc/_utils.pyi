from _typeshed import Incomplete
from collections.abc import Generator
from contextlib import contextmanager

logger: Incomplete

@contextmanager
def _group_membership_management(store, name, is_join) -> Generator[None]: ...
def _update_group_membership(worker_info, my_devices, reverse_device_map, is_join): ...
