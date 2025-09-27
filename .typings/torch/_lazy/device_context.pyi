from _typeshed import Incomplete
from typing import Any

class DeviceContext:
    _CONTEXTS: dict[str, Any]
    _CONTEXTS_LOCK: Incomplete
    device: Incomplete
    def __init__(self, device: str) -> None: ...

def get_device_context(device: str | None = None) -> DeviceContext: ...
