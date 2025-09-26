import torch
from dataclasses import dataclass
from io import BufferedIOBase
from torch.serialization import DEFAULT_PROTOCOL as DEFAULT_PROTOCOL, MAP_LOCATION as MAP_LOCATION, _load as _load, _save as _save
from typing import Any

__all__: list[str]

@dataclass
class _Entry:
    key: str
    is_storage: bool
    length: int

class _PseudoZipFile:
    records: dict[str, tuple[object, int]]
    def __init__(self) -> None: ...
    def write_record(self, key: str, data: object, length: int) -> None: ...
    def write_to(self, f: BufferedIOBase) -> None: ...
    def read_from(self, f: BufferedIOBase) -> None: ...
    def has_record(self, key: str) -> bool: ...
    def get_record(self, key: str) -> object: ...
    def get_storage_from_record(self, key: str, _length: int, _type: int) -> torch.Tensor: ...
    def serialization_id(self) -> str: ...

def _streaming_save(obj: object, f: BufferedIOBase, pickle_module: Any = ..., pickle_protocol: int = ...) -> None:
    """
    Save the object to a file-like object in a streaming fashion compatible with
    network sockets.

    This behaves similarly to :func:`torch.save` with a few notable differences:

    * A non-seekable file like object can be used when loading.
    * No forwards/backwards compatibility is provided for the serialization
      format. This is only intended to be used with a single version of PyTorch
      with transient storage (i.e. sockets or temp files).
    * mmap is not supported

    See :func:`torch.save` for more details on specific arguments.
    """
def _streaming_load(f: BufferedIOBase, map_location: MAP_LOCATION = None, pickle_module: Any = None, *, weights_only: bool = True, **pickle_load_args: Any) -> object:
    """
    Load the object from a file-like object in a streaming fashion compatible with
    network sockets.

    See :func:`_streaming_save` for more details about the streaming behavior.

    See :func:`torch.load` for more details on specific arguments.
    """
