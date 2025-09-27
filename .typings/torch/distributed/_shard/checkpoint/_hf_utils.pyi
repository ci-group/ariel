import io
import torch
from _typeshed import Incomplete
from dataclasses import dataclass
from typing import Any

_metadata_fn: str
FILE_NAME: str
SHARDED_FILE_NAME: str
SUFFIX: str
CUSTOM_METADATA_KEY: str
DEFAULT_EXTRA_METADATA_KEY: str
SAVED_OFFSETS_KEY: str
SHAPE_KEY: str
DATA_KEY: str
DTYPE_KEY: str
DATA_OFFSETS_KEY: str
DTYPE_MAP: Incomplete
HF_DCP_VERSION: float
DCP_VERSION_KEY: str
DCP_SHARDING_INFO_KEY: str
FORMAT_KEY: str
FORMAT_VALUE: str

@dataclass
class _HFStorageInfo:
    """This is the per entry storage info."""
    relative_path: str
    offset: int
    length: int
    shape: torch.Size
    dtype: torch.dtype

def _gen_file_name(index: int, largest_index: int, shard_index: int | None = None) -> str: ...
def _get_safetensors_file_metadata(file_bytes: io.IOBase) -> tuple[Any, int]: ...
def _get_dtype(dtype_str: str) -> torch.dtype: ...
def _get_dcp_custom_metadata(metadata: Any) -> Any | None: ...
