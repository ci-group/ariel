import abc
import collections
import io
import os
import queue
import torch
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from torch.distributed.checkpoint._extension import ExtensionRegistry, StreamTransformExtension
from torch.distributed.checkpoint.metadata import Metadata, STATE_DICT_TYPE, StorageMeta
from torch.distributed.checkpoint.planner import LoadPlan, LoadPlanner, ReadItem, SavePlan, SavePlanner, WriteItem
from torch.distributed.checkpoint.staging import BlockingAsyncStager
from torch.distributed.checkpoint.storage import StorageReader, StorageWriter, WriteResult
from torch.futures import Future
from typing import Any, Callable, IO

__all__ = ['FileSystemWriter', 'FileSystemReader', 'FileSystem', 'FileSystemBase', 'SerializationFormat']

@dataclass
class _StorageInfo:
    """This is the per entry storage info."""
    relative_path: str
    offset: int
    length: int
    transform_descriptors: Sequence[str] | None = ...
    def __getstate__(self): ...

@dataclass
class _StoragePrefix:
    prefix: str

class SerializationFormat(Enum):
    TORCH_SAVE = 'torch_save'
    SAFETENSORS = 'safetensors'

class _TensorLoader(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def add(self, size: int, obj: object) -> None: ...
    @abstractmethod
    def start_loading(self) -> None: ...
    @abstractmethod
    def values(self) -> Iterator[tuple[torch.Tensor, object]]: ...

class _SerialCpuLoader(_TensorLoader):
    resolve_fun: Incomplete
    items: list[tuple[int, object]]
    def __init__(self, resolve_fun: Callable) -> None: ...
    def add(self, size: int, obj: object) -> None: ...
    def start_loading(self) -> None: ...
    def values(self) -> Iterator[tuple[torch.Tensor, object]]: ...

class _OverlappingCpuLoader(_TensorLoader):
    resolve_fun: Incomplete
    items: list[tuple[int, object]]
    inflight_threshhold: Incomplete
    in_flight_data: int
    current_items: collections.deque
    idx: int
    started: bool
    device_type: Incomplete
    device_module: Incomplete
    stream: Incomplete
    def __init__(self, resolve_fun: Callable, stream: torch.Stream | None = None, inflight_threshhold: int = 1000000) -> None: ...
    @property
    def _done(self) -> bool: ...
    def _drain(self) -> list[tuple[torch.Tensor, object]]: ...
    def _refill(self) -> None: ...
    def _finish(self) -> Iterable[tuple[torch.Tensor, object]]: ...
    def add(self, size: int, obj: object) -> None: ...
    def start_loading(self) -> None: ...
    def values(self) -> Iterator[tuple[torch.Tensor, object]]: ...

class _StorageWriterTransforms:
    """
    This is experimental, and will likely move elsewhere in the
    future.  It lives here to minimize changes while we are still
    learning and gathering feedback.
    """
    extensions: Incomplete
    def __init__(self, extensions: Sequence[StreamTransformExtension] | None = None) -> None:
        """
        If the extensions arg is None, this means the implementation
        should provide whatever defaults it chooses.  An empty
        sequence indicates no extensions should be used.  At this
        time, the default extensions sequence is empty.
        """
    raw: Incomplete
    def transform_save_stream(self, write_item: WriteItem, raw_stream: io.IOBase) -> tuple[IO[bytes], list[str]]: ...

class FileSystemBase(ABC, metaclass=abc.ABCMeta):
    @contextmanager
    @abstractmethod
    def create_stream(self, path: str | os.PathLike, mode: str) -> Generator[io.IOBase, None, None]: ...
    @abstractmethod
    def concat_path(self, path: str | os.PathLike, suffix: str) -> str | os.PathLike: ...
    @abstractmethod
    def rename(self, path: str | os.PathLike, new_path: str | os.PathLike) -> None: ...
    @abstractmethod
    def init_path(self, path: str | os.PathLike) -> str | os.PathLike: ...
    @abstractmethod
    def mkdir(self, path: str | os.PathLike) -> None: ...
    @classmethod
    @abstractmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool: ...
    @abstractmethod
    def exists(self, path: str | os.PathLike) -> bool: ...
    @abstractmethod
    def rm_file(self, path: str | os.PathLike) -> None: ...

class FileSystem(FileSystemBase):
    @contextmanager
    def create_stream(self, path: str | os.PathLike, mode: str) -> Generator[io.IOBase, None, None]: ...
    def concat_path(self, path: str | os.PathLike, suffix: str) -> str | os.PathLike: ...
    def init_path(self, path: str | os.PathLike) -> str | os.PathLike: ...
    def rename(self, path: str | os.PathLike, new_path: str | os.PathLike) -> None: ...
    def mkdir(self, path: str | os.PathLike) -> None: ...
    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool: ...
    def exists(self, path: str | os.PathLike) -> bool: ...
    def rm_file(self, path: str | os.PathLike) -> None: ...
    def ls(self, path: str | os.PathLike) -> list[str]: ...

class _FileSystemWriter(StorageWriter):
    """
    Basic implementation of StorageWriter using file IO.

    This implementation makes the following assumptions and simplifications:

    * The checkpoint path is an empty or non-existing directory.
    * File creation is atomic

    The checkpoint consist of one file per write request plus
    a `.metadata` file with the serialized metadata.

    """
    fs: Incomplete
    path: Incomplete
    single_file_per_rank: Incomplete
    sync_files: Incomplete
    thread_count: Incomplete
    per_thread_copy_ahead: Incomplete
    save_id: Incomplete
    overwrite: Incomplete
    transforms: Incomplete
    serialization_format: Incomplete
    def __init__(self, path: str | os.PathLike, single_file_per_rank: bool = True, sync_files: bool = True, thread_count: int = 1, per_thread_copy_ahead: int = 10000000, overwrite: bool = True, _extensions: Sequence[StreamTransformExtension] | None = None, serialization_format: SerializationFormat = ..., *args: Any, **kwargs: Any) -> None:
        """
        Initialize the writer pointing to `path`.

        Args:
            path: directory where the checkpoint will be written to.
            single_file_per_rank: Produce one file per rank instead of one file per tensor/blob. Default to True.
            sync_files : force files to be synced to permanent storage. Default to True.
            thread_count: Number of IO threads to use to write. Default to 1.
            per_thread_copy_ahead: How many bytes to copy from the GPU ahead of saving then. Default 10Mb.
            overwrite: Whether to allow overwriting existing checkpoints. Defaults to True.
            _extensions: Extensions to apply to output streams (EXPERIMENTAL)

        N. B. If sync_files is disabled, there's no guarantee that the checkpoint will be consistent in the case of a failure.
        """
    def reset(self, checkpoint_id: str | os.PathLike | None = None) -> None: ...
    def set_up_storage_writer(self, is_coordinator: bool) -> None: ...
    def prepare_local_plan(self, plan: SavePlan) -> SavePlan: ...
    def prepare_global_plan(self, plans: list[SavePlan]) -> list[SavePlan]: ...
    def write_data(self, plan: SavePlan, planner: SavePlanner) -> Future[list[WriteResult]]: ...
    def _write_data(self, planner: SavePlanner, file_queue: queue.Queue) -> Future[list[WriteResult]]: ...
    def finish(self, metadata: Metadata, results: list[list[WriteResult]]) -> None: ...
    def storage_meta(self) -> StorageMeta | None: ...
    @property
    def metadata_path(self) -> str | os.PathLike: ...
    @property
    def checkpoint_id(self) -> str | os.PathLike:
        """
        return the checkpoint_id that will be used to save the checkpoint.
        """
    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool: ...

class _StorageReaderTransforms:
    """
    This is experimental, and will likely move elsewhere in the
    future.  It lives here to minimize changes while we are still
    learning and gathering feedback.
    """
    extension_registry: Incomplete
    def __init__(self, extension_registry: ExtensionRegistry | None = None) -> None: ...
    def transform_load_stream(self, read_item: ReadItem, transform_descriptors: Sequence[str], raw_stream: IO[bytes]) -> IO[bytes]: ...

class FileSystemReader(StorageReader):
    fs: Incomplete
    path: Incomplete
    storage_data: dict[Any, Any]
    load_id: Incomplete
    transforms: Incomplete
    def __init__(self, path: str | os.PathLike, _extension_registry: ExtensionRegistry | None = None) -> None: ...
    def _slice_file(self, file, sinfo: _StorageInfo) -> IO[bytes]: ...
    def reset(self, checkpoint_id: str | os.PathLike | None = None) -> None: ...
    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]: ...
    def read_metadata(self) -> Metadata: ...
    def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool) -> None: ...
    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan: ...
    def prepare_global_plan(self, plans: list[LoadPlan]) -> list[LoadPlan]: ...
    @property
    def checkpoint_id(self) -> str | os.PathLike:
        """
        return the checkpoint_id that will be used to load the checkpoint.
        """
    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool: ...

class FileSystemWriter(_FileSystemWriter, BlockingAsyncStager):
    """
    Basic implementation of StorageWriter using file IO.

    This implementation makes the following assumptions and simplifications:

    * The checkpoint path is an empty or non-existing directory.
    * File creation is atomic

    The checkpoint consist of one file per write request plus
    a `.metadata` file with the serialized metadata.

    """
    def __init__(self, path: str | os.PathLike, single_file_per_rank: bool = True, sync_files: bool = True, thread_count: int = 1, per_thread_copy_ahead: int = 10000000, cache_staged_state_dict: bool = False, overwrite: bool = True, _extensions: Sequence[StreamTransformExtension] | None = None, serialization_format: SerializationFormat = ...) -> None:
        """
        Initialize the writer pointing to `path`.

        Args:
            path: directory where the checkpoint will be written to.
            single_file_per_rank: Produce one file per rank instead of one file per tensor/blob. Default to True.
            sync_files : force files to be synced to permanent storage. Default to True.
            thread_count: Number of IO threads to use to write. Default to 1.
            per_thread_copy_ahead: How many bytes to copy from the GPU ahead of saving then. Default 10Mb.
            cache_staged_state_dict: Whether to cache the staged state_dict. This option decreases staging latency
                at the cost of increases memory usage. Additionally, if this parameter is set to True, it's the expectation
                that the stager is maintained and reused for multiple dcp.async_save calls. Default to False.
            overwrite: Whether to allow overwriting existing checkpoints. Defaults to True.
            _extensions: Extensions to apply to output streams (EXPERIMENTAL)

        N. B. If sync_files is disabled, there's no guarantee that the checkpoint will be consistent in the case of a failure.
        """
    per_thread_copy_ahead: int
    def stage(self, state_dict: STATE_DICT_TYPE) -> STATE_DICT_TYPE:
        """Override of AsyncStager.stage"""
