import io
import os
from _typeshed import Incomplete
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from fsspec import AbstractFileSystem
from torch.distributed.checkpoint._extension import StreamTransformExtension
from torch.distributed.checkpoint.filesystem import FileSystemBase, FileSystemReader, FileSystemWriter, SerializationFormat

__all__ = ['FsspecWriter', 'FsspecReader']

class FileSystem(FileSystemBase):
    fs: AbstractFileSystem | None
    def __init__(self) -> None: ...
    @contextmanager
    def create_stream(self, path: str | os.PathLike, mode: str) -> Generator[io.IOBase, None, None]: ...
    def concat_path(self, path: str | os.PathLike, suffix: str) -> str | os.PathLike: ...
    def init_path(self, path: str | os.PathLike, **kwargs) -> str | os.PathLike: ...
    def rename(self, path: str | os.PathLike, new_path: str | os.PathLike) -> None: ...
    def mkdir(self, path: str | os.PathLike) -> None: ...
    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool: ...
    def exists(self, path: str | os.PathLike) -> bool: ...
    def rm_file(self, path: str | os.PathLike) -> None: ...
    def ls(self, path: str | os.PathLike) -> list[str]: ...

class FsspecWriter(FileSystemWriter):
    """
    Basic implementation of StorageWriter using FFspec.

    This implementation makes the following assumptions and simplifications:

    * The checkpoint path is an empty or non-existing directory.
    * File creation is atomic

    The checkpoint consist of one file per write request plus
    a `.metadata` file with the serialized metadata.

    """
    fs: Incomplete
    path: Incomplete
    def __init__(self, path: str | os.PathLike, single_file_per_rank: bool = True, sync_files: bool = True, thread_count: int = 1, per_thread_copy_ahead: int = 10000000, overwrite: bool = True, _extensions: Sequence[StreamTransformExtension] | None = None, serialization_format: SerializationFormat = ..., **kwargs) -> None:
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
    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool: ...

class FsspecReader(FileSystemReader):
    fs: Incomplete
    path: Incomplete
    def __init__(self, path: str | os.PathLike, **kwargs) -> None: ...
    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool: ...
