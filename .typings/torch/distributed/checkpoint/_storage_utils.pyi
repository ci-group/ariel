import os
from .filesystem import FileSystemReader as FileSystemReader, FileSystemWriter as FileSystemWriter
from .storage import StorageReader as StorageReader, StorageWriter as StorageWriter

def _storage_setup(storage: StorageReader | StorageWriter | None, checkpoint_id: str | os.PathLike | None, reader: bool = False) -> None | StorageReader | StorageWriter: ...
