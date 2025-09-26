from _typeshed import Incomplete
from torch.distributed.checkpoint._fsspec_filesystem import FsspecReader, FsspecWriter
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.checkpoint.planner import LoadPlan, LoadPlanner, SavePlan, SavePlanner, WriteItem
from torch.distributed.checkpoint.storage import WriteResult
from torch.futures import Future

__all__ = ['HuggingFaceStorageWriter', 'HuggingFaceStorageReader']

class HuggingFaceStorageWriter(FsspecWriter):
    """
    A writer that writes to a huggingface repository in the huggingface format.
    Uses Fsspec back-end to communicate with back-end storage.
    Fsspec registration of the storage solution is required.
    """
    _fqn_to_index_mapping: dict[str, int] | None
    _save_sharded: Incomplete
    def __init__(self, path: str, fqn_to_index_mapping: dict[str, int] | None = None, token: str | None = None, save_sharded: bool = False) -> None:
        """
        Initialize the huggingface writer pointing to path.

        Args:
            path: hf directory where the checkpoint will be read from.
                  Needs to have .safetensors files, but can be from any fsspec supported storage,
                  including localFS and hf://.
            fqn_to_index_mapping: A mapping from tensor FQN to the index of the file that the tensor should be written to.
                              Indices are from 1 to N, where N is the number of files. If not provided,
                              the tensors will be written to a single file. If none, then all the tensors on the
                              same rank will be written to the same file.
            token: The token to use to authenticate with huggingface hub.
            save_sharded: If True, save the checkpoint as a sharded checkpoint where every rank saves its own shard.
                        Default is False which assumes full tensors are being saved.

        """
    def prepare_global_plan(self, plans: list[SavePlan]) -> list[SavePlan]: ...
    def write_data(self, plan: SavePlan, planner: SavePlanner) -> Future[list[WriteResult]]: ...
    def finish(self, metadata: Metadata, results: list[list[WriteResult]]) -> None: ...
    def _split_by_storage_plan(self, storage_plan: dict[str, int] | None, items: list[WriteItem]) -> dict[int, list[WriteItem]]: ...
    @property
    def metadata_path(self) -> str: ...

class HuggingFaceStorageReader(FsspecReader):
    """
    A reader that reads from a huggingface repository in the huggingface format.
    Uses in Fsspec back-end to communicate with storage.
    Fsspec registration of the storage solution is required.
    """
    def __init__(self, path: str, token: str | None = None) -> None:
        """
        Initialize the huggingface reader pointing to path.

        Args:
            path: hf directory where the checkpoint will be read from.
            Needs to have .safetensors file, but can be from any fsspec supported storage,
            including localFS and hf://.
            token: The token to use to authenticate with huggingface hub.
        """
    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]: ...
    def read_metadata(self) -> Metadata: ...
