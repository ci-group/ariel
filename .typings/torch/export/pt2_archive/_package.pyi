import logging
import torch
from _typeshed import Incomplete
from dataclasses import dataclass
from torch._export.serde.serialize import SerializedArtifact as SerializedArtifact, deserialize as deserialize, serialize as serialize
from torch.export._tree_utils import reorder_kwargs as reorder_kwargs
from torch.export.exported_program import ExportedProgram as ExportedProgram
from torch.export.pt2_archive._package_weights import Weights as Weights, get_complete as get_complete, group_weights as group_weights
from torch.export.pt2_archive.constants import AOTINDUCTOR_DIR as AOTINDUCTOR_DIR, ARCHIVE_FORMAT_PATH as ARCHIVE_FORMAT_PATH, ARCHIVE_FORMAT_VALUE as ARCHIVE_FORMAT_VALUE, ARCHIVE_VERSION_PATH as ARCHIVE_VERSION_PATH, ARCHIVE_VERSION_VALUE as ARCHIVE_VERSION_VALUE, CONSTANTS_DIR as CONSTANTS_DIR, CUSTOM_OBJ_FILENAME_PREFIX as CUSTOM_OBJ_FILENAME_PREFIX, EXTRA_DIR as EXTRA_DIR, MODELS_DIR as MODELS_DIR, MODELS_FILENAME_FORMAT as MODELS_FILENAME_FORMAT, SAMPLE_INPUTS_FILENAME_FORMAT as SAMPLE_INPUTS_FILENAME_FORMAT, WEIGHTS_DIR as WEIGHTS_DIR, WEIGHT_FILENAME_PREFIX as WEIGHT_FILENAME_PREFIX
from torch.types import FileLike as FileLike
from torch.utils._ordered_set import OrderedSet as OrderedSet
from typing import Any
from typing_extensions import TypeAlias

DEFAULT_PICKLE_PROTOCOL: int
AOTI_FILES: TypeAlias = list[str | Weights] | dict[str, list[str | Weights]]
logger: logging.Logger

def is_pt2_package(serialized_model: bytes | str) -> bool:
    """
    Check if the serialized model is a PT2 Archive package.
    """

class PT2ArchiveWriter:
    """
    Context manager for writing a PT2 archive.
    """
    archive_file: Incomplete
    def __init__(self, archive_path_or_buffer: FileLike) -> None: ...
    def __enter__(self) -> PT2ArchiveWriter: ...
    def __exit__(self, *args: Any) -> None: ...
    def has_record(self, name: str) -> bool:
        """
        Check if a record exists in the archive.
        """
    def count_prefix(self, prefix: str) -> int:
        """
        Count the number of records that start with a given prefix.
        """
    def write_bytes(self, name: str, data: bytes) -> None:
        """
        Write a bytes object to the archive.
        name: The destination file inside the archive.
        data: The bytes object to write.
        """
    def write_string(self, name: str, data: str) -> None:
        """
        Write a string object to the archive.
        name: The destination file inside the archive.
        data: The string object to write.
        """
    def write_file(self, name: str, file_path: str) -> None:
        """
        Copy a file into the archive.
        name: The destination file inside the archive.
        file_path: The source file on disk.
        """
    def write_folder(self, archive_dir: str, folder_dir: str) -> None:
        """
        Copy a folder into the archive.
        archive_dir: The destination folder inside the archive.
        folder_dir: The source folder on disk.
        """
    def close(self) -> None:
        """
        Close the archive.
        """

class PT2ArchiveReader:
    """
    Context manager for reading a PT2 archive.
    """
    archive_file: Incomplete
    def __init__(self, archive_path_or_buffer: FileLike) -> None: ...
    def __enter__(self) -> PT2ArchiveReader: ...
    def __exit__(self, *args: Any) -> None: ...
    def read_bytes(self, name: str) -> bytes:
        """
        Read a bytes object from the archive.
        name: The source file inside the archive.
        """
    def read_string(self, name: str) -> str:
        """
        Read a string object from the archive.
        name: The source file inside the archive.
        """
    def archive_version(self) -> int:
        """
        Get the archive version.
        """
    def get_file_names(self) -> list[str]:
        """
        Get the file names in the archive.
        """

def _package_aoti_files(archive_writer: PT2ArchiveWriter, aoti_files: AOTI_FILES | None, pickle_protocol: int = ...) -> None: ...
def _package_exported_programs(archive_writer: PT2ArchiveWriter, exported_programs: ExportedProgram | dict[str, ExportedProgram] | None, opset_version: dict[str, int] | None = None, pickle_protocol: int = ...) -> None: ...
def _package_extra_files(archive_writer: PT2ArchiveWriter, extra_files: dict[str, Any] | None) -> None: ...
def package_pt2(f: FileLike, *, exported_programs: ExportedProgram | dict[str, ExportedProgram] | None = None, aoti_files: AOTI_FILES | None = None, extra_files: dict[str, Any] | None = None, opset_version: dict[str, int] | None = None, pickle_protocol: int = ...) -> FileLike:
    '''
    Saves the artifacts to a PT2Archive format
    (https://docs.google.com/document/d/1RQ4cmywilnFUT1VE-4oTGxwXdc8vowCSZsrRgo3wFA8/edit?tab=t.0#heading=h.v2y2jgnwc56a).
    The artifact can then be loaded using ``load_pt2``.

    Args:
        f (str | os.PathLike[str] | IO[bytes]) A file-like object (has to
         implement write and flush) or a string containing a file name.

        exported_programs (Union[ExportedProgram, dict[str, ExportedProgram]]):
         The exported program to save, or a dictionary mapping model name to an
         exported program to save. The exported program will be saved under
         models/*.json. If only one ExportedProgram is specified, this will
         automatically be named "model".

        aoti_files (Union[list[str], dict[str, list[str]]): A list of files
         generated by AOTInductor via
         ``torch._inductor.aot_compile(..., {"aot_inductor.package": True})``,
         or a dictionary mapping model name to its AOTInductor generated files.
         If only one set of files is specified, this will automatically be named
         "model".

        extra_files (Optional[Dict[str, Any]]): Map from filename to contents
         which will be stored as part of the pt2.

        opset_version (Optional[Dict[str, int]]): A map of opset names
         to the version of this opset

        pickle_protocol: can be specified to override the default protocol

    '''

class AOTICompiledModel:
    """
    Callable AOT Inductor loaded model from a .pt2
    """
    loader: Incomplete
    def __init__(self, loader: torch._C._aoti.AOTIModelPackageLoader) -> None: ...
    def __call__(self, *args, **kwargs): ...
    def get_metadata(self) -> dict[str, str]: ...
    def load_constants(self, constants_map: dict[str, torch.Tensor], *, check_full_update: bool, user_managed: bool = False) -> None:
        """
        Given a mapping of constant fqns to tensors, load the constants into the model.
        You can use ``get_constant_fqns`` to get the list of constant fqns that
        are needed in the compiled model.

        Args:
            constants_map: A mapping of constant fqns to tensors.
            check_full_update: Whether to add check to see if all the constants
            are updated and have values.
        """
    def get_constant_fqns(self) -> list[str]: ...
    def __deepcopy__(self, memo: dict[Any, Any] | None) -> AOTICompiledModel: ...

@dataclass
class PT2ArchiveContents:
    exported_programs: dict[str, ExportedProgram]
    aoti_runners: dict[str, AOTICompiledModel]
    extra_files: dict[str, Any]

def _load_exported_programs(archive_reader: PT2ArchiveReader, file_names: list[str], expected_opset_version: dict[str, int] | None) -> dict[str, ExportedProgram]: ...
def _load_extra_files(archive_reader: PT2ArchiveReader, file_names: list[str]) -> dict[str, Any]: ...
def load_pt2(f: FileLike, *, expected_opset_version: dict[str, int] | None = None, run_single_threaded: bool = False, num_runners: int = 1, device_index: int = -1, load_weights_from_disk: bool = False) -> PT2ArchiveContents:
    """
    Loads all the artifacts previously saved with ``package_pt2``.

    Args:
        f (str | os.PathLike[str] | IO[bytes]): A file-like object (has to
         implement write and flush) or a string containing a file name.

        expected_opset_version (Optional[Dict[str, int]]): A map of opset names
         to expected opset versions

        num_runners (int): Number of runners to load AOTInductor artifacts

        run_single_threaded (bool): Whether the model should be run without
            thread synchronization logic. This is useful to avoid conflicts with
            CUDAGraphs.

        device_index (int): The index of the device to which the PT2 package is
            to be loaded. By default, `device_index=-1` is used, which corresponds
            to the device `cuda` when using CUDA. Passing `device_index=1` would
            load the package to `cuda:1`, for example.

    Returns:
        A ``PT2ArchiveContents`` object which contains all the objects in the PT2.
    """
def load_weights_to_pt2_contents(pt2_contents: PT2ArchiveContents, weights_map: dict[str, Any]) -> None:
    """
    Load weights into the models in PT2 archive contents

    Args:
        pt2_contents (PT2ArchiveContents): The contents of the PT2 archive.
    """
