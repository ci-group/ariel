import contextlib
import nevergrad.common.typing as tp
from . import utils as utils
from _typeshed import Incomplete
from nevergrad.common import testing as testing
from pathlib import Path

LINETOKEN: Incomplete
COMMENT_CHARS: Incomplete

def _convert_to_string(data: tp.Any, extension: str) -> str:
    """Converts the data into a string to be injected in a file"""

class Placeholder:
    """Placeholder tokens to for external code instrumentation"""
    pattern: Incomplete
    name: Incomplete
    comment: Incomplete
    def __init__(self, name: str, comment: tp.Optional[str]) -> None: ...
    @classmethod
    def finditer(cls, text: str) -> tp.List['Placeholder']: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: tp.Any) -> bool: ...
    @classmethod
    def sub(cls, text: str, extension: str, replacers: tp.Dict[str, tp.Any]) -> str: ...

def symlink_folder_tree(folder: tp.Union[Path, str], shadow_folder: tp.Union[Path, str]) -> None:
    """Utility for copying the tree structure of a folder and symlinking all files
    This can help creating lightweight copies of a project, for instantiating several
    copies with different parameters.
    """
def uncomment_line(line: str, extension: str) -> str: ...

class FileTextFunction:
    """Function created from a file and generating the text file after
    replacement of the placeholders
    """
    filepath: Incomplete
    placeholders: Incomplete
    _text: Incomplete
    parameters: tp.Set[str]
    def __init__(self, filepath: Path) -> None: ...
    def __call__(self, **kwargs: tp.Any) -> str: ...
    def __repr__(self) -> str: ...

class FolderInstantiator:
    """Folder with instrumentation tokens, which can be instantiated.

    Parameters
    ----------
    folder: str/Path
        the instrumented folder to instantiate
    clean_copy: bool
        whether to create an initial clean temporary copy of the folder in order to avoid
        versioning problems (instantiations are lightweight symlinks in any case).

    Caution
    -------
        The clean copy is generally located in /tmp and may not be accessible for
        computation in a cluster. You may want to create a clean copy yourself
        in the folder of your choice, or set the the TemporaryDirectoryCopy class
        (located in instrumentation.instantiate) CLEAN_COPY_DIRECTORY environment
        variable to a shared directory
    """
    _clean_copy: Incomplete
    folder: Incomplete
    file_functions: tp.List[FileTextFunction]
    def __init__(self, folder: tp.Union[Path, str], clean_copy: bool = False) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def placeholders(self) -> tp.List[Placeholder]: ...
    def instantiate_to_folder(self, outfolder: tp.Union[Path, str], kwargs: tp.Dict[str, tp.Any]) -> None: ...
    @contextlib.contextmanager
    def instantiate(self, **kwargs: tp.Any) -> tp.Generator[Path, None, None]: ...

class FolderFunction:
    """Turns a folder into a parametrized function
    (with nevergrad tokens)

    Parameters
    ----------
    folder: Path/str
        path to the folder to instrument
    command: list
        command to run from inside the folder. The last line in stdout will
        be the output of the function.
        The command must be performed from just outside the instrument
        directory
    verbose: bool
        whether to print the run command and from where it is run.
    clean_copy: bool
        whether to create an initial clean temporary copy of the folder in order to avoid
        versioning problems (instantiations are lightweight symlinks in any case)

    Returns
    -------
    Any
        the post-processed output of the called command

    Note
    ----
    By default, the postprocessing attribute holds a function which recovers the last line
    and converts it to float. The sequence of postprocessing can however be tampered
    with directly in order to change it

    Caution
    -------
        The clean copy is generally located in /tmp and may not be accessible for
        computation in a cluster. You may want to create a clean copy yourself
        in the folder of your choice, or set the the TemporaryDirectoryCopy class
        (located in instrumentation.instantiate) CLEAN_COPY_DIRECTORY environment
        variable to a shared directory
    """
    command: Incomplete
    verbose: Incomplete
    postprocessings: Incomplete
    instantiator: Incomplete
    last_full_output: tp.Optional[str]
    def __init__(self, folder: tp.Union[Path, str], command: tp.List[str], verbose: bool = False, clean_copy: bool = False) -> None: ...
    @staticmethod
    def register_file_type(suffix: str, comment_chars: str) -> None:
        """Register a new file type to be used for token instrumentation by providing the relevant file suffix as well as
        the characters that indicate a comment."""
    @property
    def placeholders(self) -> tp.List[Placeholder]: ...
    def __call__(self, **kwargs: tp.Any) -> tp.Any: ...

def get_last_line_as_float(output: str) -> float: ...
