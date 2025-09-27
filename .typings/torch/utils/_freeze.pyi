import types
from _typeshed import Incomplete
from dataclasses import dataclass
from pathlib import Path

PATH_MARKER: str
MAIN_INCLUDES: str
MAIN_PREFIX_TEMPLATE: str
FAKE_PREFIX: Incomplete
MAIN_SUFFIX: str
DENY_LIST: Incomplete
NUM_BYTECODE_FILES: int

def indent_msg(fn): ...

@dataclass
class FrozenModule:
    module_name: str
    c_name: str
    size: int
    bytecode: bytes

class Freezer:
    frozen_modules: list[FrozenModule]
    indent: int
    verbose: bool
    def __init__(self, verbose: bool) -> None: ...
    def msg(self, path: Path, code: str): ...
    def write_bytecode(self, install_root) -> None:
        """
        Write the `.c` files containing the frozen bytecode.

        Shared frozen modules evenly across the files.
        """
    def write_main(self, install_root, oss, symbol_name) -> None:
        """Write the `main.c` file containing a table enumerating all the frozen modules."""
    def write_frozen(self, m: FrozenModule, outfp):
        """Write a single frozen module's bytecode out to a C variable."""
    def compile_path(self, path: Path, top_package_path: Path):
        """Entry point for compiling a Path object."""
    @indent_msg
    def compile_package(self, path: Path, top_package_path: Path):
        """Compile all the files within a Python package dir."""
    def get_module_qualname(self, file_path: Path, top_package_path: Path) -> list[str]: ...
    def compile_string(self, file_content: str) -> types.CodeType: ...
    @indent_msg
    def compile_file(self, path: Path, top_package_path: Path):
        """
        Compile a Python source file to frozen bytecode.

        Append the result to `self.frozen_modules`.
        """

def main() -> None: ...
