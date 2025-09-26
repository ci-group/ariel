from _typeshed import Incomplete
from typing import Any

REPO_ROOT: Incomplete
__all__: list[str]

def __dir__() -> list[str]: ...
def materialize_lines(lines: list[str], indentation: int) -> str: ...
def gen_from_template(dir: str, template_name: str, output_name: str, replacements: list[tuple[str, Any, int]]): ...
def find_file_paths(dir_paths: list[str], files_to_exclude: set[str]) -> set[str]:
    """
    When given a path to a directory, returns the paths to the relevant files within it.

    This function does NOT recursive traverse to subdirectories.
    """
def extract_method_name(line: str) -> str:
    '''Extract method name from decorator in the form of "@functional_datapipe({method_name})".'''
def extract_class_name(line: str) -> str:
    '''Extract class name from class definition in the form of "class {CLASS_NAME}({Type}):".'''
def parse_datapipe_file(file_path: str) -> tuple[dict[str, list[str]], dict[str, str], set[str], dict[str, list[str]]]:
    """Given a path to file, parses the file and returns a dictionary of method names to function signatures."""
def parse_datapipe_files(file_paths: set[str]) -> tuple[dict[str, list[str]], dict[str, str], set[str], dict[str, list[str]]]: ...
def split_outside_bracket(line: str, delimiter: str = ',') -> list[str]:
    """Given a line of text, split it on comma unless the comma is within a bracket '[]'."""
def process_signature(line: str) -> list[str]:
    """
    Clean up a given raw function signature.

    This includes removing the self-referential datapipe argument, default
    arguments of input functions, newlines, and spaces.
    """
def get_method_definitions(file_path: str | list[str], files_to_exclude: set[str], deprecated_files: set[str], default_output_type: str, method_to_special_output_type: dict[str, str], root: str = '') -> list[str]:
    '''
    #.pyi generation for functional DataPipes Process.

    # 1. Find files that we want to process (exclude the ones who don\'t)
    # 2. Parse method name and signature
    # 3. Remove first argument after self (unless it is "*datapipes"), default args, and spaces
    '''

iterDP_file_path: str
iterDP_files_to_exclude: set[str]
iterDP_deprecated_files: set[str]
iterDP_method_to_special_output_type: dict[str, str]
mapDP_file_path: str
mapDP_files_to_exclude: set[str]
mapDP_deprecated_files: set[str]
mapDP_method_to_special_output_type: dict[str, str]

def main() -> None:
    """
    # Inject file into template datapipe.pyi.in.

    TODO: The current implementation of this script only generates interfaces for built-in methods. To generate
          interface for user-defined DataPipes, consider changing `IterDataPipe.register_datapipe_as_function`.
    """
