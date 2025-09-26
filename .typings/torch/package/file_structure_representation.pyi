from _typeshed import Incomplete

__all__ = ['Directory']

class Directory:
    """A file structure representation. Organized as Directory nodes that have lists of
    their Directory children. Directories for a package are created by calling
    :meth:`PackageImporter.file_structure`."""
    name: Incomplete
    is_dir: Incomplete
    children: dict[str, Directory]
    def __init__(self, name: str, is_dir: bool) -> None: ...
    def _get_dir(self, dirs: list[str]) -> Directory:
        """Builds path of Directories if not yet built and returns last directory
        in list.

        Args:
            dirs (List[str]): List of directory names that are treated like a path.

        Returns:
            :class:`Directory`: The last Directory specified in the dirs list.
        """
    def _add_file(self, file_path: str):
        """Adds a file to a Directory.

        Args:
            file_path (str): Path of file to add. Last element is added as a file while
                other paths items are added as directories.
        """
    def has_file(self, filename: str) -> bool:
        """Checks if a file is present in a :class:`Directory`.

        Args:
            filename (str): Path of file to search for.
        Returns:
            bool: If a :class:`Directory` contains the specified file.
        """
    def __str__(self) -> str: ...
    def _stringify_tree(self, str_list: list[str], preamble: str = '', dir_ptr: str = '─── '):
        """Recursive method to generate print-friendly version of a Directory."""
