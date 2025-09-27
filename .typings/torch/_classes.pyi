import types
from _typeshed import Incomplete

class _ClassNamespace(types.ModuleType):
    name: Incomplete
    def __init__(self, name) -> None: ...
    def __getattr__(self, attr): ...

class _Classes(types.ModuleType):
    __file__: str
    def __init__(self) -> None: ...
    def __getattr__(self, name): ...
    @property
    def loaded_libraries(self): ...
    def load_library(self, path) -> None:
        """
        Loads a shared library from the given path into the current process.

        The library being loaded may run global initialization code to register
        custom classes with the PyTorch JIT runtime. This allows dynamically
        loading custom classes. For this, you should compile your class
        and the static registration code into a shared library object, and then
        call ``torch.classes.load_library('path/to/libcustom.so')`` to load the
        shared object.

        After the library is loaded, it is added to the
        ``torch.classes.loaded_libraries`` attribute, a set that may be inspected
        for the paths of all libraries loaded using this function.

        Args:
            path (str): A path to a shared library to load.
        """

classes: Incomplete
