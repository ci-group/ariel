import functools
from types import ModuleType

def _check_module_exists(name: str) -> bool:
    """Returns if a top-level module with :attr:`name` exists *without**
    importing it. This is generally safer than try-catch block around a
    `import X`. It avoids third party libraries breaking assumptions of some of
    our tests, e.g., setting multiprocessing start method when imported
    (see librosa/#747, torchvision/#544).
    """
@functools.lru_cache
def dill_available() -> bool: ...
@functools.lru_cache
def import_dill() -> ModuleType | None: ...
