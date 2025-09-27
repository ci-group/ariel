import abc
import torch.fx
import torch.fx.graph
from abc import ABC, abstractmethod
from typing import Any, Callable
from typing_extensions import TypeAlias

class CustomGraphPass(ABC, metaclass=abc.ABCMeta):
    """
    Implement this interface for custom Graph passes:

    1) The __call__() method contains the implementation of the custom pass.

    2) The uuid() method enables inductor to cache compiled graphs when your custom
    passes are applied. This method can return any identifier as long as it uniquely
    identifies your implementation (and can be pickled). The caching logic includes this
    identifier in its key calculation, i.e., any new value will effectively invalidate
    existing entries. We expect custom passes would typically depend purely on the
    textual representation of the implementation. In that case, we recommend using the
    'get_hash_for_files' helper below to compute a unique hash from the contents of a
    static list of source files, i.e., the source(s) containing the custom pass
    implementation. That approach ensures that any change to the implementation will
    mean a new uuid.

    ** IMPORTANT ** If your custom pass's behavior depends on some external state, then
    you'll need to implement something more complicated (or disable caching).

    EXAMPLE:

    class MyCustomGraphPass(CustomGraphPass):
        def __call__(self, graph: torch.fx.graph.Graph) -> None:
            # my custom graph optimization pass
            #     ...

        def uuid(self) -> Optional[Any]:
            return get_hash_for_files((__file__,))

    """
    @abstractmethod
    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        """
        Implementation of the custom pass.
        """
    @abstractmethod
    def uuid(self) -> Any | None:
        """
        Return an ID to uniquely identify your custom pass implementation. Return None
        to skip inductor code caching entirely.
        """

class CustomGraphModulePass(ABC, metaclass=abc.ABCMeta):
    """
    Implement this interface for custom Graph passes:

    1) The __call__() method contains the implementation of the custom pass.

    2) The uuid() method enables inductor to cache compiled graphs when your custom
    passes are applied. This method can return any identifier as long as it uniquely
    identifies your implementation (and can be pickled). The caching logic includes this
    identifier in its key calculation, i.e., any new value will effectively invalidate
    existing entries. We expect custom passes would typically depend purely on the
    textual representation of the implementation. In that case, we recommend using the
    'get_hash_for_files' helper below to compute a unique hash from the contents of a
    static list of source files, i.e., the source(s) containing the custom pass
    implementation. That approach ensures that any change to the implementation will
    mean a new uuid.
    """
    @abstractmethod
    def __call__(self, gm: torch.fx.GraphModule) -> None:
        """
        Implementation of the custom pass.
        """
    @abstractmethod
    def uuid(self) -> Any | None:
        """
        Return an ID to uniquely identify your custom pass implementation. Return None
        to skip inductor code caching entirely.
        """
CustomGraphPassType: TypeAlias = CustomGraphPass | Callable[[torch.fx.graph.Graph], None] | None

def get_hash_for_files(paths: tuple[str], extra: str = '') -> bytes:
    """
    Helper to compute a unique string by hashing the contents of a list of files.
    """
