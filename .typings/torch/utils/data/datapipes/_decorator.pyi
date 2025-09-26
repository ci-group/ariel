from _typeshed import Incomplete
from torch.utils.data.datapipes._typing import _DataPipeMeta as _DataPipeMeta
from torch.utils.data.datapipes.datapipe import IterDataPipe as IterDataPipe, MapDataPipe as MapDataPipe
from typing import Any, Callable

class functional_datapipe:
    name: str
    enable_df_api_tracing: Incomplete
    def __init__(self, name: str, enable_df_api_tracing: bool = False) -> None:
        """
        Define a functional datapipe.

        Args:
            enable_df_api_tracing - if set, any returned DataPipe would accept
            DataFrames API in tracing mode.
        """
    def __call__(self, cls): ...

_determinism: bool

class guaranteed_datapipes_determinism:
    prev: bool
    def __init__(self) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None: ...

class non_deterministic:
    cls: type[IterDataPipe] | None
    deterministic_fn: Callable[[], bool]
    def __init__(self, arg: type[IterDataPipe] | Callable[[], bool]) -> None: ...
    def __call__(self, *args, **kwargs): ...
    def deterministic_wrapper_fn(self, *args, **kwargs) -> IterDataPipe: ...

def argument_validation(f): ...

_runtime_validation_enabled: bool

class runtime_validation_disabled:
    prev: bool
    def __init__(self) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None: ...

def runtime_validation(f): ...
