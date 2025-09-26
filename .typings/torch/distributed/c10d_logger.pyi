import logging
from _typeshed import Incomplete
from torch.distributed.logging_handlers import _log_handlers as _log_handlers
from torch.monitor import _WaitCounter as _WaitCounter
from typing import Any, Callable, TypeVar
from typing_extensions import ParamSpec

__all__: list[str]
_DEFAULT_DESTINATION: str

def _get_or_create_logger(destination: str = ...) -> logging.Logger: ...
def _get_logging_handler(destination: str = ...) -> tuple[logging.Handler, str]: ...

_c10d_logger: Incomplete

def _get_msg_dict(func_name, *args, **kwargs) -> dict[str, Any]: ...
_T = TypeVar('_T')
_P = ParamSpec('_P')

def _exception_logger(func: Callable[_P, _T]) -> Callable[_P, _T]: ...
def _time_logger(func: Callable[_P, _T]) -> Callable[_P, _T]: ...
