from _typeshed import Incomplete
from torch.distributed.checkpoint.logging_handlers import DCP_LOGGER_NAME as DCP_LOGGER_NAME
from typing import Any, Callable, TypeVar
from typing_extensions import ParamSpec

logger: Incomplete
__all__: list[str]
_dcp_logger: Incomplete
_T = TypeVar('_T')
_P = ParamSpec('_P')

def _msg_dict_from_dcp_method_args(*args, **kwargs) -> dict[str, Any]:
    """
    Extracts log data from dcp method args
    """
def _get_msg_dict(func_name, *args, **kwargs) -> dict[str, Any]: ...
def _dcp_method_logger(log_exceptions: bool = False, **wrapper_kwargs: Any) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """This method decorator logs the start, end, and exception of wrapped events."""
def _init_logger(rank: int): ...
