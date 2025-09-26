from . import collate as collate, fetch as fetch, pin_memory as pin_memory, signal_handling as signal_handling, worker as worker
from _typeshed import Incomplete
from torch._utils import ExceptionWrapper as ExceptionWrapper

IS_WINDOWS: Incomplete
MP_STATUS_CHECK_INTERVAL: float
python_exit_status: bool
HAS_NUMPY: bool

def _set_python_exit_flag() -> None: ...
