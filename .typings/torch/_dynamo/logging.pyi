import logging
from _typeshed import Incomplete
from torch.hub import _Faketqdm as _Faketqdm, tqdm as tqdm
from typing import Callable

disable_progress: bool

def get_loggers() -> list[logging.Logger]: ...

_step_counter: Incomplete
num_steps: int
pbar: Incomplete

def get_step_logger(logger: logging.Logger) -> Callable[..., None]: ...
