import logging
from torch.distributed._shard.sharded_tensor.logging_handlers import _log_handlers as _log_handlers

__all__: list[str]

def _get_or_create_logger() -> logging.Logger: ...
def _get_logging_handler(destination: str = 'default') -> tuple[logging.Handler, str]: ...
