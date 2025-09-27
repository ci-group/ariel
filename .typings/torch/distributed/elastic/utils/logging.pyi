import logging
from torch.distributed.elastic.utils.log_level import get_log_level as get_log_level

def get_logger(name: str | None = None) -> logging.Logger:
    """
    Util function to set up a simple logger that writes
    into stderr. The loglevel is fetched from the LOGLEVEL
    env. variable or WARNING as default. The function will use the
    module name of the caller if no name is provided.

    Args:
        name: Name of the logger. If no name provided, the name will
              be derived from the call stack.
    """
def _setup_logger(name: str | None = None) -> logging.Logger: ...
def _derive_module_name(depth: int = 1) -> str | None:
    """
    Derives the name of the caller module from the stack frames.

    Args:
        depth: The position of the frame in the stack.
    """
