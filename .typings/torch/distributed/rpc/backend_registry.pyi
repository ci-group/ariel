from _typeshed import Incomplete
from typing import NamedTuple

__all__ = ['backend_registered', 'register_backend', 'construct_rpc_backend_options', 'init_backend', 'BackendValue', 'BackendType']

class BackendValue(NamedTuple):
    construct_rpc_backend_options_handler: Incomplete
    init_backend_handler: Incomplete

BackendType: Incomplete

def backend_registered(backend_name):
    """
    Checks if backend_name is registered as an RPC backend.

    Args:
        backend_name (str): string to identify the RPC backend.
    Returns:
        True if the backend has been registered with ``register_backend``, else
        False.
    """
def register_backend(backend_name, construct_rpc_backend_options_handler, init_backend_handler):
    """Registers a new RPC backend.

    Args:
        backend_name (str): backend string to identify the handler.
        construct_rpc_backend_options_handler (function):
            Handler that is invoked when
            rpc_backend.construct_rpc_backend_options(**dict) is called.
        init_backend_handler (function): Handler that is invoked when the
            `_init_rpc_backend()` function is called with a backend.
             This returns the agent.
    """
def construct_rpc_backend_options(backend, rpc_timeout=..., init_method=..., **kwargs): ...
def init_backend(backend, *args, **kwargs): ...

# Names in __all__ with no definition:
#   BackendValue
