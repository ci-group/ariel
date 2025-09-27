from .api import *
from . import api
from .backend_registry import BackendType as BackendType
from .options import TensorPipeRpcBackendOptions as TensorPipeRpcBackendOptions

__all__ = ['is_available', 'init_rpc', 'BackendType', 'TensorPipeRpcBackendOptions', 'shutdown', 'get_worker_info', 'remote', 'rpc_sync', 'rpc_async', 'RRef', 'AllGatherStates', 'method_factory', 'new_method', 'backend_registered', 'register_backend', 'construct_rpc_backend_options', 'init_backend', 'BackendValue', 'BackendType']

def is_available() -> bool: ...
def init_rpc(name, backend=None, rank: int = -1, world_size=None, rpc_backend_options=None) -> None:
    '''
        Initializes RPC primitives such as the local RPC agent
        and distributed autograd, which immediately makes the current
        process ready to send and receive RPCs.

        Args:
            name (str): a globally unique name of this node. (e.g.,
                ``Trainer3``, ``ParameterServer2``, ``Master``, ``Worker1``)
                Name can only contain number, alphabet, underscore, colon,
                and/or dash, and must be shorter than 128 characters.
            backend (BackendType, optional): The type of RPC backend
                implementation. Supported values is
                ``BackendType.TENSORPIPE`` (the default).
                See :ref:`rpc-backends` for more information.
            rank (int): a globally unique id/rank of this node.
            world_size (int): The number of workers in the group.
            rpc_backend_options (RpcBackendOptions, optional): The options
                passed to the RpcAgent constructor. It must be an agent-specific
                subclass of :class:`~torch.distributed.rpc.RpcBackendOptions`
                and contains agent-specific initialization configurations. By
                default, for all agents, it sets the default timeout to 60
                seconds and performs the rendezvous with an underlying process
                group initialized using ``init_method = "env://"``,
                meaning that environment variables ``MASTER_ADDR`` and
                ``MASTER_PORT`` need to be set properly. See
                :ref:`rpc-backends` for more information and find which options
                are available.
        '''

# Names in __all__ with no definition:
#   AllGatherStates
#   BackendValue
#   RRef
#   backend_registered
#   construct_rpc_backend_options
#   get_worker_info
#   init_backend
#   method_factory
#   new_method
#   register_backend
#   remote
#   rpc_async
#   rpc_sync
#   shutdown
