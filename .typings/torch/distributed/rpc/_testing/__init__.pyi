from . import faulty_agent_backend_registry as faulty_agent_backend_registry
from torch._C._distributed_rpc_testing import FaultyTensorPipeAgent as FaultyTensorPipeAgent, FaultyTensorPipeRpcBackendOptions as FaultyTensorPipeRpcBackendOptions

def is_available() -> bool: ...
