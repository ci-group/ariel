import torch
from torch.fx.passes.fake_tensor_prop import FakeTensorProp as FakeTensorProp
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner as CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupport as OperatorSupport
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS as CALLABLE_NODE_OPS

class CudaGraphsSupport(OperatorSupport):
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool: ...

def partition_cudagraphs(gm, inputs):
    """
    Partition an FX graph into sub-GraphModules that can be validly run under
    CUDA graphs.  For a subgraph to be runnable under CUDA, all of the operations
    must involve CUDA tensors only/
    """
