import torch
from _typeshed import Incomplete
from torch.distributed.tensor.parallel.style import ParallelStyle
from torch.export import ExportedProgram
from torch.export.exported_program import ExportGraphSignature
from torch.fx.passes.infra.pass_base import PassBase, PassResult

__all__ = ['tensor_parallel_transformation']

def tensor_parallel_transformation(exported_program: ExportedProgram, rank: int, world_size: int, device_type: str, parallel_strategies: dict[str, ParallelStyle]) -> ExportedProgram:
    """
    The entry point function to perform graph transformations on an exported program
    to transform a single-device graph into a tensor parallel graph.

    .. warning::
        This API is experimental and subject to change.
    """

class _TensorParallelTransformPass(PassBase):
    """
    This pass is responsible for transforming a single-device graph into a tensor parallel
    graph. It will mark the OpSpec of each node in the graph, partition the graph into
    distributed graph, then shard the parameters/buffers accordingly.
    """
    rank: Incomplete
    mesh: Incomplete
    state_dict: dict[str, torch.Tensor]
    graph_signature: Incomplete
    parallel_strategies: Incomplete
    def __init__(self, rank: int, world_size: int, device_type: str, state_dict: dict[str, torch.Tensor], graph_signature: ExportGraphSignature, parallel_strategies: dict[str, ParallelStyle]) -> None: ...
    def call(self, graph_module) -> PassResult: ...
