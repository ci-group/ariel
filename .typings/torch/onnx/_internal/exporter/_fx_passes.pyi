import torch.export
import torch.fx
from torch.onnx._internal.exporter import _decomp as _decomp, _registration as _registration
from torch.onnx._internal.fx import passes as passes

def decompose_with_registry(exported_program: torch.export.ExportedProgram, registry: _registration.ONNXRegistry) -> torch.export.ExportedProgram:
    """Decompose the exported program with the given registry.

    This function is needed so it shows clearly on the profiler results.
    """
def insert_type_promotion_nodes(graph_module: torch.fx.GraphModule) -> None:
    """Inplace pass to insert explicit type promotion nodes, recursively through nested modules."""
def remove_assertion_nodes(graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Remove all assertion and check nodes from the FX graph"""
