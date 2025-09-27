import dataclasses
import torch
import torch._export.serde.schema
import torch.fx
from _typeshed import Incomplete
from collections import defaultdict
from torch.export import graph_signature as graph_signature
from torch.onnx._internal.exporter import _dispatching as _dispatching, _registration as _registration

@dataclasses.dataclass
class ModelInfo:
    """Information about the model."""
    parameter_count: defaultdict[torch.dtype, int] = dataclasses.field(default_factory=Incomplete)
    buffer_count: defaultdict[torch.dtype, int] = dataclasses.field(default_factory=Incomplete)
    fx_node_count: int = ...
    fx_node_op_count: defaultdict[str, int] = dataclasses.field(default_factory=Incomplete)
    fx_node_target_count: defaultdict[str, int] = dataclasses.field(default_factory=Incomplete)
    dispatch_failures: list[tuple[torch.fx.Node, str]] = dataclasses.field(default_factory=list)
    inputs: dict[str, torch._export.serde.schema.TensorMeta] = dataclasses.field(default_factory=dict)
    outputs: dict[str, torch._export.serde.schema.TensorMeta] = dataclasses.field(default_factory=dict)

def _count_weights(exported_program: torch.export.ExportedProgram) -> tuple[defaultdict[torch.dtype, int], defaultdict[torch.dtype, int]]:
    """Count the size of the parameters in the exported program."""
def _format_model_info(model_info: ModelInfo) -> str:
    """Format the information about the model."""
def _get_io_specs(exported_program: torch.export.ExportedProgram) -> tuple[dict, dict]:
    """Get the input and output specs of the exported program."""
def _count_fx_targets(exported_program: torch.export.ExportedProgram) -> defaultdict[str, int]:
    """Count the number of targets for each node in the exported program."""
def analyze(exported_program: torch.export.ExportedProgram, registry: _registration.ONNXRegistry | None = None, file=None) -> None:
    """Analyze the compatibility of the exported program."""
def compare_ops(program_a: torch.export.ExportedProgram, program_b: torch.export.ExportedProgram) -> tuple[set[str], set[str]]:
    """Compare and get unique ops in two exported programs.

    Args:
        program_a: The first exported program.
        program_b: The second exported program.

    Returns:
        A tuple of two sets, where the first set contains the unique ops in the first program
        and the second set contains the unique ops in the second program.
    """
