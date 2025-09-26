import torch
from _typeshed import Incomplete
from dataclasses import dataclass
from torch import fx as fx

logger: Incomplete

def flatten_args_detach(args):
    """
    Flatten the args into a list form and detach the tensors from computational graph.
    """
def flatten_args(args):
    """
    Flatten the args into a list form.
    """

class PipeliningShapeError(RuntimeError):
    """Shape mismatch between configured and runtime values."""

def validate_tensor_metadata(desc, expected, given) -> None: ...
def validate_tensors_metadata(desc, expected_tensors: list[torch.Tensor] | tuple[torch.Tensor, ...], actual_tensors: list[torch.Tensor] | tuple[torch.Tensor, ...]): ...
def generate_stage_to_rank_mapping(pp_size: int, num_stages: int, style: str = 'loop') -> dict[int, int]:
    """
    Compute the stage id to rank mapping for either a looped or V-style schedule.

    Most commonly num_stages == pp_size * 2, but this function can be used to
    compute the mapping for any number of stages per rank.
    """

@dataclass
class PipeInfo:
    """
    Captures information for a pipeline (`Pipe` object).
    """
    graph: fx.Graph
    num_stages: int
    has_loss_and_backward: bool
