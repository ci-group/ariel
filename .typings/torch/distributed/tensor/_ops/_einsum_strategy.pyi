from dataclasses import dataclass
from torch.distributed.device_mesh import DeviceMesh as DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec as DTensorSpec
from torch.distributed.tensor._op_schema import OpSpec as OpSpec, OpStrategy as OpStrategy
from torch.distributed.tensor.placement_types import Partial as Partial, Placement as Placement, Replicate as Replicate, Shard as Shard

@dataclass
class EinsumDims:
    contracting_dims: list[str]
    batch_dims: list[str]
    lhs_out_only_dims: list[str]
    rhs_out_only_dims: list[str]
    @classmethod
    def parse_equation(cls, equation: str) -> tuple[list[str], str]:
        """
        Parse the einsum equation str to input dim chars and output dim char
        """
    @classmethod
    def parse_dims(cls, input_dims: list[str], output_dim: str) -> EinsumDims:
        """
        Parse the dims and extract the contracting, batch, and free dimensions
        for the left and right hand sides.
        """

def gen_einsum_strategies(equation: str, mesh: DeviceMesh, *, linearity: bool = False) -> OpStrategy:
    """
    Generate a strategy list for the ops that follow einsum style notation.
    """
