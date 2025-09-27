import torch
from _typeshed import Incomplete
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from torch import Tensor as Tensor
from torch._prims_common import DimsType as DimsType
from torch.distributed.tensor._dtensor_spec import DTensorSpec as DTensorSpec
from torch.distributed.tensor._op_schema import OpSchema as OpSchema, OpSpec as OpSpec, OpStrategy as OpStrategy, RuntimeSchemaInfo as RuntimeSchemaInfo, StrategyType as StrategyType
from torch.distributed.tensor._ops.utils import generate_redistribute_costs as generate_redistribute_costs, normalize_dim as normalize_dim, normalize_dims as normalize_dims, prod as prod, register_op_strategy as register_op_strategy
from torch.distributed.tensor.placement_types import Placement as Placement, Replicate as Replicate, Shard as Shard
from typing import Callable

aten: Incomplete
Shape = tuple[int, ...]

@dataclass
class DimSpec:
    """Specifies how an output dimension maps to an input dimension."""
    def inputs(self) -> Iterable['DimSpec']: ...
DimMap = tuple[DimSpec, ...]

@dataclass
class Singleton(DimSpec):
    """Output dimension is a singleton."""

@dataclass
class InputDim(DimSpec):
    """Output dimension maps directly to an input dimension."""
    input_dim: int

@dataclass
class Broadcast(DimSpec):
    """Output is the broadcast of a singleton input dimension."""
    dim: DimSpec
    dim_size: int
    @classmethod
    def new(cls, dim: DimSpec, dim_size: int) -> DimSpec: ...
    def inputs(self) -> Iterable[DimSpec]: ...

@dataclass
class NewDim(DimSpec):
    """This is a new dimension created by the op."""
    size: int
    @classmethod
    def new(cls, size: int) -> DimSpec: ...

@dataclass
class Repeat(DimSpec):
    """Output dimension is the input dimension repeated n-times."""
    input_dim: DimSpec
    times: int
    @classmethod
    def new(cls, dim: DimSpec, times: int) -> DimSpec: ...
    def inputs(self) -> Iterable[DimSpec]: ...

@dataclass
class Flatten(DimSpec):
    """Flatten a set of input dimensions, ensuring right-most adjacent elements remain adjacent in the output."""
    input_dims: Sequence[DimSpec]
    @classmethod
    def new(cls, dims: Sequence[DimSpec]) -> DimSpec: ...
    def inputs(self) -> Iterable[DimSpec]: ...

@dataclass
class Split(DimSpec):
    """
    This dimension is a member of a decomposition of the input dim.

    Note that input_dim itself could be a Flattened set of input dims.
    """
    input_dim: DimSpec
    group_shape: Shape
    split_id: int
    @classmethod
    def new(cls, dim: DimSpec, group_shape: tuple[int, ...], idx: int) -> DimSpec: ...
    def inputs(self) -> Iterable[DimSpec]: ...

def dim_pad_left(ndim: int, min_dims: int) -> DimMap: ...
def dim_atleast_3d(ndim: int) -> DimMap: ...
def expand(input_shape: Shape, shape: Shape) -> DimMap:
    """Implement broadcast on multiple dimensions."""
def normalize_sizes(sizes: Shape | tuple[Shape]) -> Shape: ...
def dim_flatten(ndim: int, start_dim: int = 0, end_dim: int = -1) -> DimMap: ...
def dim_movedim(ndim: int, input: DimsType, destination: DimsType) -> DimMap: ...
def dim_repeat(ndim: int, sizes: Shape) -> DimMap: ...
def infer_size(total_size: int, sizes: Shape) -> Shape:
    '''
    One dimension input to view may be "-1".

    Infer the size of this dimension given the total_size.
    '''
def view_groups(from_size: Shape, to_size: Shape) -> DimMap:
    """
    Decompose a reshape operation into forwarding, flattening, or splitting dimensions for each output dimension.

    A view or reshape operation can be decomposed into a set of 3 types of smaller operations:
    1) Forward a dimension from input to output
    2) Flatten a set of dimensions into a single dimension
    3) Split one dimension into multiple dimensions

    view_groups identifies these operations and returns, for each output dimension, what
    is operation was performed in the input dimension. For example:

        view_groups([2, 3, 4], [2, 12]) -> (
            InputDim(0),
            Flatten((InputDim(1), InputDim(2)))
        )

    - ouptut dimension 0 maps to input dimension 0
    - output dimension 1 maps to a flattened input dimensions 1 and 2


        view_groups([2, 3], [3, 2]) -> (
            Split(Flatten((InputDim(0), InputDim(1))), (3, 2), 0),
            Split(Flatten((InputDim(0), InputDim(1))), (3, 2), 1),
        )

    - in the above, input is flattened into a single dimension and then split
      into two separate dimensions with different sizes from the input.
    """
def dim_tile(ndim: int, dims: tuple[int, ...]) -> DimMap: ...
def dim_transpose(ndim: int, dim1: int, dim2: int) -> DimMap: ...
def dim_squeeze(shape: Shape, dim: int | None = None) -> DimMap: ...
def dim_unsqueeze(ndim: int, dim: int) -> DimMap: ...
def dim_view_as_real(shape: Shape) -> DimMap: ...
def dim_reduction(ndim: int, dim_or_dims: DimsType | None, keepdim: bool) -> DimMap:
    """
    General fallback for reduction ops where Partial() does not apply.

    This will cause incoming tensor to be replicated on the reducing dimensions.
    """

dim_maps: dict[Callable[..., torch.Tensor], Callable[..., DimMap]]

def propagate_shape_and_sharding(input_src_placements: Sequence[Placement], global_input_shape: Shape, rule: DimMap, mesh_sizes: Shape, strict_view: bool = False) -> tuple[Sequence[Placement], Sequence[Placement]]:
    """
    Determine input target sharding and output sharding based on
    given global tensor shape and input source sharding.

    Sharding propagation follows mapped dimensions:
    - An output dimension that maps directly to an input dimension is sharded equally
    - An output dimension that is a flattened set of input dimensions can only be
      sharded if only the leftmost flattened dimension is sharded.
    - An output dimension that is a split of the input dimension can only be sharded
      if the leftmost split size is divisible by the mesh dimension
    """
def register_op_strategy_map(aten_op_overload: torch._ops.OpOverload, local_op_name: Callable[..., torch.Tensor], schema_info: RuntimeSchemaInfo | None = None, strict_view: bool = False) -> None:
    '''
    Helper that registers strategies for view-like operators that follow a pattern:
      (1) define the way input dims are split/combined to form output dims (dim_maps)
      (2) register a strategy for the op schema that uses the dim_map as a sharding prop rule

    strict_view: if True, we will error out if the view-operation would require resharding the input.
       Currently, this should be set to \'true\' for any "view" ops.
       We could diverge behavior for "reshape" ops which could perform a redistribute implicitly.
    '''
