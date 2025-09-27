from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property as cached_property
from torch._ops import OpOverload as OpOverload
from torch.distributed.device_mesh import DeviceMesh as DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec as DTensorSpec
from torch.distributed.tensor.placement_types import Placement as Placement
from torch.utils._pytree import TreeSpec as TreeSpec, tree_leaves as tree_leaves, tree_map_only as tree_map_only

ArgsType = tuple[object, ...]
KwargsType = dict[str, object]
PlacementList = list[Placement | None]
OutputSpecType = DTensorSpec | Sequence[DTensorSpec | None] | None

def _rebuild_tensor_from_dtensor_meta(arg) -> object:
    """
    This is used to propagate tensor metadata, must be under fake mode
    """
def _pretty_print_spec(spec: object) -> str: ...

@dataclass
class OpSpec:
    """
    An OpSpec describes an acceptable sharding placements of an operation, with the
    specified DTensorSpecs for both the output and the inputs.

    note: when the op return value is a single DTensor object, output_specs is
    DTensorSpec; when the return value is a tuple of Optional[DTensor],
    output_specs is a tuple of Optional[DTensorSpec].
    """
    output_specs: DTensorSpec | tuple[DTensorSpec | None, ...]
    input_specs: Sequence[DTensorSpec] | None = ...
    redistribute_cost: list[list[float]] | None = ...
    @cached_property
    def output_spec(self) -> DTensorSpec:
        """
        This function requires that the strategy have exactly one DTensorSpec as the
        output spec. If the output_specs is a tuple, we throw an exception.
        """
    @cached_property
    def mesh(self): ...
    def input_spec(self, index: int = 0) -> DTensorSpec: ...
    def __str__(self) -> str: ...

class StrategyType:
    """
    Base class type for op strategy, We have two StrategyType:
        OpStrategy and TupleStrategy
    """

class OpStrategy(StrategyType):
    """
    OpStrategy that consists of a list of sharding strategies associated with the op,
    where each strategy is an OpSpec that describes the acceptable input/output sharding.
    """
    strategies: list[OpSpec]
    def __init__(self, strategies: list[OpSpec]) -> None: ...
    def __str__(self) -> str: ...
    def max_num_shards(self) -> int:
        """
        Returns the max number of shards across all OpSpecs
        """
    @property
    def mesh(self): ...
    @property
    def mesh_shape(self): ...
    @property
    def ndim(self): ...
    @property
    def shape(self): ...

class TupleStrategy(StrategyType):
    '''
    TupleStrategy represents the output strategy of this op is a tuple of OpStrategies,
    i.e. If the output of this op is a tuple of tensors or list of tensors with possibly
    different OpStrategies, we should return a TupleStrategy that contains a tuple of
    OpStrategy, where each child represents the sharding strategy of "each element" of
    the tuple/list of tensors the op returns.

    NOTE: if the output of the op is a List[Tensor] and they share the same OpStrategy,
    then we should return a single OpStrategy instead of a TupleStrategy
    '''
    childs: Sequence[StrategyType]
    def __init__(self, childs: Sequence[StrategyType]) -> None: ...
    def child_mesh(self, index: int) -> DeviceMesh: ...
    def __str__(self) -> str: ...

@dataclass
class RuntimeSchemaInfo:
    """
    RuntimeSchemaInfo stores the operator schema related information for runtime (eager)
    execution. This is mainly used for two ways: 1. to generate hash for args to determine
    whether to re-run sharding prop or not 2. to determine if we need pytree
    """
    static_argnum: int = ...
    static_kwargkey: list[str] | None = ...
    needs_pytree: bool = ...

@dataclass
class OpSchema:
    """
    OpSchema is a data class that describes an operator input schemas, it includes
    DTensorSpecs/OpStrategies (instead of DTensor) and non-tensor args/kwargs (positional
    order preserved). It is mainly used by the DTensor's dispatching logic to perform various
    actions (i.e. sharding propagation, caching sharding decisions, redistribute, etc.)

    NOTE: this should be used as a read only data class
    TODO: make this a frozen dataclass

    Args:
        op: the operator overload we are intercepting
        args_schema: contains args except that the DTensor args have been replaced
            with its DTensorSpec or OpStrategy
        kwargs_schema: contains kwargs except that the DTensor kwargs have been replaced
            with its DTensorSpec or OpStrategy
    """
    op: OpOverload
    args_schema: ArgsType
    kwargs_schema: KwargsType
    schema_info: RuntimeSchemaInfo | None = ...
    @property
    def args_spec(self) -> tuple[DTensorSpec, ...]:
        """
        args_spec: Tuple[DTensorSpec, ...]: contains a clean list of args spec list
            with NO non-DTensor positional arguments (i.e. int/float/tuple, etc)
            mainly used by sharding propagation to propagate the output spec
        """
    @property
    def args_strategy(self) -> tuple[OpStrategy, ...]: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    has_symints = ...
    def __post_init__(self) -> None: ...
    def arg_type_tensor_or_tensor_list_like(self, arg_idx: int) -> bool: ...
    def return_type_tuple_tensor_like(self) -> bool: ...
    def return_type_tensor(self) -> bool: ...
    def get_mesh_from_args(self, validate: bool = True) -> DeviceMesh:
        '''
        This util can be used to get a mesh from the OpSchema that contains multiple
        DTensors as arguments. When `validate` is True, it will try to validate that all the
        arguments have the same mesh to avoid unexpected cross mesh errors.

        NOTE: this util currently does not handle TupleStrategy when `validate=True`,
        this is because for TupleStrategy there could be different types of checks, i.e.:
            - for stack and cat like op, we need to check within a TupleStrategy is every
              input is on the same mesh
            - for foreach like ops we need to check "zipped" inputs are on the same mesh
              for each index.
        '''
    def is_inplace_op(self) -> bool: ...
    def is_out_variant_op(self) -> bool: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def gen_fake_args(self) -> ArgsType:
        """
        gen_fake_args: generate fake args for the operator, this is mainly used
            by sharding propagation rules to generate fake args for the operator
            to run the local tensor operator and get the output spec.
        """
    def gen_fake_kwargs(self) -> KwargsType:
        """
        gen_fake_kwargs: generate fake kwargs for the operator, this is mainly used
            by sharding propagation rules to generate fake kwargs for the operator
            to run the local tensor operator and get the output spec.
        """
    def _inplace_rewrap_schema_suggestion(self, origin_schema: OpSchema) -> None: ...

@dataclass
class OutputSharding:
    """
    OutputSharding is a data class that is used by the sharding propagation,
    it could set the output_spec upon successful propagation. If needs_redistribute
    is set to True, a redistribute_schema would be returned together to indicate
    the input arguments needs to be redistributed before the op execution.

    NOTE: the redistribute_schema generated by sharding propagation should be
    exactly the same as the operator OpSchema, except the DTensorSpecs
    """
    output_spec: OutputSpecType
    redistribute_schema: OpSchema | None = ...
    needs_redistribute: bool = ...
    @cached_property
    def mesh(self): ...

@dataclass
class OpInfo:
    """
    All Runtime Op execution info are packed here
    """
    compute_mesh: DeviceMesh
    schema: OpSchema
    flat_args_schema: list[object]
    local_args: Sequence[object]
    local_kwargs: dict[str, object]
    args_tree_spec: TreeSpec | None = ...
    output_sharding: OutputSharding | None = ...
