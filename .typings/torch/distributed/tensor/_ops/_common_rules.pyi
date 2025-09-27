from torch.distributed.tensor._dtensor_spec import DTensorSpec as DTensorSpec, TensorMeta as TensorMeta
from torch.distributed.tensor._op_schema import OpSchema as OpSchema, OutputSharding as OutputSharding
from torch.distributed.tensor._ops.utils import prod as prod
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset as compute_local_shape_and_global_offset

def _replace_char_in_str(string: str, new_char: str, idx: int) -> str: ...
def _gen_reshard_suggestions(op_schema: OpSchema, input_dims: list[str], input_specs: tuple[DTensorSpec, ...], dim_to_sharding: dict[str, int], pending_sum: list[int]) -> OutputSharding: ...
def einop_rule(equation: str, op_schema: OpSchema, *, linearity: bool = False, enforce_sharding: dict[str, int] | None = None) -> OutputSharding:
    """
    Propagate the sharding of inputs to output for ops whose data moves according to einsum notation.

    This is mostly borrowed from @zdevito's sharding simulator. Examples:
        mk,kn->mn - einsum
        ij,ij->ij - addition
        ij,j->ij - broadcasted addition
        ij->i - reduction
    Other ops could use this propagation algorithm when applied, note
    that einsum propagation only deal with list of specs (DTensor specs)
    as it only works on list of tensors!

    linearity in einop_rule means that the calling op `f` follows this rule:
        f(a + b) = f(a) + f(b)

    In this case we can propagate the partial sum, note that linearity in einop
    only applies to partial sum, not other operations like min/max (which are
    associative but not linear).
    """
def pointwise_rule(op_schema: OpSchema, linearity: bool = False) -> OutputSharding:
    """
    Propagate the sharding for pointwise operations.

    Examples:
        ij,ij->ij - addition/mul
        ij,j->ij - broadcasted addition
    """
