import dataclasses
from _typeshed import Incomplete
from torch._higher_order_ops.base_hop import BaseHOP as BaseHOP, FunctionWithNoFreeVars as FunctionWithNoFreeVars

class InvokeQuantTracer(BaseHOP):
    def __init__(self) -> None: ...
    def __call__(self, subgraph, *operands, scheme=None, quant_options=None): ...

invoke_quant_packed: Incomplete

class InvokeQuantUnpacked(BaseHOP):
    def __init__(self) -> None: ...
    def __call__(self, subgraph, *operands, scheme=None): ...

invoke_quant: Incomplete

@dataclasses.dataclass(frozen=True, repr=True)
class InvokeQuant:
    """
    Invoke a quantization function that will be preserved as a single operator. Preservation
    as a single operator aids in pattern matching and custom lowerings.

    The operation appears as:
        torch.ops.higher_order.invoke_quant(subgraph, *args, scheme=scheme)

    Args:
        codegen_low_precision: Use observed subgraph dtypes for codegen instead of
            upcasting to fp32. Can improve performance for prologue fusion but
            requires careful testing of numerics.
    """
    codegen_low_precision: bool = ...
    def __call__(self, *args, scheme: str | None = None, **kwargs): ...
