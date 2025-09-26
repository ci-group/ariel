import torch
from _typeshed import Incomplete
from torch._export.pass_base import Argument as Argument, PassResult as PassResult, _ExportPassBaseDeprecatedDoNotUse as _ExportPassBaseDeprecatedDoNotUse
from torch._export.pass_infra.node_metadata import NodeMetadata as NodeMetadata
from torch._export.pass_infra.proxy_value import ProxyValue as ProxyValue
from torch._ops import OpOverload as OpOverload

aten: Incomplete
_NON_FUNCTIONAL_TO_FUNCTIONAL_SIDE_EFFECTFUL_FUNCS: dict[OpOverload, OpOverload]

class _FunctionalizeSideEffectfulOpsPass(_ExportPassBaseDeprecatedDoNotUse):
    """
    Functionalize ops with side effect in graph module by replacing the op with
    functional version of it. A new dependency token (`dep_token`) will be
    created and propagated through functional ops to output.
    For example:
    ```
    def f(x):
        sym_constrain_range(x.shape[0], min=1, max=3)
        return x.add(3)
    ```
    Will be transformed to:
    ```
    def f(x):
        dep_token0 = _make_dep_token()
        dep_token1 = _functional_sym_constrain_range(
            x.shape[0], min=1, max=3, dep_token=dep_token0
        )

        return x.add(3), dep_token1
    ```
    """
    _dep_token: ProxyValue | None
    _next_dep_token_index: int | None
    def __init__(self) -> None: ...
    def call(self, graph_module: torch.fx.GraphModule) -> PassResult: ...
    def call_operator(self, op: OpOverload, args: tuple[Argument, ...], kwargs: dict[str, Argument], meta: NodeMetadata) -> ProxyValue: ...
    def output(self, results: list[Argument], meta: NodeMetadata) -> ProxyValue: ...
