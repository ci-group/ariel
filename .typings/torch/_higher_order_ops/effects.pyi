import torch
from _typeshed import Incomplete
from enum import Enum
from torch._C import DispatchKey as DispatchKey
from torch._higher_order_ops.torchbind import call_torchbind as call_torchbind
from torch._library.fake_class_registry import FakeScriptObject as FakeScriptObject
from torch._ops import HigherOrderOperator as HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode as FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode as ProxyTorchDispatchMode, disable_proxy_modes_tracing as disable_proxy_modes_tracing, track_tensor_tree as track_tensor_tree
from typing import Any

class _EffectType(Enum):
    ORDERED = 'Ordered'
OpType = torch._ops.HigherOrderOperator | torch._ops.OpOverload
SIDE_EFFECTS: Incomplete

def _register_effectful_op(op: OpType, effect: _EffectType): ...
def _deregister_effectful_op(op: OpType): ...

class WithEffects(HigherOrderOperator):
    '''
    with_effects(token, op, args, kwargs) -> (new_token, op_results)

    This HOP helps ensure ordering between side effectful ops like prints or ops
    using torchbind objects. This is needed to ensure a traced graph from
    AOTAutograd is functional so that future optimization passes do not reorder
    these operators. This is done through threading "effect tokens" through the
    graph to enforce data dependence between side effectful ops.

    The tokens are basically dummy values (torch.tensor([])). We create a token
    per "effect type", which are enumerated in the _EffectType enum.
    '''
    def __init__(self) -> None: ...
    def __call__(self, token, op: OpType, *args: tuple[Any, ...], **kwargs: dict[str, Any]) -> tuple[Any, ...]: ...

with_effects: Incomplete

def has_aliasing(op: OpType): ...
def has_effects(op, args, kwargs) -> bool: ...
def get_effect_key(op, args, kwargs) -> _EffectType | None: ...
def new_token_tensor() -> torch.Tensor: ...
def with_effects_dense(token: torch.Tensor, op: torch._ops.OpOverload, *args: tuple[Any, ...], **kwargs: dict[str, Any]) -> tuple[torch.Tensor, ...]: ...
def with_effects_fake(mode, token: torch.Tensor, op: torch._ops.OpOverload, *args: tuple[Any, ...], **kwargs: dict[str, Any]) -> tuple[torch.Tensor, ...]: ...
def with_effects_proxy(mode, token: torch.Tensor, op: torch._ops.OpOverload, *args: tuple[Any, ...], **kwargs: dict[str, Any]) -> tuple[torch.Tensor, ...]: ...
def _get_schema(op, args) -> torch.FunctionSchema: ...
def handle_effects(allow_token_discovery: bool, tokens: dict[_EffectType, torch.Tensor], op: OpType, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """
    Args:
        allow_token_discovery: Whether or not we are discovering tokens. If this
        is true, we will create a token for every side effect type seen that
        does not have a token assigned yet.  If this is false, the tokens
        should've all been created ahead of time, so we will error if there is
        no token mapping to every effect type.

        tokens: Map of effect type to tokens. This is to chain operators of the
        same effects together so that they do not get reordered in later
        optimization passes.
    """
