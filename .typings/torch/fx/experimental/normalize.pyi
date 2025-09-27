import torch.fx
from .schema_type_annotation import AnnotateTypesWithSchema as AnnotateTypesWithSchema
from _typeshed import Incomplete
from torch.fx import Proxy as Proxy, Transformer as Transformer
from torch.fx.node import Argument as Argument, Node as Node, Target as Target, map_aggregate as map_aggregate
from torch.fx.operator_schemas import create_type_hint as create_type_hint, normalize_function as normalize_function, normalize_module as normalize_module
from typing import Any, Callable

class NormalizeArgs(Transformer):
    """
    Normalize arguments to Python targets. This means that
    `args/kwargs` will be matched up to the module/functional's
    signature and rewritten to exclusively kwargs in positional order
    if `normalize_to_only_use_kwargs` is true. Also populates default
    values. Does not support positional-only parameters or varargs
    parameters (*args, **kwargs).

    If the nodes have 'type' metadata, it will use it to disambiguate
    overloads. Otherwise, it will throw an error.

    Example usage:
        m = torchvision.models.resnet18()
        traced = torch.fx.symbolic_trace(m)
        traced = NormalizeArgs(traced).transform()
    """
    node_map: dict[Proxy, Node]
    normalize_to_only_use_kwargs: Incomplete
    def __init__(self, module: torch.fx.GraphModule, normalize_to_only_use_kwargs: bool = True) -> None: ...
    def run_node(self, n: Node) -> Any: ...
    def call_function(self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any], arg_types: tuple[Any, ...] | None = None, kwarg_types: dict[str, Any] | None = None): ...
    def call_module(self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]): ...

class NormalizeOperators(AnnotateTypesWithSchema):
    '''
    Normalize callsites that are different ways of "spelling" the same
    invocation into a single, canonical call. Currently supports:

    1. Normalize operators (e.g. operator.add) to the `torch` ops they
       ultimately invoke (e.g. torch.add) when it is possible to statically
       reason that

    Example usage:

        m = torchvision.models.resnet18()

        traced = torch.fx.symbolic_trace(m)

        traced = NormalizeOperators(traced).transform()
    '''
    binary_magic_method_remap: dict[Callable[[Any, Any], Any], Callable[[Any, Any], Any]]
    def call_function(self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]): ...
