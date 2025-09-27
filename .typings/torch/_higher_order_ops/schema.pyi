import torch
import torch.utils._pytree as pytree
from _typeshed import Incomplete
from dataclasses import dataclass
from torch.fx.node import Target as Target
from typing import Any

@dataclass(frozen=True)
class HopArgumentInfo:
    name: str
    example_value: Any
    default_value: Any
    is_mutated: bool
    kw_only: bool

class HopArgumentInfoGen:
    @staticmethod
    def from_example(example_value: Any, *, name: str = '', default_value: Any | None = None, is_mutated: bool = False, kw_only: bool = False) -> HopArgumentInfo: ...

class CTypeGen:
    convert_to_base_ty: Incomplete
    @staticmethod
    def from_example(obj: Any) -> Any: ...

class CArgumentGen:
    @staticmethod
    def from_hop_argument_info(arg_idx: int, arg_info: HopArgumentInfo, is_output: bool = False) -> Any: ...

class HopSchemaGenerator:
    arg_infos: list[HopArgumentInfo]
    example_outputs: list[Any]
    schema_tree_spec: pytree.TreeSpec | None
    hop: Incomplete
    def __init__(self, hop: torch._ops.HigherOrderOperator) -> None: ...
    def add_arg(self, name: str, example_value: Any, default_value: Any | None = None, is_mutated: bool = False, kw_only: bool = False) -> None: ...
    def add_output(self, output: Any) -> None: ...
    def add_schema_tree_spec(self, *args: Any, **kwargs: Any) -> None:
        """schema tree spec is the tree spec from flattening all inputs to the hop with pytree.tree_flatten
        Since torch.FunctionSchema only have proper mutation/alias support for flattened inputs, we need
        to store the tree spec in order to reconstruct the inputs to the hop.
        """
    def gen_schema(self) -> torch._C.FunctionSchema: ...

class CFunctionSchemaGen:
    '''
    Note: [HigherOrderOperator schema generation]
    Each invocation of a HigherOrderOperator will have a different schema.
    For example, the schema of torch.cond varies depending on the true_fn and
    false_fn. So we need a way to generate the schema for each invocation of a HOP.

    We want to enforce the following invariants for HOP\'s schema:
        1. Flattened inputs. There should be no pytree structure in it.
        2. Flattened outputs. Note even if the hop returns a single value, it should be wrapped as a tuple.
        3. No aliasing. This includes inp-inp aliasing, inp-out aliasing and out-out aliasing.

    By enforcing these invariants, we could make HOP\'s schema meets the requirement of schema parser
    and makes hop easier to handle downstream. For example, suppose we have an invoke_quant_test HOP:

    class GraphModule(torch.nn.Module):
        def forward(self, l_x_, l_y_):
            subgraph_0 = self.subgraph_0
            invoke_quant_test = torch.ops.higher_order.invoke_quant_test(subgraph_0, l_x_, l_y_, scheme = \'nf4\');

        class subgraph_0(torch.nn.Module):
            def forward(self, l_x_, l_y_):
                add_ = l_x_.add_(1)
                matmul = l_x_ @ l_y_
                sin = matmul.sin()
                child = sin.cos()
                child_1 = l_x_ + l_y_
                child_2 = l_x_ - l_y_
                child_3 = l_x_ @ l_y_
                return (child, child_1, child_2, child_3)

    By encoding the inputs of hop into a list of HopArgumentInfo and output as a single HopArgumentInfo,
    we would get the following schema:
        invoke_quant_test(Any arg0, Tensor(!) arg1, Tensor arg2, str scheme="\\"nf4\\"") -> (Tensor, Tensor, Tensor, Tensor)
    '''
    @staticmethod
    def from_hop_argument_info(op_name: str, inp_argument_info: list[HopArgumentInfo], out_argument_info: HopArgumentInfo, schema_tree_spec: pytree.TreeSpec | None) -> Any: ...

class HopSchema(torch._C.FunctionSchema):
    tree_spec: Incomplete
    is_vararg: Incomplete
    is_varret: Incomplete
    def __init__(self, name: str, overload_name: str, arguments: list[torch._C.Argument], returns: list[torch._C.Argument], is_vararg: bool, is_varret: bool, schema_tree_spec: pytree.TreeSpec | None) -> None: ...
    def __deepcopy__(self, memo: Any) -> HopSchema: ...

def find_hop_schema(gm: torch.fx.GraphModule, target: Target) -> list[torch._C.FunctionSchema]: ...
