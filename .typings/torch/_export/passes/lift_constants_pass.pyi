import collections
import torch
from _typeshed import Incomplete
from torch._export.verifier import SpecViolationError as SpecViolationError
from torch._guards import detect_fake_mode as detect_fake_mode
from torch._library.fake_class_registry import FakeScriptObject as FakeScriptObject
from torch._subclasses.fake_tensor import unset_fake_temporarily as unset_fake_temporarily
from torch.export.exported_program import ArgumentSpec as ArgumentSpec, CustomObjArgument as CustomObjArgument, ExportGraphSignature as ExportGraphSignature, InputKind as InputKind, InputSpec as InputSpec, TensorArgument as TensorArgument
from torch.fx._symbolic_trace import _ConstantAttributeType as _ConstantAttributeType
from torch.fx.graph_module import _get_attr as _get_attr
from typing import Any

log: Incomplete

class ConstantAttrMap(collections.abc.MutableMapping):
    """A mapping class that understands how to use module constants (tensors,
    ScriptObjects, FakeScriptObjects) as keys. We store tensors and FakeScriptObjects normally,
    but ScriptObjects are stored by hash, because different torch.ScriptObjects can point to
    the same underlying value (but we guarantee that they will `hash()` to the same value
    if that's the case).
    """
    _constant_attrs: dict[int | torch.Tensor | FakeScriptObject | torch.utils._pytree.TreeSpec, list[Any]]
    _script_object_map: dict[int, torch.ScriptObject]
    def __init__(self) -> None: ...
    def __getitem__(self, key: _ConstantAttributeType) -> Any: ...
    def __setitem__(self, key: _ConstantAttributeType, value): ...
    def add(self, key: _ConstantAttributeType, value: Any) -> None: ...
    def __delitem__(self, key: _ConstantAttributeType): ...
    def __iter__(self): ...
    def __len__(self) -> int: ...
    def __contains__(self, key: object) -> bool: ...

def get_constant_fqn(node: torch.fx.Node, constant_name: str) -> str: ...
def _get_first_fqn(const_attrs: ConstantAttrMap, key: _ConstantAttributeType) -> Any: ...
def _unused_constant(node: torch.fx.Node) -> list[torch.fx.Node] | None:
    """
    If there is a tensor constant created while tracing, here is how the graph
    looks like:

        %_tensor_constant0 : [num_users=1] = get_attr[target=_tensor_constant0]
        %lift_fresh_copy : [num_users=1] = call_function[target=torch.ops.aten.lift_fresh_copy.default](args = (%_tensor_constant0,))
        %detach_ : [num_users=?] = call_function[target=torch.ops.aten.detach_.default](args = (%lift_fresh_copy,))

    To check to see if the tensor constant is being used, we want to traverse to
    the detach node to see if it's actually being used.

    This function returns None if this constant is being used, otherwise it returns the
    lift_fresh and detach node to be removed later.
    """
def lift_constants_pass(gm: torch.fx.GraphModule, graph_signature: ExportGraphSignature, constant_attrs: ConstantAttrMap) -> dict[str, _ConstantAttributeType]:
    """
    Takes a graph module, graph signature, and modifies them implace to lift any
    constants (tensors or custom classes) as inputs to the graph. Returns a
    dictionary of names to constants.

    Arguments:
        gm (torch.fx.GraphModule): The graph module containing the graph and constants to lift.
        graph_signature (ExportGraphSignature): This graph signature will be
            mutated to add additional CONSTANT_TENSOR and CUSTOM_OBJ inputs.
        constant_attrs (ConstantAttr): A mapping from a constant value to its
            fully-qualified path in `gm`. This is used to maintain consistent
            location of constants between the original module and the exported
            version.

    Returns:
        A dictionary of fqn => constant value.
    """
def rewrite_script_object_meta(gm: torch.fx.GraphModule) -> dict[str, _ConstantAttributeType]:
    '''When tracing, we produce a graph with FakeScriptObject in the
    meta["val"].

    For now, we rewrie meta["val"] to be a placeholder CustomObjArgument
    '''
def _materialize_and_lift_constants(gm: torch.fx.GraphModule, export_graph_signature: ExportGraphSignature, constant_attrs: ConstantAttrMap) -> dict[str, _ConstantAttributeType]: ...
