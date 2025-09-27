import torch
from _typeshed import Incomplete
from dataclasses import dataclass, field
from torch._C import DispatchKey as DispatchKey
from torch._dispatch.python import suspend_functionalization as suspend_functionalization
from torch._higher_order_ops.utils import FunctionalizeCtxWrapper as FunctionalizeCtxWrapper, HopInstance as HopInstance, _from_fun as _from_fun, _maybe_reenter_make_fx as _maybe_reenter_make_fx, _set_compilation_env as _set_compilation_env, clone_outputs_aliasing_inputs as clone_outputs_aliasing_inputs, get_dummy_aot_autograd_config as get_dummy_aot_autograd_config, prepare_fw_with_masks as prepare_fw_with_masks, reenter_make_fx as reenter_make_fx, register_fake as register_fake, save_tensors_and_symints_for_backward as save_tensors_and_symints_for_backward, saved_tensors_and_symints as saved_tensors_and_symints
from torch._ops import HigherOrderOperator as HigherOrderOperator
from torch._subclasses.functional_tensor import disable_functional_mode as disable_functional_mode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode as ProxyTorchDispatchMode, _temp_remove_metadata_torch_function_mode as _temp_remove_metadata_torch_function_mode, _temp_remove_pre_dispatch_torch_function_mode as _temp_remove_pre_dispatch_torch_function_mode, disable_proxy_modes_tracing as disable_proxy_modes_tracing, track_tensor_tree as track_tensor_tree
from torch.fx.graph_module import GraphModule as GraphModule
from torch.fx.passes.runtime_assert import insert_deferred_runtime_asserts as insert_deferred_runtime_asserts

invoke_subgraph_counter: int

@dataclass
class OutputMetadata:
    num_fw_outs: int | None = ...
    indexes_with_none: set[int] = field(default_factory=set)
    indexes_with_no_grad: set[int] = field(default_factory=set)

class InvokeSubgraphHOP(HigherOrderOperator):
    subgraph_indexes: Incomplete
    def __init__(self) -> None: ...
    def __call__(self, subgraph: GraphModule | FunctionalizeCtxWrapper, identifier: str | None, *operands): ...
    def gen_schema(self, subgraph, identifier, *operands): ...

invoke_subgraph: Incomplete

def invoke_subgraph_placeholder(func, *args, **kwargs): ...
def mark_compile_region(fn=None):
    """
    This wrapper instructs torch.compile to compile the wrapped region once and
    reuse the compiled artifact, instead of the usual way of aggressively
    inlining the function.

    Under the hood, it tells TorchDynamo to use InvokeSubgraph HOP for the
    region. For PyTorch eager, this is a no-op.
    """
def get_invoke_subgraph_cache(): ...
def trace_joint_graph(fn, fw_inputs, fw_outputs):
    """
    Naively trace out a joint graph. This simplifies the reconstruction of joint
    graph in the min-cut partitioner later on.
    """
def create_fw_bw_graph(subgraph, operands, grad_outputs=None): ...
def get_output_metadata(subgraph, *operands): ...
def trace_joint_graph_as_bwd(subgraph, num_primals, joint_operands, include_key_set, exclude_key_set):
    """
    Naively trace out a joint graph. This simplifies the reconstruction of joint
    graph in the min-cut partitioner later on.
    """

class InvokeSubgraphAutogradOp(torch.autograd.Function):
    """
    Saves the subgraph, i.e. original callable, in the forward method. And then
    traces out a joint graph in the backward. This delaying of tracing in
    backward, also called as lazy backward, ensures that the assumptions about
    the grad_out strides and tensor-subclass-ness are already accounted for.
    """
    @staticmethod
    def forward(ctx, subgraph, identifier, output_metadata, *operands): ...
    @staticmethod
    def backward(ctx, *grad_outs): ...

@invoke_subgraph.py_autograd_impl
def _(subgraph, identifier, *operands): ...
