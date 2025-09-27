import torch
import torch.utils._pytree as pytree
from _typeshed import Incomplete
from torch._library.fake_class_registry import FakeScriptObject as FakeScriptObject
from torch._logging import getArtifactLogger as getArtifactLogger
from torch._subclasses.fake_tensor import FakeTensor as FakeTensor
from torch._subclasses.functional_tensor import FunctionalTensor as FunctionalTensor
from torch.fx.experimental._backward_state import BackwardState as BackwardState
from torch.fx.experimental.proxy_tensor import py_sym_types as py_sym_types
from typing import Any, Callable

KNOWN_TYPES: Incomplete
original_zip = zip
aot_graphs_effects_log: Incomplete

def strict_zip(*iterables, strict: bool = True, **kwargs): ...
def _get_symint_hints(exprs):
    """
    Get the hints of a list/tuple of int/SymInt.
    """
def partial_flatten_asdict(obj: Any) -> Any: ...
def normalize_as_list(x): ...
def _get_autocast_states(): ...
def make_boxed_func(f): ...
def make_boxed_compiler(compiler): ...
def call_func_at_runtime_with_args(f, args: tuple[Any] | list[Any], steal_args: bool = False, disable_amp: bool = False): ...

class PytreeThunk:
    spec: pytree.TreeSpec | None
    is_simple: bool | None
    is_really_simple: bool | None
    def set(self, spec: pytree.TreeSpec) -> None: ...
    def unflatten(self, x: list[Any]) -> Any: ...

def create_tree_flattened_fn(fn, args, kwargs=None) -> tuple[Callable, PytreeThunk]: ...
def maybe_to_fresh_input(idx, t, meta): ...
def is_with_effects(node): ...
def is_with_effects_op(node, op): ...
def unlift_tokens(fw_module, fw_metadata, aot_config, bw_module=None) -> None: ...
def root_module_when_exporting_non_strict(flat_fn): ...
def copy_fwd_metadata_to_bw_nodes(fx_g):
    """
    Input: `fx_g` which contains the joint fwd+bwd FX graph created by
    aot_autograd.

    This function walks the graph and copies over metadata from forward nodes
    to backward nodes, using the `seq_nr` field as a one-to-many mapping
    from forward node to backward node. This metadata is useful for performance
    profiling and debugging.
    """
def register_buffer_assignment_hook(mod, assigned_buffers):
    """
    Register a hook that intercepts buffer assignments.
    This is used to detect when a buffer is assigned to, and then we can
    map that buffer to the corresponding proxy node in the graph.
    """
def contain_metadata_mutation_ops(module: torch.fx.GraphModule) -> bool:
    """
    Checks if the module contains any metadata mutation ops.
    """
def get_cuda_generator_meta_val(device_idx: int):
    """
    Get a generator value to use as a meta val

    newly cloned generator will not contain tensors. it is only Generators that are
    registered to a CUDAGraph that contain tensors. since this does not contain Tensor
    it is fine to use in the meta.
    """
def top_saved_tensors_hooks(): ...
def saved_tensors_hooks_are_inlineable(hooks) -> bool: ...
