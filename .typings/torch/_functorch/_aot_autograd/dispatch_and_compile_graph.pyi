import torch
from .. import config as config
from .functional_utils import assert_functional_graph as assert_functional_graph, propagate_input_mutation_stacktraces as propagate_input_mutation_stacktraces
from .schemas import AOTConfig as AOTConfig, SubclassMeta as SubclassMeta, ViewAndMutationMeta as ViewAndMutationMeta
from .traced_function_transforms import aot_dispatch_subclass as aot_dispatch_subclass, create_functionalized_fn as create_functionalized_fn, create_joint as create_joint, fn_input_mutations_to_outputs as fn_input_mutations_to_outputs, fn_prepped_for_autograd as fn_prepped_for_autograd, handle_effect_tokens_fn as handle_effect_tokens_fn
from .utils import copy_fwd_metadata_to_bw_nodes as copy_fwd_metadata_to_bw_nodes, register_buffer_assignment_hook as register_buffer_assignment_hook, root_module_when_exporting_non_strict as root_module_when_exporting_non_strict, unlift_tokens as unlift_tokens
from _typeshed import Incomplete
from torch import Tensor as Tensor
from torch._dispatch.python import enable_python_dispatcher as enable_python_dispatcher
from torch._dynamo.utils import detect_fake_mode as detect_fake_mode, lazy_format_graph_code as lazy_format_graph_code
from torch._logging import getArtifactLogger as getArtifactLogger, trace_structured as trace_structured
from torch._subclasses.functional_tensor import FunctionalTensorMode as FunctionalTensorMode
from torch.fx.experimental.proxy_tensor import make_fx as make_fx
from typing import Any

aot_graphs_log: Incomplete

def _create_graph(f, args, *, aot_config: AOTConfig) -> torch.fx.GraphModule: ...
def _detach_and_copy_item_memo(t): ...
def aot_dispatch_base_graph(flat_fn, flat_args: list[Tensor], aot_config: AOTConfig, *, fw_metadata: ViewAndMutationMeta) -> tuple[torch.fx.GraphModule, list[Any], SubclassMeta | None]: ...
def aot_dispatch_autograd_graph(flat_fn, flat_args: list[Any], aot_config: AOTConfig, *, fw_metadata: ViewAndMutationMeta) -> tuple[torch.fx.GraphModule, tuple[list[Any], list[Any]], SubclassMeta | None]: ...
