import torch
import torch.utils._pytree as pytree
from .collect_metadata_analysis import coerce_tangent_and_suggest_memory_format as coerce_tangent_and_suggest_memory_format
from .schemas import BackwardSignature as BackwardSignature, GraphSignature as GraphSignature, InputAliasInfo as InputAliasInfo, MemoryFormatMeta as MemoryFormatMeta, OutputAliasInfo as OutputAliasInfo, OutputType as OutputType, ViewAndMutationMeta as ViewAndMutationMeta
from .utils import strict_zip as strict_zip
from torch import Tensor as Tensor
from torch._C._dynamo.guards import compute_overlapping_tensors as compute_overlapping_tensors
from torch._functorch._aot_autograd.schemas import PlainTensorMeta as PlainTensorMeta
from torch._guards import StorageOverlap as StorageOverlap
from torch._subclasses.functional_tensor import FunctionalTensor as FunctionalTensor
from torch.fx.experimental.symbolic_shapes import is_concrete_int as is_concrete_int
from typing import Any

zip = strict_zip

def remove_dupe_metadata(m: ViewAndMutationMeta, keep_arg_mask: list[bool], add_dupe_map: list[int]) -> ViewAndMutationMeta: ...
def create_synthetic_base_metadata(m: ViewAndMutationMeta, synthetic_base_info: list[int | tuple[int, torch.Tensor]], outer_args: list[Any], inner_args: list[Any]) -> tuple[ViewAndMutationMeta, list[int]]: ...
def compute_overlapping_inputs(aot_config, fwd_inputs, aliased_input_indices): ...
def _graph_input_names(gm): ...
def _graph_output_names(gm): ...
def create_graph_signature(fx_g: torch.fx.GraphModule, fw_metadata: ViewAndMutationMeta, in_spec: pytree.TreeSpec, out_spec: pytree.TreeSpec, *, user_args_flat: list[Tensor], params_and_buffers_flat: list[Tensor], param_names: list[str], buffer_names: list[str], trace_joint: bool, num_user_fw_outs: int | None, loss_index: int | None) -> GraphSignature: ...
