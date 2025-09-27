import torch
from . import config as config
from _typeshed import Incomplete
from torch._dynamo.utils import dynamo_timed as dynamo_timed, lazy_format_graph_code as lazy_format_graph_code
from torch._functorch.aot_autograd import MutationType as MutationType
from torch._functorch.compile_utils import fx_graph_cse as fx_graph_cse
from torch._inductor.constant_folding import constant_fold as constant_fold, replace_node_with_constant as replace_node_with_constant
from torch._inductor.freezing_utils import enter_freezing as enter_freezing, record_has_frozen_params as record_has_frozen_params
from torch._inductor.fx_passes.freezing_patterns import freezing_passes as freezing_passes
from torch._inductor.fx_passes.post_grad import view_to_reshape as view_to_reshape
from typing import Any

aten: Incomplete
prims: Incomplete
log: Incomplete

def replace_params_with_constants(gm: torch.fx.GraphModule, flat_params: list[Any], fw_metadata: torch._functorch.aot_autograd.ViewAndMutationMeta) -> list[int]:
    """
    Replaces the parameters of a PyTorch GraphModule with constants wherever possible.
    Returns a list of indices representing the input parameters that were not converted to constants.
    """
def freeze(dynamo_gm: torch.fx.GraphModule, aot_autograd_gm: torch.fx.GraphModule, example_inputs: list[torch._subclasses.FakeTensor]) -> tuple[torch.fx.GraphModule, list[int]]:
    """
    Inlines parameters that are not mutated into constants and optimizes the graph through constant propagation
    and other techniques. If enabled, the function also discards the original parameters of the module for memory efficiency.

    Assumes that this function is run in dynamo tracing post aot_autograd.

    Args:
        dynamo_gm (torch.fx.GraphModule): The Dynamo constructed GraphModule.
        aot_autograd_gm (torch.fx.GraphModule): The aot_autograd constructed GraphModule to be frozen.
        example_inputs (List[torch.Tensor]): A list of example input tensors to be used in the freezing process.

    Returns:
        Tuple[torch.fx.GraphModule, List[int]]: A tuple containing the frozen GraphModule and a list of indices
        of the inputs that were preserved (not turned into constants).
    """
def _freeze(dynamo_gm: torch.fx.GraphModule, aot_autograd_gm: torch.fx.GraphModule, example_inputs: list[torch._subclasses.FakeTensor]) -> tuple[torch.fx.GraphModule, list[int]]: ...

class ErasedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem, name, owning_mod): ...
    erased_name: Incomplete
    owning_mod_ref: Incomplete
    def __init__(self, elem, name: str | None, mod) -> None: ...
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None) -> None: ...

def invalidate_eager_modules() -> None: ...
def discard_traced_gm_params(mod: torch.fx.GraphModule): ...
def enforce_output_layout(gm: torch.fx.GraphModule):
    """
    Make sure the output node's layout does not change due to compiler optimizations
    by adding aten.as_strided nodes with the expected strides.

    Only used for inference so we can assume all graph outputs are model outputs.
    """
def enforce_as_strided_input_layout(gm: torch.fx.GraphModule):
    """
    Make sure the as_strided node's input's layout does not change due to compiler
    optimizations, because the as_strided strides info depends on input tensor stride info.
    """
def convert_conv_weights_to_channels_last(gm: torch.fx.GraphModule):
    """
    Convert 4d convolution weight tensor to channels last format.

    This pass is performed before freezing so the added nodes can be constant
    folded by freezing.
    """
