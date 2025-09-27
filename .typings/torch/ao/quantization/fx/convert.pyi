import torch
from .custom_config import ConvertCustomConfig
from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.ao.quantization.quant_type import QuantType
from torch.fx import GraphModule
from torch.fx.graph import Graph, Node
from typing import Any

__all__ = ['convert', 'convert_custom_module', 'convert_standalone_module', 'convert_weighted_module']

def convert_standalone_module(node: Node, modules: dict[str, torch.nn.Module], model: torch.fx.GraphModule, is_reference: bool, backend_config: BackendConfig | None) -> None:
    """Converts a observed standalone module to a quantized standalone module by calling
    the fx convert api, currently using the same `is_reference` flag as parent, but we may
    changing this behavior in the future (e.g. separating quantization and lowering for
    standalone module as well)

    Args:
      - node: The call_module node of the observed standalone module
      - modules: named_module of original model
      - model: original model
      - is_reference: a flag from parent provided by user to decide if we want to
        produce a reference model or a fbgemm/qnnpack model
      - backend_config: backend configuration of the target backend of quantization
    """
def convert_weighted_module(node: Node, modules: dict[str, torch.nn.Module], observed_node_names: set[str], node_name_to_qconfig: dict[str, QConfigAny], backend_config: BackendConfig, is_decomposed: bool = False, is_reference: bool = False) -> None:
    """Convert a weighted module to reference quantized module in the model
    If the QConfig of a QAT module is not set, the module will still be converted to
    a float module.

    Args:
      - node: The call_module node of the observed standalone module
      - modules: named_module of original model
      - observed_node_names: names for the set of observed fx node, we can skip
        this conversion if the node is not observed
    """
def convert_custom_module(node: Node, graph: Graph, modules: dict[str, torch.nn.Module], custom_module_class_mapping: dict[QuantType, dict[type, type]], statically_quantized_custom_module_nodes: set[Node]) -> None:
    """Converts an observed custom module to a quantized custom module based on
    `custom_module_class_mapping`
    For static quantization, we'll also remove the previous `dequantize` node and
    attach the observer node for output to the module, the observer for the node
    will be converted to a dequantize node instead of quantize-dequantize pairs
    later in the graph. In the end we would have a quantized custom module that
    has the same interface as a default quantized module in nn.quantized namespace,
    i.e. quantized input and quantized output.

    Args:
      - node: The call_module node of the observed standalone module
      - graph: The graph containing the node
      - modules: named_module of original model
      - custom_module_class_mapping: mapping from observed custom module class to
        quantized custom module class, used to swap custom modules
      - statically_quantized_custom_module_nodes: we'll add the custom module node
        if we find it is statically quantized, this will be used later when converting
        observers to quant/dequant node pairs, if the observed node is a statically
        quantized custom module nodes, we'll convert the observer to a dequantize node,
        this is to keep the interface the same as the default quantized module.
        TODO: maybe we want to redesign this part to align with reference model design
        as well, but there has been some discussions around the interface, so we can do
        it later.
    """
def convert(model: GraphModule, is_reference: bool = False, convert_custom_config: ConvertCustomConfig | dict[str, Any] | None = None, is_standalone_module: bool = False, _remove_qconfig_flag: bool = True, qconfig_mapping: QConfigMapping | dict[str, Any] | None = None, backend_config: BackendConfig | dict[str, Any] | None = None, is_decomposed: bool = False, keep_original_weights: bool = False) -> GraphModule:
    """
    We will convert an observed model (a module with observer calls) to a reference
    quantized model, the rule is simple:
    1. for each observer module call in the graph, we'll convert it to calls to
       quantize and dequantize functions based on the observer instance
    2. for weighted operations like linear/conv, we need to convert them to reference
       quantized module, this requires us to know whether the dtype configured for the
       weight is supported in the backend, this is done in prepare step and the result
       is stored in observed_node_names, we can decide whether we need to swap the
       module based on this set

    Args:
       * `is_standalone_module`: when this flag is True, it means we are quantizing
       a submodule that is not inlined in parent module, and will be quantized
       separately as one unit.

       * `is_decomposed`: a boolean flag to indicate whether we want to use the
        quantize operator for decomposed quantized tensor
        (torch.ops.quantized_decomposed.quantize_per_tensor) or default/standalone
        quantized tensor (torch.quantize_per_tensor)

    Returns:
         a quantized standalone module, whether input/output is quantized is
         specified by prepare_custom_config, with
         input_quantized_idxs, output_quantized_idxs, please
         see docs for :func:`~torch.ao.quantization.prepare_fx` for details
    """
