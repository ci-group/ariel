import torch
import torch.nn as nn
from .backend_config import BackendConfig, DTypeConfig
from torch.ao.quantization.utils import Pattern
from typing import Any, Callable

__all__ = ['get_pattern_to_dtype_configs', 'get_qat_module_classes', 'get_fused_module_classes', 'get_pattern_to_input_type_to_index', 'get_root_module_to_quantized_reference_module', 'get_fuser_method_mapping', 'get_module_to_qat_module', 'get_fusion_pattern_to_root_node_getter', 'get_fusion_pattern_to_extra_inputs_getter', 'remove_boolean_dispatch_from_name', 'pattern_to_human_readable', 'entry_to_pretty_str']

def get_pattern_to_dtype_configs(backend_config: BackendConfig) -> dict[Pattern, list[DTypeConfig]]: ...
def get_qat_module_classes(backend_config: BackendConfig) -> tuple[type, ...]: ...
def get_fused_module_classes(backend_config: BackendConfig) -> tuple[type, ...]: ...
def get_pattern_to_input_type_to_index(backend_config: BackendConfig) -> dict[Pattern, dict[str, int]]: ...
def get_root_module_to_quantized_reference_module(backend_config: BackendConfig) -> dict[type[torch.nn.Module], type[torch.nn.Module]]: ...
def get_fuser_method_mapping(backend_config: BackendConfig) -> dict[Pattern, nn.Sequential | Callable]: ...
def get_module_to_qat_module(backend_config: BackendConfig) -> dict[Pattern, type[torch.nn.Module]]: ...
def get_fusion_pattern_to_root_node_getter(backend_config: BackendConfig) -> dict[Pattern, Callable]:
    '''Get a map from fusion pattern to a function that returns the root node
    from the fusion pattern, e.g. the most common one is:
    def get_root_node(node_pattern):
        while not isinstance(node_pattern[-1], Node):
            node_pattern = node_pattern[-1]
        return node_pattern[-1]
    This can work for all patterns whose root node is the "last node" in the pattern,
    e.g. (torch.add, MatchAllNode, (torch.ReLU, torch.Conv2d))
    '''
def get_fusion_pattern_to_extra_inputs_getter(backend_config: BackendConfig) -> dict[Pattern, Callable]:
    """Get a map from fusion pattern to a function that returns extra input nodes
    from the fusion pattern, in the order required by the root node. This is optional,
    if not specified, we will not copy over any extra inputs for the root node.
    Example:
    # Let's say we have the pattern (torch.add, MatchAllNode, (torch.nn.BatchNorm2d, torch.nn.Conv2d))
    # and root node is torch.nn.Conv2d, and the node in MatchAllNode would be an extra
    # argument to the fused module, we can unpack the pattern and return the node at
    # MatchAllNode here
    # we can implement extra_inputs_getter as follows:
    def extra_inputs_getter(pattern) -> List[Any]:
        add, extra_input, conv_pattern = pattern
        return [extra_input]
    """
def remove_boolean_dispatch_from_name(p) -> Any:
    """
    Some ops have a default string representation such as
    '<function boolean_dispatch.<locals>.fn at 0x7ff1106bf280>',
    this function replaces them with the hardcoded function names.
    """
def pattern_to_human_readable(p) -> Any: ...
def entry_to_pretty_str(entry) -> str:
    """
    Given a backend_config_dict entry, returns a string with the human readable
    representation of it.
    """
