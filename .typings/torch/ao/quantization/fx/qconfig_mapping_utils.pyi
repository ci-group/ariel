import torch
from torch.ao.nn.intrinsic import _FusedModule as _FusedModule
from torch.ao.quantization import QConfig as QConfig
from torch.ao.quantization.backend_config import BackendConfig as BackendConfig, DTypeConfig as DTypeConfig
from torch.ao.quantization.backend_config.utils import get_module_to_qat_module as get_module_to_qat_module
from torch.ao.quantization.observer import _is_activation_post_process as _is_activation_post_process
from torch.ao.quantization.qconfig import QConfigAny as QConfigAny, _add_module_to_qconfig_obs_ctr as _add_module_to_qconfig_obs_ctr, qconfig_equals as qconfig_equals
from torch.ao.quantization.qconfig_mapping import QConfigMapping as QConfigMapping, _MODULE_NAME_DICT_KEY as _MODULE_NAME_DICT_KEY, _MODULE_NAME_REGEX_DICT_KEY as _MODULE_NAME_REGEX_DICT_KEY, _OBJECT_TYPE_DICT_KEY as _OBJECT_TYPE_DICT_KEY
from torch.ao.quantization.utils import _parent_name as _parent_name, get_qconfig_dtypes as get_qconfig_dtypes
from torch.fx import GraphModule as GraphModule
from torch.fx.graph import Graph as Graph
from typing import Any, Callable

__all__: list[str]

def _maybe_adjust_qconfig_for_module_name_object_type_order(qconfig_mapping: QConfigMapping, cur_module_path: str, cur_object_type: Callable, cur_object_type_idx: int, fallback_qconfig: QConfigAny) -> QConfigAny: ...
def _update_qconfig_for_fusion(model: GraphModule, qconfig_mapping: QConfigMapping):
    """
    Update the QConfigMapping to account for fused modules such as LinearReLU.
    This assumes the QConfigMapping's attributes have already been converted to OrderedDicts.
    """
def _generate_node_name_to_qconfig(root: torch.nn.Module, modules: dict[str, torch.nn.Module], input_graph: Graph, qconfig_mapping: QConfigMapping, node_name_to_scope: dict[str, tuple[str, type]]) -> dict[str, QConfigAny]: ...
def _check_is_valid_config_dict(config_dict: Any, allowed_keys: set[str], dict_name: str) -> None:
    """Checks if the given config_dict has the correct keys

    Args:
      `config_dict`: dictionary whose keys we want to check
    """
def _compare_prepare_convert_qconfig_mappings(prepare_qconfig_mapping: QConfigMapping, convert_qconfig_mapping: QConfigMapping):
    """Compare the qconfig_mapping passed in convert to the one from prepare and check the values

    Args:
      `prepare_qconfig_mapping`: configuration for prepare quantization step
      `convert_qconfig_mapping`: configuration for convert quantization step
    """
def _is_qconfig_supported_by_dtype_configs(qconfig: QConfig, dtype_configs: list[DTypeConfig]): ...
def _get_object_type_qconfig(qconfig_mapping: QConfigMapping, object_type: Callable | str, fallback_qconfig: QConfigAny) -> QConfigAny: ...
def _get_module_name_regex_qconfig(qconfig_mapping, module_name, fallback_qconfig): ...
def _get_module_name_qconfig(qconfig_mapping, module_name, fallback_qconfig): ...
def _maybe_adjust_qconfig_for_module_type_or_name(qconfig_mapping, module_type, module_name, global_qconfig): ...
def _get_flattened_qconfig_dict(qconfig_mapping: QConfigMapping) -> dict[Callable | str, QConfigAny]:
    '''flatten the global, object_type and module_name qconfig
    to the same qconfig_dict so that it can be used by
    propagate_qconfig_ function.
    "module_name_regex" is ignored for now since it\'s not supported
    in propagate_qconfig_, but it can be fixed later.

    For example:
    Input: {
      "": qconfig,
      "object_type": [
        (torch.add, qconfig)
      ],
      "module_name": [
        ("conv", qconfig)
      ]
    }

    Output: {
      "": qconfig,
      torch.add: qconfig,
      "conv": qconfig
    }
    '''
def _update_qconfig_for_qat(qconfig_mapping: QConfigMapping, backend_config: BackendConfig):
    """
    Update the qconfig_mapping to account for module swaps during QAT.
    During QAT we perform a module swap on the nn.Module types to the corresponding nn.qat.modules types.
    """
