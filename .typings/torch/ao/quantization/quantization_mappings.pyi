from typing import Any, Callable

__all__ = ['DEFAULT_REFERENCE_STATIC_QUANT_MODULE_MAPPINGS', 'DEFAULT_STATIC_QUANT_MODULE_MAPPINGS', 'DEFAULT_QAT_MODULE_MAPPINGS', 'DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS', 'DEFAULT_FLOAT_TO_QUANTIZED_OPERATOR_MAPPINGS', 'DEFAULT_MODULE_TO_ACT_POST_PROCESS', 'DEFAULT_STATIC_SPARSE_QUANT_MODULE_MAPPINGS', 'DEFAULT_DYNAMIC_SPARSE_QUANT_MODULE_MAPPINGS', 'no_observer_set', 'get_default_static_quant_module_mappings', 'get_default_static_quant_reference_module_mappings', 'get_embedding_static_quant_module_mappings', 'get_default_static_sparse_quant_module_mappings', 'get_static_quant_module_class', 'get_dynamic_quant_module_class', 'get_default_qat_module_mappings', 'get_embedding_qat_module_mappings', 'get_default_dynamic_quant_module_mappings', 'get_default_dynamic_sparse_quant_module_mappings', 'get_default_qconfig_propagation_list', 'get_default_compare_output_module_list', 'get_default_float_to_quantized_operator_mappings', 'get_quantized_operator']

DEFAULT_REFERENCE_STATIC_QUANT_MODULE_MAPPINGS: dict[Callable, Any]
DEFAULT_STATIC_QUANT_MODULE_MAPPINGS: dict[Callable, Any]
DEFAULT_QAT_MODULE_MAPPINGS: dict[Callable, Any]
DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS: dict[Callable, Any]
DEFAULT_FLOAT_TO_QUANTIZED_OPERATOR_MAPPINGS: dict[Callable | str, Callable]
DEFAULT_MODULE_TO_ACT_POST_PROCESS: dict[Callable, Callable]
DEFAULT_STATIC_SPARSE_QUANT_MODULE_MAPPINGS: dict[Callable, Any]
DEFAULT_DYNAMIC_SPARSE_QUANT_MODULE_MAPPINGS: dict[Callable, Any]

def no_observer_set() -> set[Any]:
    """These modules cannot have observers inserted by default."""
def get_default_static_quant_module_mappings() -> dict[Callable, Any]:
    """Get module mapping for post training static quantization"""
def get_default_static_quant_reference_module_mappings() -> dict[Callable, Any]:
    """Get reference module mapping for post training static quantization"""
def get_embedding_static_quant_module_mappings() -> dict[Callable, Any]:
    """Get module mapping, including mapping for embedding QAT"""
def get_default_static_sparse_quant_module_mappings() -> dict[Callable, Any]:
    """Get module mapping for post training static sparse quantization"""
def get_static_quant_module_class(float_module_class: Callable, additional_static_quant_mapping: dict[Callable, Any] | None = None, is_reference: bool = False) -> Any:
    """n Get the statically quantized module class corresponding to
    the floating point module class
    """
def get_dynamic_quant_module_class(float_module_class: Callable, additional_dynamic_quant_mapping: dict[Callable, Any] | None = None) -> Any:
    """n Get the dynamically quantized module class corresponding to
    the floating point module class
    """
def get_default_qat_module_mappings() -> dict[Callable, Any]:
    """Get default module mapping for quantization aware training"""
def get_embedding_qat_module_mappings() -> dict[Callable, Any]:
    """Get module mapping for quantization aware training
    This is includes default values in addition to
    enabling qat for embeddings.
    """
def get_default_dynamic_quant_module_mappings() -> dict[Callable, Any]:
    """Get module mapping for post training dynamic quantization"""
def get_default_dynamic_sparse_quant_module_mappings() -> dict[Callable, Any]:
    """Get module mapping for post training dynamic sparse quantization"""
def get_default_qconfig_propagation_list() -> set[Callable]:
    """Get the default list of module types that we'll attach qconfig
    attribute to in prepare
    """
def get_default_compare_output_module_list() -> set[Callable]:
    """Get list of module class types that we will record output
    in numeric suite
    """
def get_default_float_to_quantized_operator_mappings() -> dict[Callable | str, Callable]: ...
def get_quantized_operator(float_op: Callable | str) -> Callable:
    """Get the quantized operator corresponding to the float operator"""
