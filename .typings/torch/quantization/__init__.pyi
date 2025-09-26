from .fake_quantize import *
from .fuser_method_mappings import *
from .observer import *
from .qconfig import *
from .quant_type import *
from .quantization_mappings import *
from .quantize import *
from .quantize_jit import *
from .stubs import *
from .fuse_modules import fuse_modules as fuse_modules

__all__ = ['QuantWrapper', 'QuantStub', 'DeQuantStub', 'quantize', 'quantize_dynamic', 'quantize_qat', 'prepare', 'convert', 'prepare_qat', 'quantize_jit', 'quantize_dynamic_jit', '_prepare_ondevice_dynamic_jit', '_convert_ondevice_dynamic_jit', '_quantize_ondevice_dynamic_jit', 'QuantType', 'get_default_static_quant_module_mappings', 'get_static_quant_module_class', 'get_default_dynamic_quant_module_mappings', 'get_default_qat_module_mappings', 'get_default_qconfig_propagation_list', 'get_default_compare_output_module_list', 'get_quantized_operator', 'get_fuser_method', 'propagate_qconfig_', 'add_quant_dequant', 'swap_module', 'default_eval_fn', 'ObserverBase', 'WeightObserver', 'HistogramObserver', 'observer', 'default_observer', 'default_weight_observer', 'default_placeholder_observer', 'default_per_channel_weight_observer', 'default_fake_quant', 'default_weight_fake_quant', 'default_fixed_qparams_range_neg1to1_fake_quant', 'default_fixed_qparams_range_0to1_fake_quant', 'default_per_channel_weight_fake_quant', 'default_histogram_fake_quant', 'QConfig', 'default_qconfig', 'default_dynamic_qconfig', 'float16_dynamic_qconfig', 'float_qparams_weight_only_qconfig', 'default_qat_qconfig', 'prepare_qat', 'quantize_qat', 'fuse_modules']

def default_eval_fn(model, calib_data) -> None:
    """
    Default evaluation function takes a torch.utils.data.Dataset or a list of
    input Tensors and run the model on the dataset
    """

# Names in __all__ with no definition:
#   DeQuantStub
#   HistogramObserver
#   ObserverBase
#   QConfig
#   QuantStub
#   QuantType
#   QuantWrapper
#   WeightObserver
#   _convert_ondevice_dynamic_jit
#   _prepare_ondevice_dynamic_jit
#   _quantize_ondevice_dynamic_jit
#   add_quant_dequant
#   convert
#   default_dynamic_qconfig
#   default_fake_quant
#   default_fixed_qparams_range_0to1_fake_quant
#   default_fixed_qparams_range_neg1to1_fake_quant
#   default_histogram_fake_quant
#   default_observer
#   default_per_channel_weight_fake_quant
#   default_per_channel_weight_observer
#   default_placeholder_observer
#   default_qat_qconfig
#   default_qconfig
#   default_weight_fake_quant
#   default_weight_observer
#   float16_dynamic_qconfig
#   float_qparams_weight_only_qconfig
#   get_default_compare_output_module_list
#   get_default_dynamic_quant_module_mappings
#   get_default_qat_module_mappings
#   get_default_qconfig_propagation_list
#   get_default_static_quant_module_mappings
#   get_fuser_method
#   get_quantized_operator
#   get_static_quant_module_class
#   observer
#   prepare
#   prepare_qat
#   prepare_qat
#   propagate_qconfig_
#   quantize
#   quantize_dynamic
#   quantize_dynamic_jit
#   quantize_jit
#   quantize_qat
#   quantize_qat
#   swap_module
