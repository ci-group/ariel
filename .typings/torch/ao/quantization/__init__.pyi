from .fake_quantize import *
from .fuser_method_mappings import *
from .observer import *
from .qconfig import *
from .qconfig_mapping import *
from .quant_type import *
from .quantization_mappings import *
from .quantize import *
from .quantize_jit import *
from .stubs import *
import torch
from .fuse_modules import fuse_modules as fuse_modules, fuse_modules_qat as fuse_modules_qat
from .pt2e._numeric_debugger import CUSTOM_KEY as CUSTOM_KEY, NUMERIC_DEBUG_HANDLE_KEY as NUMERIC_DEBUG_HANDLE_KEY, compare_results as compare_results, extract_results_from_loggers as extract_results_from_loggers, generate_numeric_debug_handle as generate_numeric_debug_handle, prepare_for_propagation_comparison as prepare_for_propagation_comparison
from .pt2e.export_utils import _allow_exported_model_train_eval as allow_exported_model_train_eval, _move_exported_model_to_eval as move_exported_model_to_eval, _move_exported_model_to_train as move_exported_model_to_train
from _typeshed import Incomplete
from torch import Tensor
from typing import Callable

__all__ = ['DeQuantStub', 'FakeQuantize', 'FakeQuantizeBase', 'FixedQParamsFakeQuantize', 'FixedQParamsObserver', 'FusedMovingAvgObsFakeQuantize', 'HistogramObserver', 'MatchAllNode', 'MinMaxObserver', 'MovingAverageMinMaxObserver', 'MovingAveragePerChannelMinMaxObserver', 'NoopObserver', 'ObserverBase', 'ObserverOrFakeQuantize', 'Pattern', 'PerChannelMinMaxObserver', 'PlaceholderObserver', 'QConfig', 'QConfigAny', 'QConfigDynamic', 'QConfigMapping', 'QuantStub', 'QuantType', 'QuantWrapper', 'RecordingObserver', 'ReuseInputObserver', 'UniformQuantizationObserverBase', 'add_quant_dequant', 'convert', 'convert_dynamic_jit', 'convert_jit', 'default_affine_fixed_qparams_fake_quant', 'default_affine_fixed_qparams_observer', 'default_debug_observer', 'default_dynamic_fake_quant', 'default_dynamic_quant_observer', 'default_embedding_fake_quant', 'default_embedding_fake_quant_4bit', 'default_eval_fn', 'default_fake_quant', 'default_fixed_qparams_range_0to1_fake_quant', 'default_fixed_qparams_range_0to1_observer', 'default_fixed_qparams_range_neg1to1_fake_quant', 'default_fixed_qparams_range_neg1to1_observer', 'default_float_qparams_observer', 'default_float_qparams_observer_4bit', 'default_fused_act_fake_quant', 'default_fused_per_channel_wt_fake_quant', 'default_fused_wt_fake_quant', 'default_histogram_fake_quant', 'default_histogram_observer', 'default_observer', 'default_per_channel_weight_fake_quant', 'default_per_channel_weight_observer', 'default_placeholder_observer', 'default_reuse_input_observer', 'default_symmetric_fixed_qparams_fake_quant', 'default_symmetric_fixed_qparams_observer', 'default_weight_fake_quant', 'default_weight_observer', 'disable_fake_quant', 'disable_observer', 'enable_fake_quant', 'enable_observer', 'fuse_conv_bn', 'fuse_conv_bn_jit', 'fuse_conv_bn_relu', 'fuse_convtranspose_bn', 'fuse_linear_bn', 'fuse_modules', 'fuse_modules_qat', 'fused_per_channel_wt_fake_quant_range_neg_127_to_127', 'fused_wt_fake_quant_range_neg_127_to_127', 'get_combined_dict', 'get_default_compare_output_module_list', 'get_default_custom_config_dict', 'get_default_dynamic_quant_module_mappings', 'get_default_dynamic_sparse_quant_module_mappings', 'get_default_float_to_quantized_operator_mappings', 'get_default_qat_module_mappings', 'get_default_qat_qconfig', 'get_default_qat_qconfig_dict', 'get_default_qat_qconfig_mapping', 'get_default_qconfig', 'get_default_qconfig_dict', 'get_default_qconfig_mapping', 'get_default_qconfig_propagation_list', 'get_default_static_quant_module_mappings', 'get_default_static_quant_reference_module_mappings', 'get_default_static_sparse_quant_module_mappings', 'get_dynamic_quant_module_class', 'get_embedding_qat_module_mappings', 'get_embedding_static_quant_module_mappings', 'get_fuser_method', 'get_fuser_method_new', 'get_observer_state_dict', 'get_quantized_operator', 'get_static_quant_module_class', 'load_observer_state_dict', 'move_exported_model_to_eval', 'move_exported_model_to_train', 'allow_exported_model_train_eval', 'no_observer_set', 'per_channel_weight_observer_range_neg_127_to_127', 'prepare', 'prepare_dynamic_jit', 'prepare_jit', 'prepare_qat', 'propagate_qconfig_', 'qconfig_equals', 'quantize', 'quantize_dynamic', 'quantize_dynamic_jit', 'quantize_jit', 'quantize_qat', 'script_qconfig', 'script_qconfig_dict', 'swap_module', 'weight_observer_range_neg_127_to_127', 'generate_numeric_debug_handle', 'CUSTOM_KEY', 'NUMERIC_DEBUG_HANDLE_KEY', 'prepare_for_propagation_comparison', 'extract_results_from_loggers', 'compare_results', 'AffineQuantizedObserverBase', 'Granularity', 'MappingType', 'PerAxis', 'PerBlock', 'PerGroup', 'PerRow', 'PerTensor', 'PerToken', 'TorchAODType', 'ZeroPointDomain', 'get_block_size']

ObserverOrFakeQuantize = ObserverBase | FakeQuantizeBase

def default_eval_fn(model, calib_data) -> None:
    """Define the default evaluation function.

    Default evaluation function takes a torch.utils.data.Dataset or a list of
    input Tensors and run the model on the dataset
    """

class _DerivedObserverOrFakeQuantize(ObserverBase):
    """This observer is used to describe an observer whose quantization parameters
    are derived from other observers
    """
    obs_or_fqs: Incomplete
    derive_qparams_fn: Incomplete
    quant_min: Incomplete
    quant_max: Incomplete
    qscheme: Incomplete
    ch_axis: Incomplete
    def __init__(self, dtype: torch.dtype, obs_or_fqs: list[ObserverOrFakeQuantize], derive_qparams_fn: Callable[[list[ObserverOrFakeQuantize]], tuple[Tensor, Tensor]], quant_min: int | None = None, quant_max: int | None = None, qscheme: torch.qscheme | None = None, ch_axis: int | None = None) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    def calculate_qparams(self): ...

# Names in __all__ with no definition:
#   AffineQuantizedObserverBase
#   DeQuantStub
#   FakeQuantize
#   FakeQuantizeBase
#   FixedQParamsFakeQuantize
#   FixedQParamsObserver
#   FusedMovingAvgObsFakeQuantize
#   Granularity
#   HistogramObserver
#   MappingType
#   MatchAllNode
#   MinMaxObserver
#   MovingAverageMinMaxObserver
#   MovingAveragePerChannelMinMaxObserver
#   NoopObserver
#   ObserverBase
#   Pattern
#   PerAxis
#   PerBlock
#   PerChannelMinMaxObserver
#   PerGroup
#   PerRow
#   PerTensor
#   PerToken
#   PlaceholderObserver
#   QConfig
#   QConfigAny
#   QConfigDynamic
#   QConfigMapping
#   QuantStub
#   QuantType
#   QuantWrapper
#   RecordingObserver
#   ReuseInputObserver
#   TorchAODType
#   UniformQuantizationObserverBase
#   ZeroPointDomain
#   add_quant_dequant
#   convert
#   convert_dynamic_jit
#   convert_jit
#   default_affine_fixed_qparams_fake_quant
#   default_affine_fixed_qparams_observer
#   default_debug_observer
#   default_dynamic_fake_quant
#   default_dynamic_quant_observer
#   default_embedding_fake_quant
#   default_embedding_fake_quant_4bit
#   default_fake_quant
#   default_fixed_qparams_range_0to1_fake_quant
#   default_fixed_qparams_range_0to1_observer
#   default_fixed_qparams_range_neg1to1_fake_quant
#   default_fixed_qparams_range_neg1to1_observer
#   default_float_qparams_observer
#   default_float_qparams_observer_4bit
#   default_fused_act_fake_quant
#   default_fused_per_channel_wt_fake_quant
#   default_fused_wt_fake_quant
#   default_histogram_fake_quant
#   default_histogram_observer
#   default_observer
#   default_per_channel_weight_fake_quant
#   default_per_channel_weight_observer
#   default_placeholder_observer
#   default_reuse_input_observer
#   default_symmetric_fixed_qparams_fake_quant
#   default_symmetric_fixed_qparams_observer
#   default_weight_fake_quant
#   default_weight_observer
#   disable_fake_quant
#   disable_observer
#   enable_fake_quant
#   enable_observer
#   fuse_conv_bn
#   fuse_conv_bn_jit
#   fuse_conv_bn_relu
#   fuse_convtranspose_bn
#   fuse_linear_bn
#   fused_per_channel_wt_fake_quant_range_neg_127_to_127
#   fused_wt_fake_quant_range_neg_127_to_127
#   get_block_size
#   get_combined_dict
#   get_default_compare_output_module_list
#   get_default_custom_config_dict
#   get_default_dynamic_quant_module_mappings
#   get_default_dynamic_sparse_quant_module_mappings
#   get_default_float_to_quantized_operator_mappings
#   get_default_qat_module_mappings
#   get_default_qat_qconfig
#   get_default_qat_qconfig_dict
#   get_default_qat_qconfig_mapping
#   get_default_qconfig
#   get_default_qconfig_dict
#   get_default_qconfig_mapping
#   get_default_qconfig_propagation_list
#   get_default_static_quant_module_mappings
#   get_default_static_quant_reference_module_mappings
#   get_default_static_sparse_quant_module_mappings
#   get_dynamic_quant_module_class
#   get_embedding_qat_module_mappings
#   get_embedding_static_quant_module_mappings
#   get_fuser_method
#   get_fuser_method_new
#   get_observer_state_dict
#   get_quantized_operator
#   get_static_quant_module_class
#   load_observer_state_dict
#   no_observer_set
#   per_channel_weight_observer_range_neg_127_to_127
#   prepare
#   prepare_dynamic_jit
#   prepare_jit
#   prepare_qat
#   propagate_qconfig_
#   qconfig_equals
#   quantize
#   quantize_dynamic
#   quantize_dynamic_jit
#   quantize_jit
#   quantize_qat
#   script_qconfig
#   script_qconfig_dict
#   swap_module
#   weight_observer_range_neg_127_to_127
