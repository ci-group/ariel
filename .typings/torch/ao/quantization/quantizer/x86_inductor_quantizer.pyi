import functools
import torch
from _typeshed import Incomplete
from dataclasses import dataclass
from torch.ao.quantization.quantizer.quantizer import QuantizationAnnotation, Quantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import QuantizationConfig
from torch.fx import Node
from torch.fx.passes.utils.source_matcher_utils import SourcePartition
from typing import Callable
from typing_extensions import TypeAlias

__all__ = ['X86InductorQuantizer', 'get_default_x86_inductor_quantization_config', 'get_x86_inductor_linear_dynamic_fp16_config']

FilterFn: TypeAlias = Callable[[list[Node]], bool]

@dataclass
class _X86InductorQuantizationAnnotation(QuantizationAnnotation):
    _is_output_of_quantized_pattern: bool = ...
propagation_quantizable_ops = int8_in_int8_out_ops

@functools.lru_cache
def get_default_x86_inductor_quantization_config(is_qat: bool = False, is_dynamic: bool = False, reduce_range: bool = False):
    """
    reduce_range is False by default. Set it to True on earlier CPUs without VNNI to avoid accuracy issue.
    """
@functools.lru_cache
def get_x86_inductor_linear_dynamic_fp16_config():
    """
    For linear_dynamic_fp16. The name may be confusing.
    The op's behavior is fp32_input * (fp16_weight -> to_fp32) -> fp32_output.
    """

@dataclass
class _CurrentQuantizationMode:
    """Configuration defining the current quantization mode for the quantizer.

    All possible current quantization modes are listed below:
    ----------------------------------------------------------------------------------------------------------
                |                                       dynamic_state
     qat_state  |---------------------------------------------------------------------------------------------
                |                           None                              |    True       |  False
    ----------------------------------------------------------------------------------------------------------
        None    | quantizer does not receive a non-None `quantization_config` | \\             | \\\n        False   | quantizer will not do QAT                                   | dynamic       | static
        True    | quantizer will do QAT                                       | QAT + dynamic | QAT + static
    """
    qat_state: bool | None
    dynamic_state: bool | None

class X86InductorQuantizer(Quantizer):
    module_function_to_aten_operator_type: Incomplete
    global_config: QuantizationConfig | None
    operator_type_qconfig: dict[torch._ops.OpOverloadPacket, QuantizationConfig | None]
    module_name_qconfig: dict[str, QuantizationConfig | None]
    def __init__(self) -> None: ...
    def _get_current_quantization_mode(self) -> _CurrentQuantizationMode:
        """Retrieves the current quantization mode based on all configurations."""
    def _need_skip_config(self, quantization_config: QuantizationConfig | None) -> bool:
        """Check if the provided quantization config is valid for X86InductorQuantizer.

        Mixed static/dynamic configurations or mixed QAT/non-QAT configurations are not supported.
        To avoid such a mix, we compare the incoming configuration with current configuration status.
        Refer the `_CurrentQuantizationMode` definition for all possible modes.
        """
    def set_global(self, quantization_config: QuantizationConfig): ...
    def get_global_quantization_config(self): ...
    @_config_checker
    def set_function_type_qconfig(self, function_type: Callable, quantization_config: QuantizationConfig | None) -> X86InductorQuantizer: ...
    @_config_checker
    def set_module_type_qconfig(self, module_type: torch.nn.Module, quantization_config: QuantizationConfig | None) -> X86InductorQuantizer: ...
    @_config_checker
    def set_module_name_qconfig(self, module_name: str, quantization_config: QuantizationConfig | None):
        '''Set quantization_config for a submodule with name: `module_name`, for example:
        quantizer.set_module_name_qconfig("blocks.sub"), it will quantize all supported operator/operator
        patterns in the submodule with this module name with the given `quantization_config`

        The supported operators include `quantizable_ops` and `propagation_quantizable_ops`.
        '''
    def _set_aten_operator_qconfig(self, operator_type: torch._ops.OpOverloadPacket, quantization_config: QuantizationConfig | None) -> X86InductorQuantizer: ...
    def _annotate_conv_node_helper(self, conv_node: torch.fx.Node, annotate_output: bool, quantization_config: QuantizationConfig | None) -> None:
        """Helper function to annotate the conv node"""
    def _annotate_linear_node_helper(self, linear_node: torch.fx.Node, annotate_output: bool, quantization_config: QuantizationConfig | None) -> None:
        """Helper function to annotate the linear node"""
    def _get_output_nodes_of_partitions(self, partition_list: list[SourcePartition]) -> list[torch.fx.Node]:
        """Helper function to get the output node list from partition list"""
    def _get_input_idx_for_binary_node(self, conv_gemm_node: torch.fx.Node, binary_node: torch.fx.Node):
        """Helper function to check conv_gemm and extra input node index
        for binary node fused with conv_gemm.
        """
    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """Annotate the given model with quantization configurations.

        Annotation contracts:
        1. Annotate each node according to the user's qconfig in the following order:
        `module_name_qconfig`, `operator_type_qconfig`, and `global_config`.
        2. Avoid re-annotating nodes already annotated in prior stages. For example,
        if `linear1` has been annotated by `module_name_qconfig`, it won't be annotated again
        during the processing of the 'operator_type_qconfig' or 'global_config'.
        3. For config is `None`, the node will be annotated with `_X86InductorQuantizationAnnotation(_annotated=True)`.

        For each pair of (module_name_or_operator_type_or_global, qconfig), a filter function is created.
        This filter function checks if the node is marked by current stage and not annotated by the previous stage.
        """
    def _annotate_with_config(self, model: torch.fx.GraphModule, quantization_config: QuantizationConfig | None, filter_fn: FilterFn) -> None:
        """Annotate the model with the given quantization configuration.

        High-level description of quantization recipe for X86 Inductor Backend:
        Step 1: Apply quantization recipe for fusion patterns of conv/linear to enable int8 data type actively.
        Step 2: Propagate quantization annotation for patterns besides conv/linear. Go through the pattern in model
        from start to the end. If a pattern supports computation with int8 data type and inputs connected to
        quantized patterns, annotate its inputs as quantized pattern.
        """
    def _annotate_qat_conv2d_fusion_pattern(self, model: torch.fx.GraphModule, quantization_config: QuantizationConfig | None, filter_fn: FilterFn | None = None): ...
    def _annotate_qat_conv2d_bn_binary_unary(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig | None, filter_fn: FilterFn | None = None) -> None: ...
    def _annotate_qat_conv2d_bn_binary(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig | None, filter_fn: FilterFn | None = None) -> None: ...
    def _annotate_qat_conv2d_bn_unary(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig | None, filter_fn: FilterFn | None = None) -> None: ...
    def _annotate_qat_conv2d_bn(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig | None, filter_fn: FilterFn | None = None) -> None: ...
    def _annotate_conv2d_fusion_pattern(self, model: torch.fx.GraphModule, quantization_config: QuantizationConfig | None, filter_fn: FilterFn | None = None): ...
    def _annotate_linear_fusion_pattern(self, model: torch.fx.GraphModule, quantization_config: QuantizationConfig | None, filter_fn: FilterFn | None = None): ...
    def _annotate_matmul(self, model: torch.fx.GraphModule, quantization_config: QuantizationConfig | None, filter_fn: FilterFn | None = None): ...
    def _annotate_conv2d_binary_unary(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig | None, filter_fn: FilterFn | None = None) -> None: ...
    def _annotate_conv2d_binary(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig | None, filter_fn: FilterFn | None = None) -> None: ...
    def _annotate_conv2d_unary(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig | None, filter_fn: FilterFn | None = None) -> None: ...
    def _annotate_conv2d(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig | None, filter_fn: FilterFn | None = None) -> None: ...
    def _annotate_maxpool2d(self, node: Node, quantization_config: QuantizationConfig | None) -> None: ...
    def _annotate_cat(self, node: Node, quantization_config: QuantizationConfig) -> None: ...
    def _annotate_propagation_quantizable_pattern_entry(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig | None, filter_fn: FilterFn | None = None): ...
    def _annotate_propagation_quantizable_pattern(self, node: Node, quantization_config, filter_fn) -> None: ...
    def _annotate_output_share_observer_as_input(self, input_node: Node, source_node: Node): ...
    def _annotate_output_for_int8_in_int8_out_pattern_entry(self, model: torch.fx.GraphModule): ...
    def _annotate_output_for_int8_in_int8_out_pattern(self, node: Node) -> None:
        """
        Check and insert observer at output of node in int8_in_int8_out_ops if needed.
        Recipe refers to
        https://github.com/intel/intel-extension-for-pytorch/blob/90d19323d96afc53fcc22ba5a7bb3fb07fdd6c1c/intel_extension_for_pytorch/quantization/_utils.py#L495
        """
    def _annotate_linear(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig | None, filter_fn: FilterFn | None = None) -> None: ...
    def _annotate_linear_unary(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig | None, filter_fn: FilterFn | None = None) -> None: ...
    def _annotate_linear_binary_unary(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig | None, filter_fn: FilterFn | None = None) -> None: ...
    def validate(self, model: torch.fx.GraphModule) -> None: ...
