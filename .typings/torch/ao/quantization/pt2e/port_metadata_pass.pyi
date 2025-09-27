import torch
from torch.fx.passes.infra.pass_base import PassBase, PassResult

__all__ = ['PortNodeMetaForQDQ']

class PortNodeMetaForQDQ(PassBase):
    '''
    Port metadata for nodes added by quantization flow.
    For static quant these are:
    - quantizer_per_tensor.default, dequantize_per_tensor.default
    - quantizer_per_channel.default, dequantize_per_channel.default
    For dynamic quant these are:
    - choose_qparams.tensor
    - quantizer_per_tensor.tensor, dequantize_per_tensor.tensor
    - quantizer_per_channel.default, dequantize_per_channel.default

    Rules of porting metadata:
    - Metadata to be ported:
      - nn_module_stack
      - stack_trace
      - quantization_tag
    - Metadata to NOT be ported:
      - Everything else
    - Rules:
      - Statically quantized patterns:
        - Dequantize nodes on the inputs to be quantized inherit metadata of the consumer node.
        - Quantize nodes on the outputs inherit metadata of the producer node.
        - Example 1:
          - Original: [Conv -> AvgPool -> Linear]
          - Quantized [Q-> DQ -> Conv -> Q -> DQ -> AvgPool -> Q -> DQ -> Linear -> Q -> DQ]
          - Inner brackets specify which nodes Q/DQ inherit metdata from
          - [Q-> [DQ -> Conv -> Q] -> [DQ -> AvgPool -> Q] -> [DQ -> Linear -> Q] -> DQ]
          - Note first Q and last DQ do not inherit metadata from any nodes
        - Example 2:
          - Original: [Conv -> AvgPool -> Linear]
          - AvgPool is not quantized
          - Quantized [Q-> DQ -> Conv -> Q -> DQ -> AvgPool -> Q -> DQ -> Linear -> Q -> DQ]
          - Inner brackets specify which nodes Q/DQ inherit metdata from
          - [Q-> [DQ -> Conv -> Q] -> DQ -> [AvgPool] -> Q -> [DQ -> Linear -> Q] -> DQ]
          - Note DQ and Q nodes around AvgPool do not inherit metadata from AvgPool because
            AvgPool was not supposed to be quantized. Metadata porting relies on quantization_annotation
            on the nodes (in this case AvgPool node) to conclude if the node or patter was
            supposed to be quantized. And subsequntly decide if the preceding Q, if any, should
            inherit metadata from AvgPool.
      - Dynamically quantized patterns:
        - Input that are dynamically quantized have choose_qparams, quantize and dequantize nodes
        - For example, below linear is dynamically quantized while rest statically:
          - Original: [Conv -> AvgPool -> Linear]
          - Quantized [Q-> DQ -> Conv -> Q -> DQ -> AvgPool -> Q -> DQ -> choose_params -> Q -> DQ -> Linear]
          - Quantized [Q-> [DQ -> Conv -> Q] -> [DQ -> AvgPool -> Q] -> DQ -> [choose_params -> Q -> DQ -> Linear]]
          - Note first Q does not inherit metadata from any nodes
    NB:
    - The best place for porting metadata is during observer conversion to q/dq. This is because it precisely
      knows which quantization spec is converted to q/dq and thus from where the metadata should be ported.
      However, since FX and PT2E quant workflow are on a common code-base, this hurts readability quite a bit.
      Doing it via a separate pass, helps readability of the code. Once we are able to refactor PT2E quant
      code, this pass should like to be integrated in the refactored variant of "convert" step.
    '''
    def call(self, graph_module: torch.fx.GraphModule) -> PassResult: ...
