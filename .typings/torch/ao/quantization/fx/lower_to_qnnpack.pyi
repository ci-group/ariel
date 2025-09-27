from torch.ao.quantization.qconfig import QConfigAny
from torch.fx import GraphModule

__all__ = ['lower_to_qnnpack']

def lower_to_qnnpack(model: GraphModule, qconfig_map: dict[str, QConfigAny], node_name_to_scope: dict[str, tuple[str, type]]) -> GraphModule:
    """Lower a quantized reference model (with reference quantized operator patterns)
    to qnnpack
    """
