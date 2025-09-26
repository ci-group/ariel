from torch.fx import GraphModule

__all__ = ['prepare']

def prepare(model: GraphModule, node_name_to_scope: dict[str, tuple[str, type]], is_qat: bool, obs_or_fq_callback=None) -> GraphModule: ...
