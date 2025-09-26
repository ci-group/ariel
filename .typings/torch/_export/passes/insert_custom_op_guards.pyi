import torch
from torch._export.passes._node_metadata_hook import _node_metadata_hook as _node_metadata_hook, _set_node_metadata_hook as _set_node_metadata_hook
from torch._library.fake_profile import OpProfile as OpProfile, TensorMetadata as TensorMetadata

def insert_custom_op_guards(gm: torch.fx.GraphModule, ops_to_guard: set[str]) -> None:
    """
    This is used by draft_export to insert guards in front of calls to custom
    operators which have a generated fake kernel.
    """
def get_op_profiles(gm: torch.fx.GraphModule, ops_to_guard: set[str]) -> dict[str, set[OpProfile]]:
    """
    This is used by draft_export to get a list of custom operator profiles so
    that we can generate fake kernels.
    """
