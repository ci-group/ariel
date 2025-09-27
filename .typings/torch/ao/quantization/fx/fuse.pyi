from .custom_config import FuseCustomConfig
from .fuse_handler import FuseHandler as FuseHandler
from torch.ao.quantization.backend_config import BackendConfig
from torch.fx import GraphModule
from typing import Any

__all__ = ['fuse', 'FuseHandler']

def fuse(model: GraphModule, is_qat: bool, fuse_custom_config: FuseCustomConfig | dict[str, Any] | None = None, backend_config: BackendConfig | dict[str, Any] | None = None) -> GraphModule: ...
