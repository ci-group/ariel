import torch.distributed as dist
from . import config as config

_COMPILE_PG: dist.ProcessGroup | None
_GUARD_PG: dist.ProcessGroup | None

def get_compile_pg() -> dist.ProcessGroup | None: ...
def get_guard_pg() -> dist.ProcessGroup | None: ...
