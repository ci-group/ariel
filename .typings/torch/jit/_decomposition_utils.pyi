import torch
from torch._ops import OpOverload as OpOverload, OpOverloadPacket as OpOverloadPacket

def _register_decomposition(op: OpOverload, graph: torch._C.Graph): ...
