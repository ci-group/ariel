from torch.distributed.tensor._api import DTensor as DTensor, distribute_module as distribute_module, distribute_tensor as distribute_tensor, empty as empty, full as full, ones as ones, rand as rand, randn as randn, zeros as zeros
from torch.distributed.tensor.placement_types import Partial as Partial, Placement as Placement, Replicate as Replicate, Shard as Shard

__all__ = ['DTensor', 'distribute_tensor', 'distribute_module', 'Shard', 'Replicate', 'Partial', 'Placement', 'ones', 'empty', 'full', 'rand', 'randn', 'zeros']
