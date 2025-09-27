from .binary_cmp import allclose as allclose, equal as equal
from .init import constant_ as constant_, kaiming_uniform_ as kaiming_uniform_, normal_ as normal_, uniform_ as uniform_
from torch.distributed._shard.sharding_spec.chunk_sharding_spec_ops.embedding import sharded_embedding as sharded_embedding
from torch.distributed._shard.sharding_spec.chunk_sharding_spec_ops.embedding_bag import sharded_embedding_bag as sharded_embedding_bag
