from torch.ao.nn.qat.modules.conv import Conv1d as Conv1d, Conv2d as Conv2d, Conv3d as Conv3d
from torch.ao.nn.qat.modules.embedding_ops import Embedding as Embedding, EmbeddingBag as EmbeddingBag
from torch.ao.nn.qat.modules.linear import Linear as Linear

__all__ = ['Linear', 'Conv1d', 'Conv2d', 'Conv3d', 'Embedding', 'EmbeddingBag']
