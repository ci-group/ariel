from .linear import Linear as Linear, LinearPackedParams as LinearPackedParams
from torch.ao.nn.sparse.quantized import dynamic as dynamic

__all__ = ['dynamic', 'Linear', 'LinearPackedParams']
