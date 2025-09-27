from torch.ao.nn.quantized.modules.normalization import GroupNorm as GroupNorm, InstanceNorm1d as InstanceNorm1d, InstanceNorm2d as InstanceNorm2d, InstanceNorm3d as InstanceNorm3d, LayerNorm as LayerNorm

__all__ = ['LayerNorm', 'GroupNorm', 'InstanceNorm1d', 'InstanceNorm2d', 'InstanceNorm3d']
