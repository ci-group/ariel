import torch.nn as nn
from torch._functorch.utils import exposed_in as exposed_in

def batch_norm_without_running_stats(module: nn.Module) -> None: ...
def replace_all_batch_norm_modules_(root: nn.Module) -> nn.Module:
    """
    In place updates :attr:`root` by setting the ``running_mean`` and ``running_var`` to be None and
    setting track_running_stats to be False for any nn.BatchNorm module in :attr:`root`
    """
