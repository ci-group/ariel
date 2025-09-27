from .base_scheduler import BaseScheduler
from _typeshed import Incomplete

__all__ = ['LambdaSL']

class LambdaSL(BaseScheduler):
    """Sets the sparsity level of each parameter group to the final sl
    times a given function. When last_epoch=-1, sets initial sl as zero.
    Args:
        sparsifier (BaseSparsifier): Wrapped sparsifier.
        sl_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in sparsifier.param_groups.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    Example:
        >>> # Assuming sparsifier has two groups.
        >>> lambda1 = lambda epoch: epoch // 30
        >>> lambda2 = lambda epoch: 0.95**epoch
        >>> # xdoctest: +SKIP
        >>> scheduler = LambdaSL(sparsifier, sl_lambda=[lambda1, lambda2])
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """
    sparsifier: Incomplete
    sl_lambdas: Incomplete
    def __init__(self, sparsifier, sl_lambda, last_epoch: int = -1, verbose: bool = False) -> None: ...
    def get_sl(self): ...
