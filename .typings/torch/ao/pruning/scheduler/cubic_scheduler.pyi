from .base_scheduler import BaseScheduler
from _typeshed import Incomplete

__all__ = ['CubicSL']

class CubicSL(BaseScheduler):
    """Sets the sparsity level of each parameter group to the final sl
    plus a given exponential function.

    .. math::

        s_i = s_f + (s_0 - s_f) \\cdot \\left( 1 - \\frac{t - t_0}{n\\Delta t} \\right)^3

    where :math:`s_i` is the sparsity at epoch :math:`t`, :math;`s_f` is the final
    sparsity level, :math:`f(i)` is the function to be applied to the current epoch
    :math:`t`, initial epoch :math:`t_0`, and final epoch :math:`t_f`.
    :math:`\\Delta t` is used to control how often the update of the sparsity level
    happens. By default,

    Args:
        sparsifier (BaseSparsifier): Wrapped sparsifier.
        init_sl (int, list): Initial level of sparsity
        init_t (int, list): Initial step, when pruning starts
        delta_t (int, list): Pruning frequency
        total_t (int, list): Total number of pruning steps
        initially_zero (bool, list): If True, sets the level of sparsity to 0
            before init_t (:math:`t_0`). Otherwise, the sparsity level before
            init_t (:math:`t_0`) is set to init_sl(:math:`s_0`)
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """
    sparsifier: Incomplete
    init_sl: Incomplete
    init_t: Incomplete
    delta_t: Incomplete
    total_t: Incomplete
    initially_zero: Incomplete
    def __init__(self, sparsifier, init_sl: float = 0.0, init_t: int = 0, delta_t: int = 10, total_t: int = 100, initially_zero: bool = False, last_epoch: int = -1, verbose: bool = False) -> None: ...
    @staticmethod
    def sparsity_compute_fn(s_0, s_f, t, t_0, dt, n, initially_zero: bool = False):
        ''' "Computes the current level of sparsity.

        Based on https://arxiv.org/pdf/1710.01878.pdf

        Args:
            s_0: Initial level of sparsity, :math:`s_i`
            s_f: Target level of sparsity, :math:`s_f`
            t: Current step, :math:`t`
            t_0: Initial step, :math:`t_0`
            dt: Pruning frequency, :math:`\\Delta T`
            n: Pruning steps, :math:`n`
            initially_zero: Sets the level of sparsity to 0 before t_0.
                If False, sets to s_0

        Returns:
            The sparsity level :math:`s_t` at the current step :math:`t`
        '''
    def get_sl(self): ...
