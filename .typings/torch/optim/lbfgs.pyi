from .optimizer import Optimizer, ParamsT
from _typeshed import Incomplete
from torch import Tensor

__all__ = ['LBFGS']

class LBFGS(Optimizer):
    """Implements L-BFGS algorithm.

    Heavily inspired by `minFunc
    <https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html>`_.

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.

    .. note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.

    Args:
        params (iterable): iterable of parameters to optimize. Parameters must be real.
        lr (float, optional): learning rate (default: 1)
        max_iter (int, optional): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int, optional): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float, optional): termination tolerance on first order optimality
            (default: 1e-7).
        tolerance_change (float, optional): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int, optional): update history size (default: 100).
        line_search_fn (str, optional): either 'strong_wolfe' or None (default: None).
    """
    _params: Incomplete
    _numel_cache: Incomplete
    def __init__(self, params: ParamsT, lr: float | Tensor = 1, max_iter: int = 20, max_eval: int | None = None, tolerance_grad: float = 1e-07, tolerance_change: float = 1e-09, history_size: int = 100, line_search_fn: str | None = None) -> None: ...
    def _numel(self): ...
    def _gather_flat_grad(self): ...
    def _add_grad(self, step_size, update) -> None: ...
    def _clone_param(self): ...
    def _set_param(self, params_data) -> None: ...
    def _directional_evaluate(self, closure, x, t, d): ...
    def step(self, closure):
        """Perform a single optimization step.

        Args:
            closure (Callable): A closure that reevaluates the model
                and returns the loss.
        """
