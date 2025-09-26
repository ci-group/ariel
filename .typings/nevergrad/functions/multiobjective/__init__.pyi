import nevergrad.common.typing as tp
from nevergrad.common import errors as errors

class MultiobjectiveFunction:
    '''MultiobjectiveFunction is deprecated and is removed after v0.4.3 "
    because it is no more needed. You should just pass a multiobjective loss to "
    optimizer.tell.
See https://facebookresearch.github.io/nevergrad/"
    optimization.html#multiobjective-minimization-with-nevergrad
",
    '''
    def __init__(self, multiobjective_function: tp.Callable[..., tp.ArrayLike], upper_bounds: tp.Optional[tp.ArrayLike] = None) -> None: ...
