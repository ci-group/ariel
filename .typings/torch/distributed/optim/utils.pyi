from .functional_adadelta import _FunctionalAdadelta as _FunctionalAdadelta
from .functional_adagrad import _FunctionalAdagrad as _FunctionalAdagrad
from .functional_adam import _FunctionalAdam as _FunctionalAdam
from .functional_adamax import _FunctionalAdamax as _FunctionalAdamax
from .functional_adamw import _FunctionalAdamW as _FunctionalAdamW
from .functional_rmsprop import _FunctionalRMSprop as _FunctionalRMSprop
from .functional_rprop import _FunctionalRprop as _FunctionalRprop
from .functional_sgd import _FunctionalSGD as _FunctionalSGD
from _typeshed import Incomplete
from torch import optim as optim

functional_optim_map: Incomplete

def register_functional_optim(key, optim) -> None:
    '''
    Interface to insert a new functional optimizer to functional_optim_map
    ``fn_optim_key`` and ``fn_optimizer`` are user defined. The optimizer and key
    need not be of :class:`torch.optim.Optimizer` (e.g. for custom optimizers)
    Example::
        >>> # import the new functional optimizer
        >>> # xdoctest: +SKIP
        >>> from xyz import fn_optimizer
        >>> from torch.distributed.optim.utils import register_functional_optim
        >>> fn_optim_key = "XYZ_optim"
        >>> register_functional_optim(fn_optim_key, fn_optimizer)
    '''
def as_functional_optim(optim_cls: type, *args, **kwargs): ...
def _create_functional_optim(functional_optim_cls: type, *args, **kwargs): ...
