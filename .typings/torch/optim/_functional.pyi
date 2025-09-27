from .adadelta import adadelta as adadelta
from .adagrad import _make_sparse as _make_sparse, adagrad as adagrad
from .adam import adam as adam
from .adamax import adamax as adamax
from .adamw import adamw as adamw
from .asgd import asgd as asgd
from .nadam import nadam as nadam
from .radam import radam as radam
from .rmsprop import rmsprop as rmsprop
from .rprop import rprop as rprop
from .sgd import sgd as sgd
from torch import Tensor as Tensor

def sparse_adam(params: list[Tensor], grads: list[Tensor], exp_avgs: list[Tensor], exp_avg_sqs: list[Tensor], state_steps: list[int], *, eps: float, beta1: float, beta2: float, lr: float, maximize: bool):
    """Functional API that performs Sparse Adam algorithm computation.

    See :class:`~torch.optim.SparseAdam` for details.
    """
