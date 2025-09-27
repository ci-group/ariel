import nevergrad.common.typing as tp
from . import callbacks as callbacks, utils as utils
from .base import registry as registry

class MetaModelFailure(ValueError):
    """Sometimes the optimum of the metamodel is at infinity."""

def learn_on_k_best(archive: utils.Archive[utils.MultiValue], k: int, algorithm: str = 'quad', degree: int = 2, shape: tp.Any = None, para: tp.Any = None) -> tp.ArrayLike:
    """Approximate optimum learnt from the k best.

    Parameters
    ----------
    archive: utils.Archive[utils.Value]
    """
