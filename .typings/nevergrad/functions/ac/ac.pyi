from ..base import ExperimentFunction as ExperimentFunction
from _typeshed import Incomplete
from nevergrad.parametrization import parameter as parameter

class NgAquacrop(ExperimentFunction):
    num_smts: Incomplete
    max_irr_seasonal: Incomplete
    def __init__(self, num_smts: int, max_irr_seasonal: float) -> None: ...
    def loss(self, smts): ...
