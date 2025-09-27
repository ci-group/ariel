import nevergrad as ng
import nevergrad.common.typing as tp
import numpy as np
from .. import base as base
from _typeshed import Incomplete

def impedance_pix(x: tp.ArrayLike, dpix: float, lam: float, ep0: float, epf: float) -> float:
    """Normalized impedance Z/Z0
    ep0, epf:  epsilons in et out
    lam: lambda in nanometers
    dpix: pixel width
    """

class ARCoating(base.ExperimentFunction):
    """
    Parameters
    ----------
    nbslab: int
        number of pixel layers
    d_ar: int
        depth of the structure in nm

    Notes
    -----
    - This is the minimization of reflexion, i.e. this is an anti-reflexive coating problem in normale incidence.
    - Typical parameters (nbslab, d_ar) = (10, 400) or (35, 700) for instance
     d_ar / nbslab must be at least 10
    - the function domain is R^nbslab. The values are then transformed to [epmin, epmax]^nbslab

    Credit
    ------
    This function is based on a code and ideas by Emmanuel Centeno and Antoine Moreau,
    University Clermont Auvergne, CNRS, SIGMA Clermont, Institut Pascal
    """
    lambdas: Incomplete
    dpix: Incomplete
    ep0: int
    epf: int
    epmin: int
    def __init__(self, nbslab: int = 10, d_ar: int = 400, bounding_method: str = 'bouncing') -> None: ...
    def _get_minimum_average_reflexion(self, x: np.ndarray) -> float: ...
    def evaluation_function(self, *recommendations: ng.p.Parameter) -> float: ...
