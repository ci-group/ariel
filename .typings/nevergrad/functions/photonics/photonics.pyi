import numpy as np
import typing as tp
from _typeshed import Incomplete

def trapezoid(a, b): ...
def bragg(X: np.ndarray) -> float:
    """
    Cost function for the Bragg mirror problem: maximizing the reflection
    when the refractive index are given for all the layers.
    Input: a vector whose components represent each the thickness of each
    layer.
    https://hal.archives-ouvertes.fr/hal-02613161
    """
def chirped(X: np.ndarray) -> float: ...
def cascade(T: np.ndarray, U: np.ndarray) -> np.ndarray: ...
def c_bas(A: np.ndarray, V: np.ndarray, h: float) -> np.ndarray: ...
def marche(a: float, b: float, p: float, n: int, x: float) -> np.ndarray: ...
def creneau(k0: float, a0: float, pol: float, e1: float, e2: float, a: float, n: int, x0: float) -> tuple[np.ndarray, np.ndarray]: ...
def homogene(k0: float, a0: float, pol: float, epsilon: float, n: int) -> tuple[np.ndarray, np.ndarray]: ...
def interface(P: np.ndarray, Q: np.ndarray) -> np.ndarray: ...
def morpho(X: np.ndarray) -> float: ...

i: Incomplete

def epscSi(lam: np.ndarray) -> np.ndarray: ...
def cascade2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    '''
    This function takes two 2x2 matrices A and B, that are assumed to be scattering matrices
    and combines them assuming A is the "upper" one, and B the "lower" one, physically.
    The result is a 2x2 scattering matrix.
    '''
def solar(lam: np.ndarray) -> np.ndarray: ...
def absorption(lam: float, epsilon: np.ndarray, mu: np.ndarray, type_: np.ndarray, hauteur: np.ndarray, pol: int, theta: float) -> np.ndarray: ...
def cf_photosic_reference(X: np.ndarray) -> float:
    """vector X is only the thicknesses of each layers, because the materials (so the epislon)
    are imposed by the function. This is similar in the chirped function.
    """
def cf_photosic_realistic(eps_and_d: np.ndarray) -> float:
    """eps_and_d is a vector composed in a first part with the epsilon values
    (the material used in each one of the layers), and in a second part with the
    thicknesses of each one of the layers, like in Bragg.
    Any number of layers can work. Basically I used between 4 and 50 layers,
    and the best results are generally obtained when the structure has between 10 and 20 layers.
    The epsilon values are generally comprised between 1.00 and 9.00.
    """

first_time_ceviche: bool
model: Incomplete
no_neg: bool

def ceviche(x: np.ndarray, benchmark_type: int = 0, discretize: bool = False, wantgrad: bool = False, wantfields: bool = False) -> tp.Any:
    """
    x = 2d or 3d array of scalars
     Inputs:
    1. benchmark_type = 0 1 2 or 3, depending on which benchmark you want
    2. discretize = True if we want to force x to be in {0,1} (just checks sign(x-0.5) )
    3. wantgrad = True if we want to know the gradient
    4. wantfields = True if we want the fields of the simulation
     Returns:
    1. the loss (to be minimized)
    2. the gradient or none (depending on wantgrad)
    3. the fields or none (depending on wantfields
    """
