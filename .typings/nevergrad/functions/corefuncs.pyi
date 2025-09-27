import nevergrad.common.typing as tp
import numpy as np
from _typeshed import Incomplete
from nevergrad.common.decorators import Registry as Registry

registry: Registry[tp.Callable[[np.ndarray], float]]

class BonnansFunction:
    N: Incomplete
    M: Incomplete
    A: Incomplete
    y: Incomplete
    def __init__(self, index: int, M: int = 100, N: int = 100) -> None: ...
    def __call__(self, x: tp.ArrayLike) -> float: ...

class DiscreteFunction:
    _arity: Incomplete
    _func: Incomplete
    def __init__(self, name: str, arity: int = 2) -> None:
        """Returns a classical discrete function for test, in the domain {0,1,...,arity-1}^d.
        The name can be onemax, leadingones, or jump.

        onemax(x) is the most classical case of discrete functions, adapted to minimization.
        It is originally designed for lists of bits. It just counts the number of 1,
        and returns len(x) - number of ones. However, the present function perturbates the location of the
        optimum, so that tests can not be easily biased by a wrong initialization. So the optimum,
        instead of being located at (1,1,...,1), is located at (0,1,2,...,arity-1,0,1,2,...).

        leadingones is the second most classical discrete function, adapted for minimization.
        Before perturbation of the location of the optimum as above,
        it returns len(x) - number of initial 1. I.e.
        leadingones([0 1 1 1]) = 4,
        leadingones([1 1 1 1]) = 0,
        leadingones([1 0 0 0]) = 3.
        The present Leadingones function uses a perturbation as documented above for OneMax: we count the number
        of initial correct values, a correct values being 0 for variable 1, 1 for variable 2, 2 for variable 3, and
        so on.

        There exists variants of jump functions: the principle of a jump function is that local descent does not succeed.
        Jumps are necessary. We are here in minimization, hence a formulation slightly different from most discrete optimization
        papers, which usually assume maximization. We use the same perturbation as detailed above for leadingones and onemax,
        i.e. the optimum is located at (0,1,2,...,arity-1,0,1,2,...).
        """
    def __call__(self, x: tp.ArrayLike) -> float: ...
    def onemax(self, x: tp.ArrayLike) -> float: ...
    def leadingones(self, x: tp.ArrayLike) -> float: ...
    def jump(self, x: tp.ArrayLike) -> float: ...

def _styblinksitang(x: np.ndarray, noise: float) -> float:
    """Classical function for testing noisy optimization."""

class DelayedSphere:
    def __call__(self, x: np.ndarray) -> float: ...
    def compute_pseudotime(self, input_parameter: tp.Any, value: float) -> float: ...

@registry.register
def sphere(x: np.ndarray) -> float:
    """The most classical continuous optimization testbed.

    If you do not solve that one then you have a bug."""
@registry.register
def sphere1(x: np.ndarray) -> float:
    """Translated sphere function."""
@registry.register
def sphere2(x: np.ndarray) -> float:
    """A bit more translated sphere function."""
@registry.register
def sphere4(x: np.ndarray) -> float:
    """Even more translated sphere function."""
@registry.register
def maxdeceptive(x: np.ndarray) -> float: ...
@registry.register
def sumdeceptive(x: np.ndarray) -> float: ...
@registry.register
def altcigar(x: np.ndarray) -> float:
    """Similar to cigar, but variables in inverse order.

    E.g. for pointing out algorithms not invariant to the order of variables."""
@registry.register
def discus(x: np.ndarray) -> float:
    """Only one variable is very penalized."""
@registry.register
def cigar(x: np.ndarray) -> float:
    """Classical example of ill conditioned function.

    The other classical example is ellipsoid.
    """
@registry.register
def bentcigar(x: np.ndarray) -> float:
    """Classical example of ill conditioned function, but bent."""
@registry.register
def multipeak(x: np.ndarray) -> float:
    """Inspired by M. Gallagher's Gaussian peaks function."""
@registry.register
def altellipsoid(y: np.ndarray) -> float:
    """Similar to Ellipsoid, but variables in inverse order.

    E.g. for pointing out algorithms not invariant to the order of variables."""
def step(s: float) -> float: ...
@registry.register
def stepellipsoid(x: np.ndarray) -> float:
    """Classical example of ill conditioned function.

    But we add a 'step', i.e. we set the gradient to zero everywhere.
    Compared to some existing testbeds, we decided to have infinitely many steps.
    """
@registry.register
def ellipsoid(x: np.ndarray) -> float:
    """Classical example of ill conditioned function.

    The other classical example is cigar.
    """
@registry.register
def rastrigin(x: np.ndarray) -> float:
    """Classical multimodal function."""
@registry.register
def bucherastrigin(x: np.ndarray) -> float:
    """Classical multimodal function. No box-constraint penalization here."""
@registry.register
def doublelinearslope(x: np.ndarray) -> float:
    """We decided to use two linear slopes rather than having a constraint artificially added for
    not having the optimum at infinity."""
@registry.register
def stepdoublelinearslope(x: np.ndarray) -> float: ...
@registry.register
def hm(x: np.ndarray) -> float:
    """New multimodal function (proposed for Nevergrad)."""
@registry.register
def rosenbrock(x: np.ndarray) -> float: ...
@registry.register
def ackley(x: np.ndarray) -> float: ...
@registry.register
def schwefel_1_2(x: np.ndarray) -> float: ...
@registry.register
def griewank(x: np.ndarray) -> float:
    """Multimodal function, often used in Bayesian optimization."""
@registry.register
def deceptiveillcond(x: np.ndarray) -> float:
    """An extreme ill conditioned functions. Most algorithms fail on this.

    The condition number increases to infinity as we get closer to the optimum."""
@registry.register
def deceptivepath(x: np.ndarray) -> float:
    """A function which needs following a long path. Most algorithms fail on this.

    The path becomes thiner as we get closer to the optimum."""
@registry.register
def deceptivemultimodal(x: np.ndarray) -> float:
    """Infinitely many local optima, as we get closer to the optimum."""
@registry.register
def lunacek(x: np.ndarray) -> float:
    """Multimodal function.

    Based on https://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/lunacek.html."""
def genzcornerpeak(y: np.ndarray) -> float:
    """One of the Genz functions, originally used in integration,

    tested in optim because why not."""
def minusgenzcornerpeak(y: np.ndarray) -> float:
    """One of the Genz functions, originally used in integration,

    tested in optim because why not."""
@registry.register
def genzgaussianpeakintegral(x: np.ndarray) -> float:
    """One of the Genz functions, originally used in integration,

    tested in optim because why not."""
@registry.register
def minusgenzgaussianpeakintegral(x: np.ndarray) -> float:
    """One of the Genz functions, originally used in integration,

    tested in optim because why not."""
@registry.register
def slope(x: np.ndarray) -> float: ...
@registry.register
def linear(x: np.ndarray) -> float: ...
@registry.register
def st0(x: np.ndarray) -> float:
    """Styblinksitang function with 0 noise."""
@registry.register
def st1(x: np.ndarray) -> float:
    """Styblinksitang function with noise 1."""
@registry.register
def st10(x: np.ndarray) -> float:
    """Styblinksitang function with noise 10."""
@registry.register
def st100(x: np.ndarray) -> float:
    """Styblinksitang function with noise 100."""
