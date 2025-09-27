import nevergrad.common.typing as tp
import numpy as np
from . import utils as utils
from _typeshed import Incomplete
from nevergrad.parametrization import discretization as discretization

class Mutator:
    """Class defining mutations, and holding a random state used for random generation."""
    random_state: Incomplete
    def __init__(self, random_state: np.random.RandomState) -> None: ...
    def significantly_mutate(self, v: float, arity: int):
        """Randomly drawn a normal value, and redraw until it's different after discretization by the quantiles
        1/arity, 2/arity, ..., (arity-1)/arity.
        """
    def doerr_discrete_mutation(self, parent: tp.ArrayLike, arity: int = 2) -> tp.ArrayLike:
        """Mutation as in the fast 1+1-ES, Doerr et al. The exponent is 1.5."""
    def doubledoerr_discrete_mutation(self, parent: tp.ArrayLike, max_ratio: float = 1.0, arity: int = 2) -> tp.ArrayLike:
        """Doerr's recommendation above can mutate up to half variables
        in average.
        In our high-arity context, we might need more than that.

        Parameters
        ----------
        parent: array-like
            the point to mutate
        max_ratio: float (between 0 and 1)
            the maximum mutation ratio (careful: this is not an exact ratio)
        """
    def rls_mutation(self, parent: tp.ArrayLike, arity: int = 2) -> tp.ArrayLike:
        """Good old one-variable mutation.

        Parameters
        ----------
        parent: array-like
            the point to mutate
        arity: int
            the number of possible distinct values
        """
    def portfolio_discrete_mutation(self, parent: tp.ArrayLike, intensity: tp.Optional[int] = None, arity: int = 2) -> tp.ArrayLike:
        """Mutation discussed in
        https://arxiv.org/pdf/1606.05551v1.pdf
        We mutate a randomly drawn number of variables on average.
        The mutation is the same for all variables - coordinatewise mutation will be different from this point of view and will make it possible
        to do anisotropic mutations.
        """
    def coordinatewise_mutation(self, parent: tp.ArrayLike, velocity: tp.ArrayLike, boolean_vector: tp.ArrayLike, arity: int) -> tp.ArrayLike:
        """This is the anisotropic counterpart of the classical 1+1 mutations in discrete domains
        with tunable intensity: it is useful for anisotropic adaptivity."""
    def discrete_mutation(self, parent: tp.ArrayLike, arity: int = 2) -> tp.ArrayLike:
        """This is the most classical discrete 1+1 mutation of the evolution literature."""
    def crossover(self, parent: tp.ArrayLike, donor: tp.ArrayLike, rotation: bool = False, crossover_type: str = 'none') -> tp.ArrayLike: ...
    def get_roulette(self, archive: utils.Archive[utils.MultiValue], num: tp.Optional[int] = None) -> tp.Any:
        """Apply a roulette tournament selection."""
