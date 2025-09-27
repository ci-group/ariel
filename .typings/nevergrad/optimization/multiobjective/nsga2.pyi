import nevergrad.common.typing as tp
from _typeshed import Incomplete
from nevergrad.parametrization import parameter as p

logger: Incomplete

class CrowdingDistance:
    """This class implements the calculation of crowding distance for NSGA-II."""
    def accumulate_distance_per_objective(self, front: tp.List[p.Parameter], i: int): ...
    def compute_distance(self, front: tp.List[p.Parameter]):
        """This function assigns the crowding distance to the solutions.
        :param front: The list of solutions.
        """
    def sort(self, candidates: tp.List[p.Parameter], in_place: bool = True) -> tp.List[p.Parameter]: ...

class FastNonDominatedRanking:
    """Non-dominated ranking of NSGA-II proposed by Deb et al., see [Deb2002]"""
    def compare(self, candidate1: p.Parameter, candidate2: p.Parameter) -> int:
        """Compare the domainance relation of two candidates.

        :param candidate1: Candidate.
        :param candidate2: Candidate.
        """
    def compute_ranking(self, candidates: tp.List[p.Parameter], k: tp.Optional[int] = None) -> tp.List[tp.List[p.Parameter]]:
        """Compute ranking of candidates.

        :param candidates: List of candidates.
        :param k: Number of individuals.
        """

def rank(population: tp.List[p.Parameter], n_selected: tp.Optional[int] = None) -> tp.Dict[str, tp.Tuple[int, int, float]]:
    """implements the multi-objective ranking function of NSGA-II."""
