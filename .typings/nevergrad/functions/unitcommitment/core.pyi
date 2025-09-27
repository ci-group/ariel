from ..base import ExperimentFunction as ExperimentFunction
from _typeshed import Incomplete

class UnitCommitmentProblem(ExperimentFunction):
    """Model that uses conventional implementation for semi-continuous variables
    The model is adopted from Pyomo model 1: Conventional implementation for semi-continuous variables
    (https://jckantor.github.io/ND-Pyomo-Cookbook/04.06-Unit-Commitment.html)
    The constraints are added to the objective with a heavy penalty for violation.

    Parameters
    ----------
    num_timepoints: int
        number of time points
    num_generators: int
        number of generators
    penalty_weight: float
        weight to penalize for violation of constraints
    """
    num_timepoints: Incomplete
    demands: Incomplete
    num_generators: Incomplete
    p_max: Incomplete
    p_min: Incomplete
    cost_a: Incomplete
    cost_b: Incomplete
    penalty_weight: Incomplete
    def __init__(self, problem_name: str = 'semi-continuous', num_timepoints: int = 13, num_generators: int = 3, penalty_weight: float = 10000) -> None: ...
    def unit_commitment_obj_with_penalization(self, operational_output, operational_states): ...
