import nevergrad.common.typing as tp
from . import base as base
from _typeshed import Incomplete
from nevergrad.parametrization import parameter as p

class SpecialEvaluationExperiment(base.ExperimentFunction):
    """Experiment which uses one experiment for the optimization,
    and another for the evaluation
    """
    _experiment: Incomplete
    _evaluation: Incomplete
    _pareto_size: Incomplete
    _pareto_subset: Incomplete
    _pareto_subset_tentatives: Incomplete
    def __init__(self, experiment: base.ExperimentFunction, evaluation: base.ExperimentFunction, pareto_size: tp.Optional[int] = None, pareto_subset: str = 'random', pareto_subset_tentatives: int = 30) -> None: ...
    def _delegate_to_experiment(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Loss: ...
    def copy(self) -> SpecialEvaluationExperiment:
        """Creates with new experiments and evaluations"""
    def compute_pseudotime(self, input_parameter: tp.Any, loss: tp.Loss) -> float: ...
    def evaluation_function(self, *recommendations: p.Parameter) -> float: ...
    @property
    def descriptors(self) -> tp.Dict[str, tp.Any]:
        """Description of the function parameterization, as a dict. This base class implementation provides function_class,
        noise_level, transform and dimension
        """
    @classmethod
    def create_crossvalidation_experiments(cls, experiments: tp.List[base.ExperimentFunction], training_only_experiments: tp.Sequence[base.ExperimentFunction] = (), pareto_size: int = 12, pareto_subset_methods: tp.Sequence[str] = ('random', 'loss-covering', 'EPS', 'domain-covering', 'hypervolume')) -> tp.List['SpecialEvaluationExperiment']:
        """Returns a list of MultiExperiment, corresponding to MOO cross-validation:
        Each experiments consist in optimizing all but one of the input ExperimentFunction's,
        and then considering that the score is the performance of the best solution in the
        approximate Pareto front for the excluded ExperimentFunction.

        Parameters
        ----------
        experiments: sequence of ExperimentFunction
            iterable of experiment functions, used for creating the crossvalidation.
        training_only_experiments: sequence of ExperimentFunction
            iterable of experiment functions, used only as training functions in the crossvalidation and never for test..
        pareto_size: int
            if provided, selects a subset of the full pareto front with the given maximum size
        subset: str
            method for selecting the subset (see optimizer.pareto_front)

        """
