from _typeshed import Incomplete
from torch._inductor.autoheuristic.autoheuristic_utils import AHContext as AHContext, AHMetadata as AHMetadata, AHOperation as AHOperation, CHOICE_COL as CHOICE_COL, Choice as Choice, FEEDBACK_COL as FEEDBACK_COL, Feedback as Feedback, get_metadata_str_from_log as get_metadata_str_from_log
from torch._inductor.autoheuristic.learned_heuristic_controller import LearnedHeuristicController as LearnedHeuristicController
from torch._inductor.ir import ChoiceCaller as ChoiceCaller
from torch._inductor.runtime.runtime_utils import cache_dir as cache_dir
from torch._inductor.utils import get_gpu_shared_memory as get_gpu_shared_memory
from typing import Any, Callable

class LocalFeedback:
    """
    To be able to collect data for a choice, a function providing feedback given a choice has to be provided.
    LocalFeedback can be used when AutoHeuristic should immediately run the function to collect feedback for each choice
    (see pad_mm.py, where the autotuning happens locally, for an example).
    """
    feedback_fn: Incomplete
    def __init__(self, feedback_fn: Callable[[Choice], Feedback]) -> None: ...
    def __call__(self, choice: Choice) -> Feedback: ...

class InconsistentMetadata(Exception):
    """
    Exception that is thrown when AutoHeuristic tries to log data to a file where the metadata stored in the file does
    not match the metadata it would store if the file didn't exist.
    """

class AutoHeuristic:
    """
    AutoHeuristic is a framework that allows one to collect data, learn a heuristic (i.e. a regression tree) and
    generate the heuristic to code. This class allows one to collect data. The collected data can then be used to train
    a heuristic (see torchgen/autoheuristic/).
    """
    collected_feedback: dict[Choice, Feedback]
    fallback: Incomplete
    choices: Incomplete
    feedback: Incomplete
    context: Incomplete
    name: Incomplete
    augment_context: Incomplete
    metadata: Incomplete
    precondition: Incomplete
    log_path: Incomplete
    def __init__(self, fallback: Callable[[], Choice], choices: list[Choice], feedback: LocalFeedback | None, context: AHContext, name: str, augment_context: list[AHOperation] | None = None, precondition: Callable[[AHMetadata, AHContext], bool] | None = None) -> None:
        """
        Initializes an instance of the AutoHeuristic class.

        Args:
            fallback: A callable that returns a Choice when the heuristic is unsure which choice to make, or
            AutoHeuristic is in data collection mode.
            choices: A list of possible choices the heuristic can make.
            feedback: An instance of LocalFeedback that provides feedback for a given choice.
            context: Context to store with each choice and feedback.
            name: A string that identifies the heuristic.
            augment_context: An optional list of AHOperation instances that augment the context.
            precondition: A callable that returns a boolean indicating whether AutoHeuristic should run.
        """
    def satisfies_precondition(self) -> bool: ...
    def get_choice(self) -> Choice:
        """
        Returns the chosen option based on the value of autoheuristic_use.
        If self.name is one of the comma separated strings in autoheuristic_use,
        it queries a learned heuristic to make a decision. Otherwise, it returns the fallback option.
        """
    def get_top_k_choices(self, top_k: int, always_included: list[str] | None = None) -> list[Choice] | None: ...
    def get_collected_feedback(self, choice: Choice) -> Any: ...
    @staticmethod
    def get_device_identifier() -> str: ...
    def get_default_log_path(self) -> str: ...
    def serialize_metadata(self) -> str: ...
    def save_data(self, choice: Choice, feedback_val: Feedback) -> None: ...

class AutoHeuristicSelectAlgorithm(AutoHeuristic):
    """
    AutoHeuristicSelectAlgorithm is a subclass of AutoHeuristic that allows one to collect data and learn a heuristic
    when one wants to use AutoHeuristic for kernel choice selection.
    """
    input_nodes: Incomplete
    choicestr2choice: dict[str, ChoiceCaller]
    def __init__(self, fallback: Callable[[], ChoiceCaller | None], choices: list[ChoiceCaller], input_nodes: list[Any], context: AHContext, name: str, augment_context: list[AHOperation] | None = None, precondition: Callable[[AHMetadata, AHContext], bool] | None = None) -> None:
        """
        The arguments choices, input_nodes and name have to match the ones used in the call to
        autotune_select_algorithm(), e.g. if the following call is made
        autotune_select_algorithm(name, choices, input_nodes, layout), the same name, choices and input_nodes
        have to be used here.
        """
    def register_global_feedback(self, input_nodes: list[Any], choices: list[ChoiceCaller]) -> None:
        """
        Registers a callback in select_algorithm, which is called with the timing of each choice.
        """
    def get_choice_caller(self) -> ChoiceCaller | None: ...
    def get_top_k_choices_caller(self, top_k: int, always_included: list[str] | None = None) -> list[ChoiceCaller] | None: ...
