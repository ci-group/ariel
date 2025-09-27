from _typeshed import Incomplete
from torch._inductor.autoheuristic.autoheuristic_utils import AHContext as AHContext, AHMetadata as AHMetadata, Choice as Choice
from torch._inductor.autoheuristic.learnedheuristic_interface import LearnedHeuristic as LearnedHeuristic
from typing import Any

def find_and_instantiate_subclasses(package_name: str, base_class: Any) -> list[LearnedHeuristic]: ...

class LearnedHeuristicController:
    """
    Class that finds and instantiates all learned heuristics. It also provides
    a way to get the decision of a learned heuristic.
    """
    existing_heuristics: dict[str, list[LearnedHeuristic]]
    heuristics_initialized: bool
    metadata: Incomplete
    context: Incomplete
    def __init__(self, metadata: AHMetadata, context: AHContext) -> None: ...
    def get_heuristics(self, name: str) -> list[LearnedHeuristic]:
        """
        Returns a list of learned heuristics for the given optimization name.
        """
    def get_decision(self) -> Choice | None:
        """
        Returns the decision made by the learned heuristic or None if no heuristic was found or the heuristic is unsure
        which choice to make.
        """
    def get_decisions_ranked(self, top_k: int) -> list[Choice] | None: ...
