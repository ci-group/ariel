from .cyclist import Cyclist as Cyclist
from .simulationresult import simulationresult as simulationresult
from .teampursuit import teampursuit as teampursuit
from _typeshed import Incomplete

class womensteampursuit(teampursuit):
    team_size: int
    race_distance: int
    lap_distance: int
    race_segments: Incomplete
    maximum_transitions: Incomplete
    team: Incomplete
    def __init__(self) -> None: ...
    def simulate(self, transition_strategy, pacing_strategy): ...
