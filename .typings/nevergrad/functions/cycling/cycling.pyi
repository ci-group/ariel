from .. import base as base
from .mensteampursuit import mensteampursuit as mensteampursuit
from .womensteampursuit import womensteampursuit as womensteampursuit

class Cycling(base.ExperimentFunction):
    """
    Team Pursuit Track Cycling Simulator.

    Parameters
    ----------
    strategy: int
        Refers to Transition strategy or Pacing strategy (or both) of the cyclists; this depends on the strategy length.
        Strategy length can only be 30, 31, 61, 22, 23, 45.
        30: mens transition strategy.
        31: mens pacing strategy.
        61: mens transition and pacing strategy combined.
        22: womens transition strategy.
        23: womens pacing strategy.
        45: womens transition and pacing strategy combined.
    """
    def __init__(self, strategy_index: int = 30) -> None: ...

def team_pursuit_simulation(x) -> float: ...
