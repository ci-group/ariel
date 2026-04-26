"""Pre-built MuJoCo environments."""

from ariel.simulation.environments._amphitheatre import (
    AmphitheatreTerrainWorld,
)
from ariel.simulation.environments._base_world import (
    BaseWorld,
)
from ariel.simulation.environments._compound_world import (
    CompoundWorld,
)
from ariel.simulation.environments._crater import (
    CraterTerrainWorld,
)
from ariel.simulation.environments._rugged import (
    RuggedTerrainWorld,
)
from ariel.simulation.environments._rugged_tilted import (
    RuggedTiltedWorld,
)
from ariel.simulation.environments._simple_flat import (
    SimpleFlatWorld,
)
from ariel.simulation.environments._simple_tilted import (
    SimpleTiltedWorld,
)
from ariel.simulation.environments.olympic_arena import (
    OlympicArena,
)

__all__ = [
    "AmphitheatreTerrainWorld",
    "BaseWorld",
    "CompoundWorld",
    "CraterTerrainWorld",
    "OlympicArena",
    "RuggedTerrainWorld",
    "RuggedTiltedWorld",
    "SimpleFlatWorld",
    "SimpleTiltedWorld",
]
