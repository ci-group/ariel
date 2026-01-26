"""MuJoCo world: a combination of rugged and tilted world."""

# Standard library
from dataclasses import dataclass

# Third-party libraries
# Local libraries
from ariel.parameters.ariel_types import Dimension, Rotation
from ariel.simulation.environments._compound_world import CompoundWorld
from ariel.simulation.environments.heightmap_functions import (
    rugged_heightmap,
    smooth_edges_heightmap,
)
from ariel.utils.noise_gen import NormMethod


@dataclass
class RuggedTiltedWorld(CompoundWorld):
    """A combination of rugged and tilted world (CompoundWorld)."""

    name: str = "rugged-tilted-world"

    floor_size: Dimension = (10, 10, 1)  # meters (width, height, depth)
    floor_tilt: Rotation = (15, 0, 0)  # degrees (x, y, z)
    floor_rot_sequence: str = "XYZ"  # xyzXYZ, assume intrinsic
    checker_floor: bool = False

    # Rugged heightmap parameters
    dims: tuple[int, int] = (100, 100)
    scale_of_noise: int = 4
    normalize: NormMethod = "none"

    # Whether to load precompiled XML (if it exists)
    load_precompiled: bool = True

    def __post_init__(self) -> None:
        # Create heightmap
        self.floor_heightmap = rugged_heightmap(
            self.dims,
            self.scale_of_noise,
            self.normalize,
        )
        self.floor_heightmap *= smooth_edges_heightmap(
            self.dims,
            edge_width=0.2,
        )

        # Initialize base class
        super().__init__(
            name=self.name,
            floor_size=self.floor_size,
            floor_tilt=self.floor_tilt,
            floor_rot_sequence=self.floor_rot_sequence,
            checker_floor=self.checker_floor,
            floor_heightmap=self.floor_heightmap,
            load_precompiled=self.load_precompiled,
        )
