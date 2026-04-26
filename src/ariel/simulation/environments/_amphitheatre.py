"""MuJoCo world: an amphitheatre terrain with ruggedness."""

# Standard library
from dataclasses import dataclass

# Third-party libraries
# Local libraries
from ariel.parameters.ariel_types import (
    Dimension,
    Rotation,
)
from ariel.simulation.environments._compound_world import CompoundWorld
from ariel.simulation.environments.heightmap_functions import (
    amphitheater_heightmap,
    rugged_heightmap,
)
from ariel.utils.noise_gen import NormMethod


@dataclass
class AmphitheatreTerrainWorld(CompoundWorld):
    """An amphitheatre terrain world with ruggedness (CompoundWorld)."""

    name: str = "amphitheatre-world"

    floor_size: Dimension = (10, 10, 2)  # meters (width, height, depth)
    floor_tilt: Rotation = (0, 0, 0)  # degrees (x, y, z)
    floor_rot_sequence: str = "XYZ"  # xyzXYZ, assume intrinsic
    checker_floor: bool = False

    # Overall heightmap parameters
    dims: tuple[int, int] = (100, 100)

    # Rugged heightmap parameters
    height_of_noise: float = 0.3
    scale_of_noise: int = 5
    normalize: NormMethod = "none"

    # Amphitheater heightmap parameters
    ring_inner_radius: float = 0.2
    ring_outer_radius: float = 0.45

    # Whether to load precompiled XML (if it exists)
    load_precompiled: bool = True

    def __post_init__(self) -> None:
        """Post-initialization to set up the amphitheatre terrain world.
        Combines an amphitheatre heightmap with a rugged heightmap and initializes
        the CompoundWorld base class."""

        # Rugged part of heightmap
        rugged_part = rugged_heightmap(
            self.dims,
            self.scale_of_noise,
            self.normalize,
        )
        rugged_part *= self.height_of_noise

        # Amphitheater part of heightmap
        amphitheater_part = amphitheater_heightmap(
            dims=self.dims,
            ring_inner_radius=self.ring_inner_radius,
            ring_outer_radius=self.ring_outer_radius,
            cone_height=self.floor_size[2],
        )

        # Combine parts
        self.floor_heightmap = amphitheater_part + rugged_part

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
