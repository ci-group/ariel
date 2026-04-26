"""MuJoCo world: crater terrain with ruggedness."""

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
    crater_heightmap,
    rugged_heightmap,
)
from ariel.utils.noise_gen import NormMethod


@dataclass
class CraterTerrainWorld(CompoundWorld):
    """A crater terrain world with ruggedness (CompoundWorld)."""

    name: str = "crater-world"

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

    # Crater heightmap parameters
    crater_depth: float = 1.0
    crater_radius: float = 0.3

    # Whether to load precompiled XML (if it exists)
    load_precompiled: bool = True

    def __post_init__(self) -> None:
        # Rugged part of heightmap
        rugged_part = rugged_heightmap(
            self.dims,
            self.scale_of_noise,
            self.normalize,
        )
        rugged_part *= self.height_of_noise

        # Crater part of heightmap
        crater_part = crater_heightmap(
            dims=self.dims,
            crater_depth=self.crater_depth,
            crater_radius=self.crater_radius,
        )

        # Combine parts
        self.floor_heightmap = crater_part + rugged_part

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
