"""MuJoCo world: flat world with a chequerboard floor."""

# Standard library
from dataclasses import dataclass

# Third-party libraries
import mujoco

# Local libraries
from ariel.parameters.ariel_types import Dimension
from ariel.simulation.environments._base_world import BaseWorld


@dataclass
class SimpleFlatWorld(BaseWorld):
    """A flat world with a chequerboard floor."""

    name: str = "simple-flat-world"

    floor_size: Dimension = (10, 10, 1)  # meters (width, height, depth)
    checker_floor: bool = True

    # Whether to load precompiled XML (if it exists)
    load_precompiled: bool = True

    def __post_init__(self) -> None:
        # Initialize base class
        super().__init__(name=self.name, load_precompiled=self.load_precompiled)

        # If precompiled XML was loaded, skip regeneration
        if self.is_precompiled:
            return

        # Floor geom parameters
        self._floor_name = self.mujoco_config.floor_name
        width, height, depth = self.floor_size
        self._floor_kwargs = {
            "name": self._floor_name,
            "size": [width / 2, height / 2, depth / 2],  # mj starts from center
            "type": mujoco.mjtGeom.mjGEOM_PLANE,
        }

        # Create checker texture if enabled
        self._create_checker_texture()

        #  Expand the MuJoCo specification
        self._expand_spec()

    def _create_checker_texture(self) -> None:
        self.spec.add_texture(
            name=self._floor_name,
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            rgb1=[0.1, 0.2, 0.3],
            rgb2=[0.2, 0.3, 0.4],
            width=600,  # pixels
            height=600,  # pixels
        )
        self.spec.add_material(
            name=self._floor_name,
            textures=["", f"{self._floor_name}"],
            texrepeat=[3, 3],
            texuniform=True,
            reflectance=0,
        )

        # Update floor geom parameters
        self._floor_kwargs["material"] = self._floor_name

    def _expand_spec(self) -> None:
        floor = self.spec.worldbody.add_body(
            name=self._floor_name,
            pos=[0, 0, 0],
        )
        floor.add_geom(**self._floor_kwargs)
