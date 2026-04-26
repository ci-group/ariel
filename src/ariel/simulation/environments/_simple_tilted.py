"""MuJoCo world: same as SimpleFlatWorld but tilted."""

# Standard library
from dataclasses import dataclass

# Third-party libraries
import mujoco

# Local libraries
from ariel.parameters.ariel_types import Dimension, Rotation
from ariel.simulation.environments._base_world import BaseWorld
from ariel.utils.mujoco_ops import euler_to_quat_conversion


@dataclass
class SimpleTiltedWorld(BaseWorld):
    """Same as `SimpleFlatWorld` but tilted."""

    name: str = "simple-tilted-world"

    floor_size: Dimension = (10, 10, 1)  # meters (width, height, depth)
    floor_tilt: Rotation = (15, 0, 0)  # degrees (x, y, z)
    floor_rot_sequence: str = "XYZ"  # xyzXYZ, assume intrinsic
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

        # Set floor tilt
        self._set_floor_tilt(
            self.floor_tilt,
            self.floor_rot_sequence,
        )

        # Create checker texture if enabled
        if self.checker_floor is True:
            self._create_checker_texture()
        else:
            self._floor_kwargs["rgba"] = self._terrain_color

        #  Expand the MuJoCo specification
        self._expand_spec()

    def _set_floor_tilt(
        self,
        tilt: Rotation,  # degrees (x, y, z)
        rotation_sequence: str,  # xyzXYZ, assume intrinsic
    ) -> None:
        # Convert rotation from Euler angles (degrees) to quaternion
        self.rotation_as_quat = euler_to_quat_conversion(
            tilt,
            rotation_sequence,
        )

        # Update floor geom parameters
        self._floor_kwargs["quat"] = self.rotation_as_quat

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
