"""
MuJoCo world: accepts a heightmap and a tilt, good as a base class.

Todo
----
    [ ] Currently the floor geom is added directly to the worldbody.
        This works but prevents us from disabling collision with other bodies
        (since add_exclude doesn't work with geoms, only bodies).
        A workaround is to add the floor geom to a body, but this crashes
        the simulator when using `heightfield`.
    [ ] I can instantiate this class as a dataclass, because it's subclasses
        need to be dataclasses too; which creates an infinite recursion with
        __post_init__.
"""

# Standard library
# Third-party libraries
import mujoco

# Local libraries
from ariel.parameters.ariel_types import (
    Dimension,
    FloatArray,
    Rotation,
)
from ariel.simulation.environments._base_world import BaseWorld
from ariel.simulation.environments.heightmap_functions import flat_heightmap
from ariel.utils.mujoco_ops import euler_to_quat_conversion


class CompoundWorld(BaseWorld):
    """Use CompoundWorld to create worlds with a heightmap and tilt."""

    name: str = "compound-world"

    floor_size: Dimension = (10, 10, 1)  # meters (width, height, depth)
    floor_tilt: Rotation = (0, 0, 0)  # degrees (x, y, z)
    floor_rot_sequence: str = "XYZ"  # xyzXYZ, assume intrinsic
    checker_floor: bool = True

    # Overall heightmap parameters
    dims: tuple[int, int] = (100, 100)
    floor_heightmap: FloatArray | None = None

    _terrain_color = (0.460, 0.362, 0.216, 1.0)  # RGBA

    def __init__(
        self,
        name: str | None = None,
        floor_size: Dimension | None = None,
        floor_tilt: Rotation | None = None,
        floor_rot_sequence: str | None = None,
        dims: tuple[int, int] | None = None,
        floor_heightmap: FloatArray | None = None,
        terrain_color: tuple[float, float, float, float] | None = None,
        *,
        checker_floor: bool | None = None,
        load_precompiled: bool = False,
    ) -> None:
        """
        Initialize a CompoundWorld.

        Parameters
        ----------
        name : str, optional
            Name of the world, by default None
        floor_size : Dimension, optional
            Size of the floor (width, height, depth) in meters, by default None
        floor_tilt : Rotation, optional
            Tilt of the floor (x, y, z) in degrees, by default None
        floor_rot_sequence : str, optional
            Rotation sequence for the tilt, by default None
        dims : tuple[int, int], optional
            Dimensions of the heightmap (rows, cols), by default None
        floor_heightmap : FloatArray, optional
            Heightmap array, by default None
        terrain_color : tuple[float, float, float, float], optional
            RGBA color of the terrain, by default None
        checker_floor : bool, optional
            Whether to use a checker texture for the floor, by default None
        load_precompiled : bool, optional
            Whether to load precompiled XML if it exists, by default False
        """
        # Set name if provided
        if name is not None:
            self.name = name

        # Initialize base class
        super().__init__(name=self.name, load_precompiled=load_precompiled)

        # If precompiled XML was loaded, skip regeneration
        if self.is_precompiled:
            return

        # Override class defaults with provided arguments when given
        if floor_size is not None:
            self.floor_size = floor_size
        if floor_tilt is not None:
            self.floor_tilt = floor_tilt
        if floor_rot_sequence is not None:
            self.floor_rot_sequence = floor_rot_sequence
        if checker_floor is not None:
            self.checker_floor = checker_floor
        if dims is not None:
            self.dims = dims
        if floor_heightmap is not None:
            self.floor_heightmap = floor_heightmap
        if terrain_color is not None:
            self._terrain_color = terrain_color

        # Floor geom parameters
        self._floor_name = self.mujoco_config.floor_name
        self._floor_kwargs = {
            "name": self._floor_name,
            "size": self.floor_size,
        }

        # Use flat heightmap if none is provided
        if self.floor_heightmap is None:
            self.floor_heightmap = flat_heightmap(self.dims)

        # Set heightmap
        self._set_heightmap(self.floor_heightmap)

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

    def _set_heightmap(
        self,
        heightmap: FloatArray,
    ) -> None:
        # Add heightfield to the specification
        width, height, depth = self.floor_size
        nrow, ncol = heightmap.shape
        self.spec.add_hfield(
            name=self._floor_name,
            size=[
                width / 2,
                height / 2,
                depth / 2,  # max depth
                depth / 10,  # min depth, has to be positive
            ],
            nrow=nrow,
            ncol=ncol,
            userdata=heightmap.flatten().tolist(),
        )

        # Update floor geom parameters
        self._floor_kwargs["hfieldname"] = self._floor_name
        self._floor_kwargs["type"] = mujoco.mjtGeom.mjGEOM_HFIELD

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
        self.spec.worldbody.add_geom(**self._floor_kwargs)
        # self.spec = mjspec_deep_copy(self.spec)
