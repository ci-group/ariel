"""Hybrid MuJoCo world combining flat, rugged, and inclined terrain sections."""

# Standard library
# Third-party libraries
import mujoco
import numpy as np

# Local libraries
from ariel.simulation.environments import BaseWorld
from ariel.utils.noise_gen import PerlinNoise

USE_DEGREES = False
TERRAIN_COLOR = [0.460, 0.362, 0.216, 1.0]

# Global functions
# Warning Control
# Type Checking
# Type Aliases


def quaternion_from_axis_angle(axis: str, angle_deg: float) -> list[float]:
    match axis:
        case "x":
            axis_tup = [1, 0, 0]
        case "y":
            axis_tup = [0, 1, 0]
        case "z":
            axis_tup = [0, 0, 1]
        case _:
            msg = "Unexpected axis name!"
            msg += f" Got {axis=}. Should be 'x', 'y' or 'z'"
            raise ValueError(msg)

    axis_tup = np.asarray(axis_tup, dtype=float)
    axis_tup /= np.linalg.norm(axis_tup)
    angle_rad = np.deg2rad(angle_deg)
    half_angle = angle_rad / 2
    w = np.cos(half_angle)
    xyz = np.sin(half_angle) * axis_tup

    return [w, *xyz]


class OlympicArena(BaseWorld):
    name = "olympic_arena"

    def __init__(
        self,
        # Overall arena parameters
        arena_width: float = 2.0,
        section_length: float = 1.0,
        # Flat section parameters
        flat_thickness: float = 0.1,
        # Rugged section parameters
        rugged_resolution: int = 64,
        rugged_scale: float = 4.0,
        rugged_hillyness: float = 5.0,
        rugged_height: float = 0.07,
        # Inclined section parameters
        incline_thickness: float = 0.1,
        incline_degrees: float = -15.0,
        incline_axis: str = "y",
        *,
        load_precompiled: bool = True,
    ) -> None:
        """
        Hybrid MuJoCo world combining flat, rugged, and inclined terrain sections.
        The world is divided into three sections along the X-axis:
        1. Flat terrain section (left)
        2. Rugged terrain section (middle)
        3. Inclined terrain section (right)

        The arena is surrounded by cliffs on all sides which force the robot to stay in bounds.
        The robot can fall off the terrain, which results to a "death". No specific

        Parameters
        ----------
        arena_width : float
            Total width of the arena along the Y-axis (default: 2.0).
        section_length : float
            Length of each terrain section along the X-axis (default: 1.0).
        flat_thickness : float
            Thickness of the flat terrain section (default: 0.1).
        rugged_resolution : int
            Resolution of the rugged terrain heightmap (default: 64).
        rugged_scale : float
            Scale of the Perlin noise for rugged terrain (default: 4.0).
        rugged_hillyness : float
            Hillyness factor for rugged terrain (default: 5.0).
        rugged_height : float
            Maximum height variation of the rugged terrain (default: 0.07).
        incline_thickness : float
            Thickness of the inclined terrain section (default: 0.1).
        incline_degrees : float
            Angle of inclination in degrees (default: -15.0).
        incline_axis : str
            Axis of inclination ('x', 'y', or 'z') (default: 'y').
        load_precompiled : bool
            Whether to load a precompiled XML if available (default: True).
        """

        # Initialize base class
        super().__init__(name=self.name, load_precompiled=load_precompiled)

        # If precompiled XML was loaded, skip regeneration
        if self.is_precompiled:
            return

        # Store parameters
        self.arena_width = arena_width
        self.section_length = section_length
        self.flat_thickness = flat_thickness
        self.rugged_resolution = rugged_resolution
        self.rugged_scale = rugged_scale
        self.rugged_hillyness = rugged_hillyness
        self.rugged_height = rugged_height
        self.incline_thickness = incline_thickness
        self.incline_degrees = incline_degrees
        self.incline_axis = incline_axis.lower()

        # Generate rugged heightmap
        self.heightmap = self._generate_heightmap()

        # Build the world specification
        self._extend_spec()

    def _generate_heightmap(self) -> np.ndarray:
        size = self.rugged_resolution
        hill = self.rugged_hillyness

        # Fraction of map size (0..0.5 is sensible)
        edge_width = getattr(
            self,
            "edge_width",
            0.1,
        )

        # Create noise generator
        pnoise = PerlinNoise()

        # Generate a grid of noise
        width, height = size, size
        scale = hill
        noise = pnoise.as_grid(width, height, scale=scale, normalize=False)

        # --- Smooth edge mask (0 at borders -> 1 inside) --- #
        # Normalized coordinates in [0,1]
        u = np.linspace(0.0, 1.0, size)
        v = np.linspace(0.0, 1.0, size)
        U, V = np.meshgrid(u, v, indexing="xy")

        # Distance to nearest edge
        d = np.minimum.reduce([
            U,
            1.0 - U,
            V,
            1.0 - V,
        ])  # 0 at edge, 0.5 at center

        # Map distance to [0,1] over a band of width 'edge_width'
        t = np.clip(d / edge_width, 0.1, 1.0)

        # Smoothstep for a soft transition
        mask = t * t * (3.0 - 2.0 * t)  # smoothstep(0,1,t)

        # Apply mask so edges fade to 0 smoothly
        return noise * mask

    def _extend_spec(self) -> None:
        # --- Assets --- #
        # Grid texture and material for flat sections
        grid_name = "grid"
        self.spec.add_texture(
            name=grid_name,
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            rgb1=[0.1, 0.2, 0.3],
            rgb2=[0.2, 0.3, 0.4],
            width=600,
            height=600,
        )
        self.spec.add_material(
            name=grid_name,
            textures=["", f"{grid_name}"],
            texrepeat=[3, 3],
            texuniform=True,
            reflectance=0.1,
        )

        finish_island = "finish line"
        self.spec.add_texture(
            name=finish_island,
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            rgb1=[0, 0, 0],
            rgb2=[1, 1, 1],
            width=600,
            height=600,
        )
        self.spec.add_material(
            name=finish_island,
            textures=["", f"{finish_island}"],
            texrepeat=[8, 8],
            texuniform=True,
            reflectance=0.1,
        )

        # Heightfield for rugged section
        hf_name = "rugged_field"
        nrow = ncol = self.rugged_resolution

        self.heightmap *= 0.1

        self.spec.add_hfield(
            name=hf_name,
            size=[
                self.section_length,
                self.arena_width / 2,
                self.rugged_height,
                self.rugged_height / 10,
            ],
            nrow=nrow,
            ncol=ncol,
            userdata=self.heightmap.flatten().tolist(),
        )

        # --- Section 1: Flat terrain (X: -1.5 to -0.5) --- #
        flat_center_x = -1.0
        self.spec.worldbody.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[flat_center_x, 0, -self.flat_thickness / 2],
            size=[
                self.section_length * 1.5,
                self.arena_width / 2,
                self.flat_thickness / 2,
            ],
            material=grid_name,
        )

        # --- Section 2: Rugged terrain (X: -0.5 to 0.5) --- #
        rugged_center_x = flat_center_x + self.section_length * 2.5
        rugged_body = self.spec.worldbody.add_body(
            pos=[rugged_center_x, 0.0, -0.035],
            name="rugged_section",
        )
        rugged_body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_HFIELD,
            hfieldname=hf_name,
            rgba=TERRAIN_COLOR,
        )

        # --- Section 3: Inclined terrain (X: 0.5 to 1.5) --- #
        incline_center_x = rugged_center_x + self.section_length

        # Calculate the height offset for the inclined section
        # We want it to connect smoothly with the rugged section
        incline_quat = quaternion_from_axis_angle(
            self.incline_axis,
            self.incline_degrees,
        )

        # Position the inclined section slightly higher to create a ramp effect
        incline_height = 0.2

        self.spec.worldbody.add_geom(
            name="inclined_section",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[incline_center_x + 0.98, 0, incline_height],
            size=[
                self.section_length,
                self.arena_width / 2,
                self.incline_thickness / 2,
            ],
            quat=incline_quat,
            material=grid_name,
        )

        # --- Arena boundaries (cliffs) --- #
        cliff_depth = 2.0

        # End cliff
        self.spec.worldbody.add_geom(
            name="cliff_end",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[5.43 - 0.67, 0, incline_height + 0.105],
            size=[
                self.arena_width / 6,
                self.arena_width / 2,
                cliff_depth / 10,
            ],
            material=finish_island,
        )
        self.spec.worldbody.add_geom(
            name="finish_end",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[5.43 + 0.64, 0, incline_height + 0.105],
            size=[
                self.arena_width / 6,
                self.arena_width / 2,
                cliff_depth / 10,
            ],
            material=finish_island,
        )
        self.spec.worldbody.add_geom(
            name="finish_white",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[5.42, 0, incline_height + 0.106],
            size=[
                self.arena_width / 6,
                self.arena_width / 1.99,
                cliff_depth / 10,
            ],
            rgba=[1, 1, 1, 1],  # Dark brown cliff color
        )


if __name__ == "__main__":
    # Compile and save the XML for inspection
    arena = OlympicArena(
        load_precompiled=False,
    )
    arena.store_to_xml()
