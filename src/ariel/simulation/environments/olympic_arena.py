"""Hybrid MuJoCo world combining flat, rugged, and inclined terrain sections."""

from typing import Tuple

import mujoco
import numpy as np
from ariel.utils.mjspec_ops import compute_geom_bounding_box
# from noise import pnoise2
import quaternion as qnp
from ariel.utils.noise_gen import PerlinNoise

USE_DEGREES = False
TERRAIN_COLOR = [0.460, 0.362, 0.216, 1.0]

# np.random.seed(3)


def quaternion_from_axis_angle(axis: str, angle_deg):
    """Compute a unit quaternion from an axis and angle (degrees).

    Parameters
    -----------
    axis : str[x|y|z]
        Which of the 3 axis to turn in to quaternion.
    angle_deg : float
        Number of degrees for the axis.
    """
    if axis == "x":
        axis = [1, 0, 0]
    elif axis == "y":
        axis = [0, 1, 0]
    elif axis == "z":
        axis = [0, 0, 1]

    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    angle_rad = np.deg2rad(angle_deg)
    half_angle = angle_rad / 2
    w = np.cos(half_angle)
    xyz = np.sin(half_angle) * axis

    return [w, *xyz]


class OlympicArena:
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
        rugged_height: float = 0.15,
        # Inclined section parameters
        incline_thickness: float = 0.1,
        incline_degrees: float = -15.0,
        incline_axis: str = "y",
    ):
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
        self.spec = self._build_spec()

    # def _generate_heightmap(self) -> np.ndarray:
    #     size = self.rugged_resolution
    #     freq = self.rugged_scale

    #     noise = np.fromfunction(
    #         np.vectorize(
    #             lambda y, x: pnoise2(
    #                 x / size * freq,
    #                 y / size * freq,
    #                 octaves=6,
    #             )
    #             * self.rugged_hillyness
    #         ),
    #         (size, size),
    #         dtype=float,
    #     )

    #     # Normalize to [0, 1]
    #     # noise = (noise - noise.min()) / (noise.max() - noise.min())
    #     return np.clip(noise, -1, None)
    
    def _generate_heightmap(self) -> np.ndarray:
        size  = self.rugged_resolution
        # freq  = self.rugged_scale
        hill  = self.rugged_hillyness 
        edge_width = getattr(self, "edge_width", 0.1)  # fraction of map size (0..0.5 is sensible)
        
        # Create noise generator
        pnoise = PerlinNoise()

        # Generate a grid of noise
        width, height = size, size
        scale = hill
        noise = pnoise.as_grid(width, height, scale=scale, normalize=False)
        
        # --- Smooth edge mask (0 at borders -> 1 inside) ---
        # Normalized coordinates in [0,1]
        u = np.linspace(0.0, 1.0, size)
        v = np.linspace(0.0, 1.0, size)
        U, V = np.meshgrid(u, v, indexing="xy")

        # Distance to nearest edge
        d = np.minimum.reduce([U, 1.0 - U, V, 1.0 - V])  # 0 at edge, 0.5 at center

        # Map distance to [0,1] over a band of width 'edge_width'
        t = np.clip(d / edge_width, 0.1, 1.0)

        # Smoothstep for a soft transition
        mask = t * t * (3.0 - 2.0 * t)  # smoothstep(0,1,t)
        # mask = 0.5 - 0.5 * np.cos(np.pi * np.clip(d / edge_width, 0.0, 1.0))

        # Apply mask so edges fade to 0 smoothly
        height = noise * mask

        return height

    def _build_spec(self) -> mujoco.MjSpec:
        spec = mujoco.MjSpec()

        spec.option.integrator = int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
        spec.compiler.autolimits = True
        spec.compiler.degree = USE_DEGREES
        spec.compiler.balanceinertia = True
        spec.compiler.discardvisual = False
        spec.visual.global_.offheight = 960
        spec.visual.global_.offwidth = 1280

        # --- Assets ---
        # Grid texture and material for flat sections
        grid_name = "grid"
        spec.add_texture(
            name=grid_name,
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            rgb1=[0.1, 0.2, 0.3],
            rgb2=[0.2, 0.3, 0.4],
            width=600,
            height=600,
        )
        spec.add_material(
            name=grid_name,
            textures=["", f"{grid_name}"],
            texrepeat=[3, 3],
            texuniform=True,
            reflectance=0.1,
        )

        finish_island = "finish line"
        spec.add_texture(
            name=finish_island,
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            rgb1=[0, 0, 0],
            rgb2=[1, 1, 1],
            width=600,
            height=600,
        )
        spec.add_material(
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

        spec.add_hfield(
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

        # --- Lighting ---
        spec.worldbody.add_light(
            name="main_light",
            pos=[1.5, 0, 3],  # Position over the middle of the arena
            castshadow=True,
        )
        spec.worldbody.add_light(
            name="ambient_light",
            pos=[0, 0, 2],
            castshadow=False,
        )

        # --- Section 1: Flat terrain (X: -1.5 to -0.5) ---
        flat_center_x = -1.0
        spec.worldbody.add_geom(
            name="flat_section",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[flat_center_x, 0, -self.flat_thickness / 2],
            size=[
                self.section_length*1.5,
                self.arena_width / 2,
                self.flat_thickness / 2,
            ],
            material=grid_name,
        )

        # --- Section 2: Rugged terrain (X: -0.5 to 0.5) ---
        rugged_center_x = flat_center_x + self.section_length*2.5
        rugged_body = spec.worldbody.add_body(
            pos=[rugged_center_x, 0.0, -0.075],
            name="rugged_section",
        )
        rugged_body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_HFIELD,
            hfieldname=hf_name,
            rgba=TERRAIN_COLOR,
        )

        # --- Section 3: Inclined terrain (X: 0.5 to 1.5) ---
        incline_center_x = rugged_center_x + self.section_length

        # Calculate the height offset for the inclined section
        # We want it to connect smoothly with the rugged section
        incline_quat = quaternion_from_axis_angle(
            self.incline_axis, self.incline_degrees
        )

        # Position the inclined section slightly higher to create a ramp effect
        incline_height = 0.2

        spec.worldbody.add_geom(
            name="inclined_section",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[incline_center_x+0.98, 0, incline_height],
            size=[
                self.section_length,
                self.arena_width / 2,
                self.incline_thickness / 2,
            ],
            quat=incline_quat,
            material=grid_name,
        )

        # --- Arena boundaries (cliffs) ---
        cliff_depth = 2.0
        cliff_width = 0.5
        left_right_start_pos = 1.5

        # End cliff
        spec.worldbody.add_geom(
            name="cliff_end",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[5.43-0.67, 0, incline_height+0.105],
            size=[
                self.arena_width/6,
                self.arena_width/2,
                cliff_depth/10,
            ],
            # rgba=[0.3, 0.2, 0.1, 1.0],  # Dark brown cliff color
            # rgba=[0.3, 0.9, 0.1, 1.0],  # Dark brown cliff color
            material=finish_island
        )
        spec.worldbody.add_geom(
            name="finish_end",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[5.43+0.64, 0, incline_height+0.105],
            size=[
                self.arena_width/6,
                self.arena_width/2,
                cliff_depth/10,
            ],
            # rgba=[0.3, 0.2, 0.1, 1.0],  # Dark brown cliff color
            # rgba=[1, 1, 1, 1],  # Dark brown cliff color
            material=finish_island
        )
        spec.worldbody.add_geom(
            name="finish_white",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[5.42, 0, incline_height+0.106],
            size=[
                self.arena_width/6,
                self.arena_width/1.99,
                cliff_depth/10,
            ],
            # rgba=[0.3, 0.2, 0.1, 1.0],  # Dark brown cliff color
            rgba=[1, 1, 1, 1],  # Dark brown cliff color
            # material=finish_island
        )
        return spec

    def spawn(
        self,
        mj_spec: mujoco.MjSpec,
        spawn_position: list[float] | None = None,
        spawn_orientation: list[float] | None = None,
        *,
        small_gap: float = 0.0,
        correct_for_bounding_box: bool = True,
    ) -> None:
        # Default spawn position
        if spawn_position is None:
            spawn_position = [0, 0, 0]

        # Default spawn orientation
        if spawn_orientation is None:
            spawn_orientation = [0, 0, 0]

        # If correct_for_bounding_box is True, adjust the spawn position
        if correct_for_bounding_box:
            model = mj_spec.compile()
            data = mujoco.MjData(model)
            mujoco.mj_step(model, data, nstep=10)
            min_corner, _ = compute_geom_bounding_box(model, data)
            spawn_position[2] -= min_corner[2]

        # If small_gap is True, add a small gap to the spawn position
        spawn_position[2] += small_gap

        shift = 0  # mujoco uses xyzw instead of wxyz
        spawn_site = self.spec.worldbody.add_site(
            pos=np.array(spawn_position),
            quat=np.round(
                np.roll(
                    qnp.as_float_array(
                        qnp.from_euler_angles([
                            np.deg2rad(spawn_orientation[0]),
                            np.deg2rad(spawn_orientation[1]),
                            np.deg2rad(spawn_orientation[2]),
                        ]),
                    ),
                    shift=shift,
                ),
                decimals=3,
            ),
        )

        spawn = spawn_site.attach_body(
            body=mj_spec.worldbody,
            prefix="robot-",
        )

        spawn.add_freejoint()
        
