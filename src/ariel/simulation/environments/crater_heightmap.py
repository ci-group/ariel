"""TODO(jmdm): description of script."""

from typing import Tuple

import mujoco
import numpy as np
from noise import pnoise2

from ariel.utils.mjspec_ops import compute_geom_bounding_box

USE_DEGREES = False
TERRAIN_COLOR = [0.5, 0.4, 0.3, 1.0]


class CraterTerrainWorld:
    """MuJoCo world with a crater-like terrain using a heightfield."""

    def __init__(
        self,
        size: Tuple[float, float] = (10.0, 10.0),
        resolution: int = 128,
        crater_depth: float = 3,
        crater_radius: float = 5,
        height: float = 5,
        ruggedness: float = 0.01,
    ):
        """
        Create a crater terrain heightfield.

        Parameters
        ----------
        size : Tuple[float, float]
            Physical size of the terrain in (x, y) meters.
        resolution : int
            Heightmap resolution.
        crater_depth : float
            Maximum depth of the crater (0–1).
        crater_radius : float
            Crater radius (in normalized grid units, max 0.5).
        height : float
            Maximum elevation in meters for MuJoCo scaling.
        """
        self.size = size
        self.resolution = resolution
        self.crater_depth = crater_depth
        self.crater_radius = crater_radius
        self.height = height
        self.ruggedness = ruggedness

        self.heightmap = self._generate_heightmap()
        self.spec = self._build_spec()

    def _generate_heightmap(self) -> np.ndarray:
        """Generate a rugged conical bowl using radial slope + Perlin noise."""
        res = self.resolution
        y, x = np.mgrid[0:res, 0:res]
        x = x / res
        y = y / res

        # Elliptical cone shape
        a = self.crater_radius
        b = self.crater_radius

        # Base conical height
        r = np.sqrt(((x - 0.5) / a) ** 2 + ((y - 0.5) / b) ** 2)
        heightmap = self.crater_depth * r
        heightmap = np.clip(heightmap, 0.0, 1.0)

        if self.ruggedness > 0.0:
            # Generate Perlin noise on same grid
            freq = 6  # adjust to control bump frequency
            noise = np.fromfunction(
                np.vectorize(
                    lambda j, i: pnoise2(
                        i / res * freq, j / res * freq, octaves=3
                    )
                ),
                (res, res),
                dtype=float,
            )
            noise = (noise - noise.min()) / (
                noise.max() - noise.min()
            )  # Normalize to [0, 1]
            noise -= 0.5  # Center at 0

            # Add scaled noise to the cone
            heightmap += self.ruggedness * noise

            # Final clip to [0, 1] to avoid rendering issues
            heightmap = np.clip(heightmap, 0.0, 1.0)

        return heightmap

    def _build_spec(self) -> mujoco.MjSpec:
        """Create the MuJoCo spec with crater heightfield."""
        spec = mujoco.MjSpec()

        spec.option.integrator = int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
        spec.compiler.autolimits = True
        spec.compiler.degree = USE_DEGREES
        spec.compiler.balanceinertia = True
        spec.compiler.discardvisual = False
        spec.visual.global_.offheight = 960
        spec.visual.global_.offwidth = 1280

        hf_name = "crater_field"
        nrow = ncol = self.resolution

        spec.add_hfield(
            name=hf_name,
            size=[
                self.size[0] / 2,
                self.size[1] / 2,
                self.height,
                self.height / 10,
            ],
            nrow=nrow,
            ncol=ncol,
            userdata=self.heightmap.flatten().tolist(),
        )

        body = spec.worldbody.add_body(
            pos=[0.0, 0.0, 0.0],
            name=hf_name,
        )
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_HFIELD,
            hfieldname=hf_name,
            rgba=TERRAIN_COLOR,
        )

        return spec

    def spawn(
        self,
        mj_spec: mujoco.MjSpec,
        spawn_position: list[float] | None = None,
        *,
        small_gap: float = 0.0,
        correct_for_bounding_box: bool = True,
    ) -> None:
        """Spawn a robot inside the amphitheater world.

        Parameters
        ----------

        mj_spec : mujoco.MjSpec
            The mujoco specification of the entity you want to spawn in to the world
        spawn_position : list[float] | None
            The spawn position of the entity. [0 ,0 ,0] by default.
        small_gap : float
            Add a small gap between the entity and the ground. This can help avoid physics glitches.
        correct_for_bounding_box : bool
            In some environments, depending on the spawn_position, the bounding box might spawn inside
            the ground. If enabled, this will automatically adjust the spawn position to avoid that.
        """
        if spawn_position is None:
            spawn_position = [0.0, 0.0, 0.0]

        if correct_for_bounding_box:
            model = mj_spec.compile()
            data = mujoco.MjData(model)
            mujoco.mj_step(model, data, nstep=10)
            min_corner, _ = compute_geom_bounding_box(model, data)
            spawn_position[2] -= min_corner[2]

        spawn_position[2] += small_gap

        spawn_site = self.spec.worldbody.add_site(pos=np.array(spawn_position))
        spawn = spawn_site.attach_body(
            body=mj_spec.worldbody,
            prefix="robot-",
        )
        spawn.add_freejoint()
