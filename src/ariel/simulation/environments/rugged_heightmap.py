"""TODO(jmdm): description of script."""

from typing import Tuple

import mujoco
import numpy as np
from noise import pnoise2

from ariel.utils.mjspec_ops import compute_geom_bounding_box

USE_DEGREES = False
TERRAIN_COLOR = [0.460, 0.362, 0.216, 1.0]


class RuggedTerrainWorld:
    """MuJoCo world with a Perlin-noise-based rugged heightfield."""

    def __init__(
        self,
        size: Tuple[float, float] = (10.0, 10.0),
        resolution: int = 128,
        scale: float = 8.0,
        hillyness: float = 10.0,
        height: float = 0.5,
    ):
        """
        Initialize the rugged terrain.

        Parameters
        ----------
        size : tuple of float
            Physical size of the terrain in (x, y).
        resolution : int
            Resolution of the heightmap (square grid).
        scale : float
            Frequency scale for Perlin noise.
        hillyness : float
            Amplitude scaling for height.
        height : float
            Maximum height of terrain.
        """
        self.size = size
        self.resolution = resolution
        self.scale = scale
        self.hillyness = hillyness
        self.height = height

        self.heightmap = self._generate_heightmap()
        self.spec = self._build_spec()

    def _generate_heightmap(self) -> np.ndarray:
        """Generate Perlin-based terrain and normalize to [0, 1]."""
        size = self.resolution
        freq = self.scale

        noise = np.fromfunction(
            np.vectorize(
                lambda y, x: pnoise2(
                    x / size * freq,
                    y / size * freq,
                    octaves=6,
                )
                * self.hillyness
            ),
            (size, size),
            dtype=float,
        )

        # Normalize to [0, 1]
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        return noise

    def _build_spec(self) -> mujoco.MjSpec:
        """Create MjSpec with the heightfield and terrain geometry."""
        spec = mujoco.MjSpec()

        spec.option.integrator = int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
        spec.compiler.autolimits = True
        spec.compiler.degree = USE_DEGREES
        spec.compiler.balanceinertia = True
        spec.compiler.discardvisual = False
        spec.visual.global_.offheight = 960
        spec.visual.global_.offwidth = 1280

        # Heightfield
        hf_name = "rugged_field"
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

        # Terrain body and geom
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
