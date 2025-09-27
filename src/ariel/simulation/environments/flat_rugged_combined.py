"""Seamless hybrid MuJoCo world with flat (green) + rugged (brown) terrain."""

from typing import Tuple
import mujoco
import numpy as np
from noise import pnoise2

from ariel.utils.mjspec_ops import compute_geom_bounding_box

USE_DEGREES = False
RUGGED_COLOR = [0.460, 0.362, 0.216, 1.0]  # brown
FLAT_COLOR = [0.2, 0.6, 0.2, 1.0]          # green


class SeamlessWorld:
    """Flat green region that transitions smoothly into rugged terrain."""

    def __init__(
        self,
        flat_extent: float = 3.0,          # meters of flat area before terrain
        size: Tuple[float, float] = (20.0, 10.0),
        resolution: int = 256,
        scale: float = 8.0,
        hillyness: float = 10.0,
        height: float = 0.7,
    ):
        self.flat_extent = flat_extent
        self.size = size
        self.resolution = resolution
        self.scale = scale
        self.hillyness = hillyness
        self.height = height

        self.heightmap = self._generate_heightmap()
        self.spec = self._build_spec()

    def _generate_heightmap(self) -> np.ndarray:
        """Generate heightmap with smooth transition from flat to rugged."""
        n = self.resolution
        freq = self.scale

        def noise_fn(y, x):
            nx, ny = x / n, y / n
            raw = pnoise2(nx * freq, ny * freq, octaves=6) * self.hillyness
            return raw

        noise = np.fromfunction(np.vectorize(noise_fn), (n, n), dtype=float)

        # Normalize to [0,1]
        noise = (noise - noise.min()) / (noise.max() - noise.min())

        # Blending mask along X
        blend = np.linspace(0, 1, n)
        mask = blend.reshape(1, -1)
        heightmap = noise * mask

        return heightmap

    def _build_spec(self) -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        spec.option.integrator = int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
        spec.compiler.autolimits = True
        spec.compiler.degree = USE_DEGREES
        spec.compiler.balanceinertia = True
        spec.compiler.discardvisual = False
        spec.visual.global_.offheight = 960
        spec.visual.global_.offwidth = 1280

        # Heightfield
        hf_name = "seamless_field"
        nrow = ncol = self.resolution
        spec.add_hfield(
            name=hf_name,
            size=[self.size[0] / 2, self.size[1] / 2, self.height, self.height / 10],
            nrow=nrow,
            ncol=ncol,
            userdata=self.heightmap.flatten().tolist(),
        )

        # Add rugged terrain geom
        body = spec.worldbody.add_body(pos=[0.0, 0.0, 0.0], name=hf_name)
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_HFIELD,
            hfieldname=hf_name,
            rgba=RUGGED_COLOR,
            pos=[-self.size[0] / 2 + self.flat_extent, 0, 0],
        )

        # Add flat green plane covering the flat extent
        body.add_geom(
            name="flat_green",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[self.flat_extent, self.size[1] / 2, 0.1],
            pos=[-self.size[0] / 2 + self.flat_extent, 0, 0],
            rgba=FLAT_COLOR,
        )

        # Lighting
        spec.worldbody.add_light(
            name="light",
            pos=[0, 0, 3],
            castshadow=False,
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
        spawn = spawn_site.attach_body(body=mj_spec.worldbody, prefix="robot-")
        spawn.add_freejoint()
