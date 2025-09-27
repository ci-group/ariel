"""TODO(jmdm): description of script."""

from typing import Tuple

import mujoco
import numpy as np
from noise import pnoise2

from ariel.utils.mjspec_ops import compute_geom_bounding_box

USE_DEGREES = False
TERRAIN_COLOR = [0.5, 0.4, 0.3, 1.0]


class AmphitheatreTerrainWorld:
    """MuJoCo world with an amphitheater-shaped terrain."""

    def __init__(
        self,
        size: Tuple[float, float] = (10.0, 10.0),
        resolution: int = 128,
        ring_inner_radius: float = 0.2,
        ring_outer_radius: float = 0.45,
        cone_height: float = 1.0,
        ruggedness: float = 0.05,
        height: float = 3.0,
    ):
        """
        Initialize the amphitheater terrain.

        Parameters
        ----------
        size : Tuple[float, float]
            Physical terrain size (x, y).
        resolution : int
            Number of heightmap pixels along one axis.
        ring_inner_radius : float
            Radius (normalized) of the flat inner region [0,–0.5].
        ring_outer_radius : float
            Radius where the outer ring ends [> inner radius].
        cone_height : float
            Total height difference from flat base to outer ring.
        ruggedness : float
            Amplitude of added Perlin noise.
        height : float
            Vertical scaling in MuJoCo (meters).
        """
        self.size = size
        self.resolution = resolution
        self.ring_inner_radius = ring_inner_radius
        self.ring_outer_radius = ring_outer_radius
        self.cone_height = cone_height
        self.ruggedness = ruggedness
        self.height = height

        self.heightmap = self._generate_heightmap()
        self.spec = self._build_spec()

    def _generate_heightmap(self) -> np.ndarray:
        """Generate an amphitheater-style terrain with flat base and sloped ring.

        Returns
        --------
        np.ndarray
            The heightmap of the amphitheater environment.
        """
        res = self.resolution
        y, x = np.mgrid[0:res, 0:res]
        x = x / res
        y = y / res

        # Radial distance from center
        r = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)

        r0 = self.ring_inner_radius
        r1 = self.ring_outer_radius
        d = self.cone_height

        # Piecewise slope: flat -> conical rise -> plateau
        heightmap = np.piecewise(
            r,
            [r <= r0, (r > r0) & (r <= r1), r > r1],
            [0.0, lambda r: d * (r - r0) / (r1 - r0), d],
        )

        # Add Perlin noise for ruggedness
        if self.ruggedness > 0:
            freq = 4.0
            noise = np.fromfunction(
                np.vectorize(
                    lambda j, i: pnoise2(
                        i / res * freq, j / res * freq, octaves=3
                    )
                ),
                (res, res),
                dtype=float,
            )
            noise = (noise - noise.min()) / (noise.max() - noise.min()) - 0.5
            heightmap += self.ruggedness * noise
            heightmap = np.clip(heightmap, 0.0, 1.0)

        return heightmap

    def _build_spec(self) -> mujoco.MjSpec:
        """Create MuJoCo MjSpec with the amphitheater heightfield.

        Returns
        -------
        mujoco.MjSpec
            Creates the mujoco specification for the amphitheater environment
        """
        spec = mujoco.MjSpec()

        # Compiler and visual settings
        spec.option.integrator = int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
        spec.compiler.autolimits = True
        spec.compiler.degree = USE_DEGREES
        spec.compiler.balanceinertia = True
        spec.compiler.discardvisual = False
        spec.visual.global_.offheight = 960
        spec.visual.global_.offwidth = 1280

        # Add heightfield asset
        hf_name = "amphitheater_field"
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

        # Add terrain geom to world
        body = spec.worldbody.add_body(
            pos=[0.0, 0.0, 0.0],
            name="amphitheater_body",
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
