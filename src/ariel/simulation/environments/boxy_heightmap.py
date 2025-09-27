"""Boxes extrude from the ground to form a terrain."""

# Third-party libraries
import random

import mujoco as mj
import numpy as np

# Local libraries
from ariel.utils.mjspec_ops import compute_geom_bounding_box

# Global constants
USE_DEGREES = False


class BoxyRugged:
    def __init__(
        self,
        floor_size: tuple[float, float, float] = (10, 10, 0.05),
        pos: list[float] = [0, 0, 0],
    ) -> None:
        """
        Effectively creates a boxy terrain in the specified location.
        """
        grid_name = "boxy_grid"
        self.floor_size = floor_size
        self.pos = pos

        # --- Root ---
        spec = mj.MjSpec()
        spec.option.integrator = int(mj.mjtIntegrator.mjINT_IMPLICITFAST)
        spec.compiler.autolimits = True
        spec.compiler.degree = USE_DEGREES
        spec.compiler.balanceinertia = True
        spec.compiler.discardvisual = False
        spec.visual.global_.offheight = 960
        spec.visual.global_.offwidth = 1280

        # --- Assets ---
        spec.add_texture(
            name=grid_name,
            type=mj.mjtTexture.mjTEXTURE_2D,
            builtin=mj.mjtBuiltin.mjBUILTIN_CHECKER,
            rgb1=[0.1, 0.2, 0.3],
            rgb2=[0.2, 0.3, 0.4],
            width=600,  # pixels
            height=600,  # pixels
        )
        spec.add_material(
            name=grid_name,
            textures=["", f"{grid_name}"],
            texrepeat=[3, 3],
            texuniform=True,
            reflectance=0.2,
        )

        # --- Worldbody ---
        spec.worldbody.add_light(
            name="light",
            pos=[0, 0, 1],
            castshadow=False,
        )

        SQUARE_LENGTH = 2
        CUBE_LENGTH = 0.05
        GRID_SIZE = int(SQUARE_LENGTH / CUBE_LENGTH)
        STEP = CUBE_LENGTH * 2
        BROWN = [0.460, 0.362, 0.216, 1.0]

        if spec == None:
            spec = mj.MjSpec()

        # Defaults
        main = spec.default
        main.geom.type = mj.mjtGeom.mjGEOM_BOX

        # Create tile
        body = spec.worldbody.add_body(pos=self.pos, name=grid_name)

        x_beginning = -SQUARE_LENGTH + CUBE_LENGTH
        y_beginning = SQUARE_LENGTH - CUBE_LENGTH
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                body.add_geom(
                    pos=[
                        x_beginning + i * STEP,
                        y_beginning - j * STEP,
                        random.randint(-1, 1) * CUBE_LENGTH,
                    ],
                    size=[CUBE_LENGTH] * 3,
                    rgba=BROWN,
                )
        self.spec = spec

    def spawn(
        self,
        mj_spec: mj.MjSpec,
        spawn_position: list[float, float, float] | None = None,
        *,
        small_gap: float = 0.0,
        correct_for_bounding_box: bool = True,
    ) -> None:
        """
        Spawn a robot at a specific position in the world.

        Parameters
        ----------
        mj_spec : mujoco.MjSpec
            The MuJoCo specification for the robot.
        spawn_position : list[float, float, float] | None, optional
            The position (x, y, z) to spawn the robot at, by default (0, 0, 0)
        small_gap : float, optional
            A small gap to add to the spawn position, by default 0.0
        correct_for_bounding_box : bool
            If True, the spawn position will be adjusted to account for the robot's bounding box,
            by default True
        """
        # Default spawn position
        if spawn_position is None:
            spawn_position = [0, 0, 0]

        # If correct_for_bounding_box is True, adjust the spawn position
        if correct_for_bounding_box:
            model = mj_spec.compile()
            data = mj.MjData(model)
            mj.mj_step(model, data, nstep=10)
            min_corner, _ = compute_geom_bounding_box(model, data)
            spawn_position[2] -= min_corner[2]

        # If small_gap is True, add a small gap to the spawn position
        spawn_position[2] += small_gap

        spawn_site = self.spec.worldbody.add_site(
            pos=np.array(spawn_position),
        )

        spawn = spawn_site.attach_body(
            body=mj_spec.worldbody,
            prefix="robot-",
        )

        spawn.add_freejoint()
