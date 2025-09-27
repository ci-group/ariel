"""Pyramid on top of flat terrain."""

# Third-party libraries
import mujoco
import numpy as np

# Local libraries
from ariel.utils.mjspec_ops import compute_geom_bounding_box

# Global constants
USE_DEGREES = False


class PyramidWorld:
    """MuJoCo world with stairs placed on a flat terrain."""

    def __init__(
        self,
        pos: list[float] = [0, 0, 0],
        num_stairs: int = 8,
        gap: int = 1,
        name: str = "stair",
        floor_size: tuple[float, float, float] = (10, 10, 0.05),
    ) -> None:
        """
        Create a basic specification for MuJoCo.

        Parameters
        ----------
        floor_size : tuple[float, float, float], optional
            The size of the floor geom, by default (1, 1, 0.1)
        """

        self.SQUARE_LENGTH = 2
        self.V_SIZE = 0.076
        self.H_SIZE = 0.12
        self.H_STEP = self.H_SIZE * 2
        self.V_STEP = self.V_SIZE * 2
        self.SAND = [0.85, 0.75, 0.60, 1.0]
        self.pos = pos
        self.num_stairs = num_stairs
        self.gap = gap
        self.name = name
        self.floor_size = floor_size

        # --- Root ---
        spec = mujoco.MjSpec()
        spec.option.integrator = int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
        spec.compiler.autolimits = True
        spec.compiler.degree = USE_DEGREES
        spec.compiler.balanceinertia = True
        spec.compiler.discardvisual = False
        spec.visual.global_.offheight = 960
        spec.visual.global_.offwidth = 1280

        # Set default geom type to BOX
        spec.default.geom.type = mujoco.mjtGeom.mjGEOM_BOX

        # --- Terrain ---
        grid_name = "grid"
        spec.worldbody.add_geom(
            name=grid_name,
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=self.floor_size,
            rgba=self.SAND,  # Use same color as stairs
        )

        # --- Stairs ---
        self._build_stairs(spec)

        # Save the spec
        self.spec: mujoco.MjSpec = spec

    def _build_stairs(self, spec: mujoco.MjSpec) -> None:
        """Build stair geometry above flat floor with alternating colors."""
        body = spec.worldbody.add_body(pos=self.pos, name=self.name)

        x_beginning = -self.SQUARE_LENGTH + self.H_SIZE
        x_end = self.SQUARE_LENGTH - self.H_SIZE
        y_end = -self.SQUARE_LENGTH + self.H_SIZE
        y_beginning = self.SQUARE_LENGTH - self.H_SIZE

        size_one = [self.H_SIZE, self.SQUARE_LENGTH, self.V_SIZE]
        size_two = [self.SQUARE_LENGTH, self.H_SIZE, self.V_SIZE]

        x_pos_l = [x_beginning, 0, self.gap * self.V_SIZE]
        x_pos_r = [x_end, 0, self.gap * self.V_SIZE]
        y_pos_up = [0, y_beginning, self.gap * self.V_SIZE]
        y_pos_down = [0, y_end, self.gap * self.V_SIZE]

        RED = [1.0, 0.0, 0.0, 1.0]
        BLUE = [0.0, 0.0, 1.0, 1.0]

        for i in range(self.num_stairs):
            color = RED if i % 2 == 0 else BLUE

            size_one[1] = self.SQUARE_LENGTH - self.H_STEP * i
            size_two[0] = self.SQUARE_LENGTH - self.H_STEP * i

            stair_z = self.gap * (self.V_SIZE + self.V_STEP * i)
            x_pos_l[2] = x_pos_r[2] = y_pos_up[2] = y_pos_down[2] = stair_z

            x_pos_l[0] = x_beginning + self.H_STEP * i
            x_pos_r[0] = x_end - self.H_STEP * i
            y_pos_up[1] = y_beginning - self.H_STEP * i
            y_pos_down[1] = y_end + self.H_STEP * i

            body.add_geom(pos=x_pos_l.copy(), size=size_one.copy(), rgba=color)
            body.add_geom(pos=x_pos_r.copy(), size=size_one.copy(), rgba=color)
            body.add_geom(pos=y_pos_up.copy(), size=size_two.copy(), rgba=color)
            body.add_geom(
                pos=y_pos_down.copy(), size=size_two.copy(), rgba=color
            )

        # Final top platform stays same color (optional: pick RED/BLUE or a third color)
        top_color = RED if self.num_stairs % 2 == 0 else BLUE
        top_size = [
            self.SQUARE_LENGTH - self.H_STEP * self.num_stairs,
            self.SQUARE_LENGTH - self.H_STEP * self.num_stairs,
            self.V_SIZE,
        ]
        top_pos = [
            0,
            0,
            self.gap * (self.V_SIZE + self.V_STEP * self.num_stairs),
        ]
        body.add_geom(pos=top_pos, size=top_size, rgba=top_color)

    def spawn(
        self,
        mj_spec: mujoco.MjSpec,
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
        spawn_position : list[float, float, float] | None
            The position (x, y, z) to spawn the robot at, by default (0, 0, 0)
        small_gap : float
            A small gap to add to the spawn position, by default 0.0
        correct_for_bounding_box : bool
            If True, the spawn position will be adjusted to account for the robot's bounding box,
            by default True
        """
        # Default spawn position
        if spawn_position is None:
            spawn_position = [5, 0, 0]

        # If correct_for_bounding_box is True, adjust the spawn position
        if correct_for_bounding_box:
            model = mj_spec.compile()
            data = mujoco.MjData(model)
            mujoco.mj_step(model, data, nstep=10)
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
