"""TODO(jmdm): description of script."""

# Third-party libraries
import mujoco
import numpy as np

# Local libraries
from ariel.utils.mjspec_ops import compute_geom_bounding_box

# Global constants
USE_DEGREES = False


class SimpleFlatWorld:
    """Specification for a basic MuJoCo world."""

    def __init__(
        self,
        floor_size: tuple[float, float, float] = (1, 1, 0.1),  # meters
    ) -> None:
        """
        Create a basic specification for MuJoCo.

        Parameters
        ----------
        floor_size : tuple[float, float, float], optional
            The size of the floor geom, by default (1, 1, 0.1)
        """
        # Fixed parameters
        grid_name = "grid"

        # Passed parameters
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

        # --- Assets ---
        spec.add_texture(
            name=grid_name,
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
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
        spec.worldbody.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            material=grid_name,
            size=self.floor_size,
        )

        # Save specification
        self.spec: mujoco.MjSpec = spec

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
        spawn_position : list[float, float, float] | None, optional
            The position (x, y, z) to spawn the robot at, by default (0, 0, 0)
        small_gap : float, optional
            A small gap to add to the spawn position, by default 0.0
        correct_for_bounding_box : bool, optional
            If True, the spawn position will be adjusted to account for the robot's bounding box,
            by default True
        """
        # Default spawn position
        if spawn_position is None:
            spawn_position = [0, 0, 0]

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
