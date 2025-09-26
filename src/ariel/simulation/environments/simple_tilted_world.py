"""Tilted flat terrain with a specified tilt angle and axis."""

# Third-party libraries
import mujoco
import numpy as np

# Local libraries
from ariel.utils.mjspec_ops import compute_geom_bounding_box

# Global constants
USE_DEGREES = False


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


class TiltedFlatWorld:
    """A simple MuJoCo world with a flat box floor tilted around the X, Y or Z axis."""

    def __init__(
        self,
        floor_size: tuple[float, float, float] = (1, 1, 0.1),  # meters
        tilt_degrees: float = 10.0,
        axis: str = "y",
    ):
        """Create a tilted flat world.

        Parameters
        ----------
        floor_size : tuple[float, float, float], optional
            The size of the floor geom, by default (1, 1, 0.1)
        tilt_degrees : float, optional
            The angle to tilt the floor, by default 10.0
        axis : str, optional
            The axis to tilt around, by default "y"
        """
        # Fixed parameters
        grid_name = "tilted_grid"

        self.floor_size = floor_size
        self.tilt_degrees = tilt_degrees
        self.axis = axis.lower()

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

        # Add a light
        spec.worldbody.add_light(name="light", pos=[0, 0, 2], castshadow=False)

        # Add the tilted floor
        spec.worldbody.add_geom(
            name="tilted_floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=self.floor_size,
            quat=quaternion_from_axis_angle(self.axis, self.tilt_degrees),
        )

        self.spec = spec

    def spawn(
        self,
        mj_spec: mujoco.MjSpec,
        spawn_position: list[float, float, float] | None = None,
        *,
        small_gap: float = 0.01,
        correct_for_bounding_box: bool = True,
    ):
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

        if spawn_position is None:
            spawn_position = [0, 0, 0]

        if correct_for_bounding_box:
            model = mj_spec.compile()
            data = mujoco.MjData(model)
            mujoco.mj_step(model, data, nstep=10)
            min_corner, _ = compute_geom_bounding_box(model, data)
            spawn_position[2] -= min_corner[2]

        spawn_position[2] += small_gap

        site = self.spec.worldbody.add_site(pos=np.array(spawn_position))

        attachment = site.attach_body(body=mj_spec.worldbody, prefix="robot-")

        attachment.add_freejoint()
