"""
MuJoCo world: base class for MuJoCo world specifications.

References
----------
    [1] https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mju-euler2quat

Todo
----
    [ ] Document the class methods
"""

# Standard library
from abc import abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import cast

# Third-party libraries
import mujoco as mj
import numpy as np

# Local libraries
from ariel import log
from ariel.parameters.ariel_types import Position, Rotation
from ariel.parameters.mujoco_params import MujocoConfig
from ariel.utils.mujoco_ops import euler_to_quat_conversion


class BaseWorld:
    """Base class for MuJoCo world specifications."""

    name: str = "base-world"

    spawns: int = 0
    spawn_prefix: str = "robot"
    default_spawn_position: Position = (0, 0, 0)  # x, y, z
    default_spawn_rotation: Rotation = (0, 0, 0)  # x, y, z

    is_precompiled: bool = False

    def __init__(
        self,
        name: str | None = None,
        mujoco_config: MujocoConfig | None = None,
        *,
        load_precompiled: bool = True,
    ) -> None:
        """
        Initialize the world specification.

        Parameters
        ----------
        name : str, optional
            Name of the world, by default None
        mujoco_config : MujocoConfig, optional
            Configuration parameters for MuJoCo, by default None
        load_precompiled : bool, optional
            Whether to load a precompiled XML file if available, by default True
        """

        # Use default mujoco config if none is provided
        if mujoco_config is None:
            self.mujoco_config = MujocoConfig()

        # Set world name
        if name is not None:
            self.name = name

        # Load precompiled XML if requested
        if load_precompiled is True:
            log.debug("Attempting to load precompiled XML...")
            self.is_precompiled = self.load_from_xml()
            if self.is_precompiled:
                log.debug("Precompiled XML loaded successfully.")
                return

        # Build and save specification
        self.spec: mj.MjSpec = self._init_spec()

    @abstractmethod
    def _expand_spec(self) -> None:
        """Expand the world specification with additional elements."""

    def _init_spec(self) -> mj.MjSpec:
        """Initialize the MuJoCo specification."""

        spec = mj.MjSpec()

        # Model name
        spec.modelname = self.name.replace("-", " ").title()

        # Copy during attach
        spec.copy_during_attach = True

        # --- Option --- #
        spec.option.integrator = self.mujoco_config.integrator

        # --- Compiler --- #
        spec.compiler.autolimits = self.mujoco_config.autolimits
        spec.compiler.balanceinertia = self.mujoco_config.balanceinertia
        spec.compiler.degree = self.mujoco_config.degree

        # --- Visual --- #
        spec.visual.global_.offheight = self.mujoco_config.offheight
        spec.visual.global_.offwidth = self.mujoco_config.offwidth

        # Headlight: diffuse, ambient, specular
        spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
        spec.visual.headlight.ambient = [0.1, 0.1, 0.1]
        spec.visual.headlight.specular = [0.9, 0.9, 0.9]

        # RGBA and Global
        spec.visual.rgba.force = [1, 0, 0, 1]
        spec.visual.rgba.haze = [0.15, 0.25, 0.35, 1]
        spec.visual.global_.azimuth = 140
        spec.visual.global_.elevation = -20

        # Map and Scale
        spec.visual.map.force = 0.01
        spec.visual.scale.forcewidth = 0.3
        spec.visual.scale.contactwidth = 0.5
        spec.visual.scale.contactheight = 0.2

        # Quality
        spec.visual.quality.shadowsize = 8192

        # --- Assets ---
        # Skybox Texture
        spec.add_texture(
            type=mj.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mj.mjtBuiltin.mjBUILTIN_GRADIENT,
            rgb1=[
                61 / 255,
                163 / 255,
                179 / 255,
            ],
            rgb2=[
                82 / 255,
                57 / 255,
                153 / 255,
            ],
            width=512,
            height=3072,
        )

        # Add a default light source
        spec.worldbody.add_light(
            name="light",
            pos=[0, 0, 1],
            castshadow=False,
            type=mj.mjtLightType.mjLIGHT_DIRECTIONAL,
        )

        # Add ortho camera and normal camera
        spec.worldbody.add_camera(
            name="ortho-cam",
            orthographic=True,
            pos=[-5, 0, 5],
            xyaxes=[0, -1, 0, 0.75, 0, 0.75],
            fovy=5,
        )

        spec.worldbody.add_camera(
            name="pretty-cam",
            orthographic=False,
            pos=[-0.015, -3.003, 1.765],
            xyaxes=[1.000, -0.005, -0.000, 0.002, 0.507, 0.862],
            fovy=45,
        )
        return spec

    def _find_lowest_position(
        self,
        spawn_name: str,
    ) -> float:
        """Find the lowest Z position of the spawned robot.

        Parameters
        ----------
        spawn_name : str
            Prefix name of the spawned robot to identify its geometries.

        Returns
        -------
            float
                The lowest Z position of the robot in the world."""
        # Generate model and data from a temporary copy of the spec
        model: mj.MjModel = cast("mj.MjModel", self.spec.compile())
        data = mj.MjData(model)

        # Step the simulation to ensure everything is stable
        mj.mj_forward(model, data)

        # Iterate over all geoms
        lowest_point = np.inf
        for i in range(model.ngeom):
            # Get the geometry
            geom = data.geom(i)
            bodyid = int(model.geom_bodyid[geom.id])
            parentid = int(model.body(bodyid).parentid[0])

            # Possible names
            name_of_geom = geom.name
            name_of_body = model.body(bodyid).name
            name_of_parent = model.body(parentid).name

            # If the geom does not belong to the spawned robot, skip it
            if (
                (spawn_name not in name_of_geom)
                and (spawn_name not in name_of_body)
                and (spawn_name not in name_of_parent)
            ):
                continue

            # Global position of the geometry (x, y, z)
            pos = data.geom_xpos[geom.id]

            # World rotation matrix (flat 9 values in row-major)
            r_mat = np.array(data.geom_xmat[geom.id]).reshape(3, 3)

            # Local half-sizes (sx, sy, sz)
            sx, sy, sz = model.geom_size[geom.id]  # box half extents

            # Generate 8 local corner offsets
            corners_local = np.array([
                [dx * sx, dy * sy, dz * sz]
                for dx in (-1, 1)
                for dy in (-1, 1)
                for dz in (-1, 1)
            ])

            # Transform corners: world_corner = pos + R @ local_corner
            corners_world = pos + corners_local @ r_mat.T  # (8,3)

            # Return the lowest Z value
            maybe_lowest_point = np.min(corners_world[:, 2])
            lowest_point = min(lowest_point, maybe_lowest_point)

        # Clear the temporary objects
        del model, data

        # Return the lowest position rounded to avoid floating point issues
        if lowest_point == np.inf:
            return 0.0
        return np.round(lowest_point, 6)

    def _find_contacts(self) -> set[tuple[str, str, float]]:
        # Generate model and data
        model = self.spec.compile()
        data = mj.MjData(model)

        # Step the simulation to ensure everything is stable
        mj.mj_forward(model, data)

        # Discover contacts between the world and the spawned robots
        contact_pairs = set()
        for contact in data.contact:
            geom1 = mj.mj_id2name(
                m=model,
                type=mj.mjtObj.mjOBJ_GEOM,
                id=contact.geom1,
            )
            geom2 = mj.mj_id2name(
                m=model,
                type=mj.mjtObj.mjOBJ_GEOM,
                id=contact.geom2,
            )
            contact_pairs |= {(geom1, geom2, contact.dist)}

        # Clear the temporary objects
        del model, data

        # Return the set of contact pairs
        return contact_pairs

    def _check_and_correct_spawn(
        self,
        spawn_site: mj.MjsBody,
        spawn_body: mj.MjsBody,
        spawn_name: str,
        base_point: float = 0.01,
        *,
        validate_no_collisions: bool = False,
    ) -> None:
        """
        Check and correct the spawn position to avoid collisions with the floor.


        Parameters
        ----------
        spawn_site : mj.MjsBody
            The site where the robot is spawned.
        spawn_body : mj.MjsBody
            The body of the spawned robot.
        spawn_name : str
            The prefix name of the spawned robot.
        base_point : float, optional
            Minimum distance above the lowest point of the robot, by default 0.01
        validate_no_collisions : bool, optional
            Whether to validate the spawn position for collisions, by default False
        """

        # Log the correction process
        msg = "-" * 60
        log.debug(msg)

        # Get the spawn position
        msg = f"Initial spawn position: {spawn_site.pos}"
        log.debug(msg)

        # Find lowest position of the robot
        lowest_position = self._find_lowest_position(spawn_name)
        msg = f"Lowest robot position: {lowest_position} m"
        log.debug(msg)

        # Adjust the spawn position to ensure the robot is above ground
        diff_from_base = (base_point + spawn_site.pos[2]) - lowest_position
        spawn_body.pos[2] += diff_from_base
        msg = f"Adjusted spawn position: {spawn_body.pos}"
        log.debug(msg)

        # Validate the spawn position by checking for collisions
        if validate_no_collisions is True:
            contact_pairs = self._find_contacts()
            for contact in contact_pairs:
                # Unpack contact details
                geom1_name, geom2_name, dist = contact

                # If there is a collision with the floor, log a warning
                floor_name = self.mujoco_config.floor_name
                if floor_name in geom1_name or floor_name in geom2_name:
                    msg = "Spawn position causes collision with the floor!\n"
                    msg += f"--> '{geom1_name}', '{geom2_name}'\n"
                    msg += f"\t With distance: {dist}\n"
                    msg += " Please adjust the spawn position: \n"
                    msg += f"\t {spawn_site.pos=}"
                    log.warning(msg)
                else:
                    # Log other collisions as debug info
                    msg = "Spawn position causes collision!\n"
                    msg += f"--> '{geom1_name}', '{geom2_name}'\n"
                    msg += f"\t With distance: {dist}"
                    log.debug(msg)

    def spawn(
        self,
        robot_spec: mj.MjSpec,
        position: Position | None = None,
        rotation: Rotation | None = None,
        spawn_prefix: str | None = None,
        *,
        correct_collision_with_floor: bool = True,
        validate_no_collisions: bool = False,
        rotation_sequence: str = "XYZ",  # xyzXYZ, assume intrinsic
    ) -> mj.MjSpec:
        """
        Spawn a robot into the world at a specified position and orientation.

        Parameters
        ----------
        robot_spec : mj.MjSpec
            The MuJoCo specification of the robot to be spawned.
        position : Position, optional
            The (x, y, z) position to spawn the robot, by default None
        rotation : Rotation, optional
            The (x, y, z) Euler angles (in degrees) for the robot's orientation, by default None
        spawn_prefix : str, optional
            Prefix for naming the spawned robot, by default None
        correct_collision_with_floor : bool, optional
            Whether to adjust the spawn position to avoid collisions with the floor, by default True
        validate_no_collisions : bool, optional
            Whether to validate the spawn position for collisions after adjustment, by default False
        rotation_sequence : str, optional
            The sequence of axes for Euler to quaternion conversion, by default "XYZ"

        Returns
        -------
            mj.MjSpec
                The updated MuJoCo world specification with the spawned robot.
        """
        # Default spawn position
        if position is None:
            position = self.default_spawn_position
        else:
            position = deepcopy(position)

        # Default spawn orientation
        if rotation is None:
            rotation = self.default_spawn_rotation
        else:
            rotation = deepcopy(rotation)

        # If no prefix is given, use the default one
        if spawn_prefix is None:
            spawn_prefix = self.spawn_prefix

        # Convert rotation from Euler angles (degrees) to quaternion
        rotation_as_quat = euler_to_quat_conversion(
            rotation,
            rotation_sequence,
        )

        # Increment the spawn count
        self.spawns += 1

        # Create a spawn site at the specified position
        spawn_site = self.spec.worldbody.add_site(
            pos=np.array(position),
            quat=np.array(rotation_as_quat),
        )

        # Attach the robot body to the spawn site
        spawn_name = f"{spawn_prefix}{self.spawns}_"
        spawn_body = spawn_site.attach_body(
            body=robot_spec.worldbody,
            prefix=spawn_name,
        )

        # Correct the spawn position if requested
        if correct_collision_with_floor is True:
            self._check_and_correct_spawn(
                spawn_site,
                spawn_body,
                spawn_name,
                validate_no_collisions=validate_no_collisions,
            )

        # Allow the robot to move freely
        spawn_body.add_freejoint()

        # Return a copy of the updated spec
        return self.spec

    def store_to_xml(self) -> None:
        # Derive save path
        this_script_path = Path(__file__)
        save_dir = this_script_path.parent / "pre_compiled"
        save_path = save_dir / f"{self.name}.xml"
        save_dir.mkdir(parents=True, exist_ok=True)
        xml = self.spec.to_xml()

        # Save file
        with save_path.open("w") as f:
            f.write(xml)

    def load_from_xml(self) -> bool:
        # Derive XML path
        this_script_path = Path(__file__)
        xml_path = this_script_path.parent / "pre_compiled" / f"{self.name}.xml"

        # Check if file exists
        if not xml_path.exists():
            return False

        # Load the XML file
        self.spec = mj.MjSpec.from_file(str(xml_path))
        return True
