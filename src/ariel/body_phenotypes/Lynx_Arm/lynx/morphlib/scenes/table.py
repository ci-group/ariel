from dataclasses import dataclass

import mujoco

from ariel import ROOT, CWD

print(ROOT)

@dataclass
class SimpleFlatWorld:
    """A flat world with a chequerboard floor."""

    name: str = "simple-flat-world"
    floor_size: tuple = (10, 10, 1)

    def __init__(self) -> None:
        self.spec = mujoco.MjSpec()
        self.spec.copy_during_attach = True
        self._floor_name = "floor"
        width, height, depth = self.floor_size

        # Create checker texture
        self.spec.add_texture(
            name=self._floor_name,
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            rgb1=[0.1, 0.2, 0.3],
            rgb2=[0.2, 0.3, 0.4],
            width=600,
            height=600,
        )
        self.spec.add_material(
            name=self._floor_name,
            textures=["", self._floor_name],
            texrepeat=[3, 3],
            texuniform=True,
            reflectance=0,
        )

        # Add floor body
        floor = self.spec.worldbody.add_body(name=self._floor_name, pos=[0, 0, 0])
        floor.add_geom(
            name=self._floor_name,
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[width / 2, height / 2, depth / 2],
            material=self._floor_name,
        )

        # Add lights
        self.spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8], ambient=[0.3, 0.3, 0.3])
        self.spec.worldbody.add_light(pos=[1, 1, 3], dir=[-1, -1, -1], diffuse=[0.5, 0.5, 0.5])

    def spawn(self, other_spec) -> None:
        # Create a spawn site at the specified position
        spawn_site = self.spec.worldbody.add_site(
            pos=[0, 0, 0.02],  # Slightly above floor
            quat=[1, 0, 0, 0],
        )

        # Manually copy meshes to the world spec with the expected prefix
        mesh_files = [
            str(CWD / "src/ariel/body_phenotypes/Lynx_Arm/models/clamp_0205.stl"),
            str(CWD / "src/ariel/body_phenotypes/Lynx_Arm/models/clamp_0226.stl"),
            str(CWD / "src/ariel/body_phenotypes/Lynx_Arm/models/clamp_0227.stl"),
            str(CWD / "src/ariel/body_phenotypes/Lynx_Arm/models/clamp_1216.stl"),
        ]
        for mesh_file in mesh_files:
            mesh_base_name = mesh_file.split("/")[-1].replace(".stl", "")
            for i in range(1, 6):
                self.spec.add_mesh(
                    name=f"robot_tube{i}_{mesh_base_name}",
                    file=mesh_file,
                    scale=[0.001, 0.001, 0.001],
                )

        spawn_site.attach_body(
            body=other_spec.worldbody,
            prefix="robot_",
        )

        # Do NOT add freejoint to keep it fixed to the spawn site (which is fixed to worldbody)
        # spawn_body.add_freejoint()


@dataclass
class TableWorld:
    """A world with a table and a floor."""

    name: str = "table-world"

    def __init__(self) -> None:
        self.spec = mujoco.MjSpec()
        self.spec.copy_during_attach = True

        # Table dimensions
        self.table_width = 1.0
        self.table_depth = 0.8
        self.table_height = 0.6
        self.table_thickness = 0.018
        self.leg_width = 0.045

        # Add skybox
        self.spec.add_texture(
            name="skybox",
            type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
            rgb1=[0.25, 0.25, 0.25],
            rgb2=[0.5, 0.5, 0.5],
            width=512,
            height=512,
        )

        # Add lighting (4 lights at corners)
        light_positions = [
            [self.table_width * 5, self.table_depth * 5, 10],
            [-self.table_width * 5, self.table_depth * 5, 10],
            [self.table_width * 5, -self.table_depth * 5, 10],
            [-self.table_width * 5, -self.table_depth * 5, 10],
        ]
        for i, pos in enumerate(light_positions):
            self.spec.worldbody.add_light(
                name=f"light_{i}",
                cutoff=100,
                diffuse=[0.3, 0.3, 0.3],  # Reduced from [1, 1, 1]
                dir=[0, 0, -1],
                exponent=1,
                pos=pos,
                specular=[0.1, 0.1, 0.1],
                castshadow=False,
            )

        # Table top (square)
        self.spec.worldbody.add_geom(
            name="table_top",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0, 0, self.table_height + self.table_thickness / 2 + 1e-6],
            size=[self.table_width / 2, self.table_depth / 2, self.table_thickness / 2],
            rgba=[0.95, 0.88, 0.7, 1],  # Yellower: increased R/G, decreased B
            friction=[0.7, 0.1, 0.1],
            contype=0,
            conaffinity=0,
        )

        # Table legs (4 corners)
        leg_positions = [
            [self.table_width / 2 - self.leg_width / 2, self.table_depth / 2 - self.leg_width / 2],   # Front right
            [-self.table_width / 2 + self.leg_width / 2, self.table_depth / 2 - self.leg_width / 2],  # Front left
            [self.table_width / 2 - self.leg_width / 2, -self.table_depth / 2 + self.leg_width / 2],  # Back right
            [-self.table_width / 2 + self.leg_width / 2, -self.table_depth / 2 + self.leg_width / 2],  # Back left
        ]

        for i, (x, y) in enumerate(leg_positions):
            self.spec.worldbody.add_geom(
                name=f"table_leg_{i + 1}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[x, y, self.table_height / 2],
                size=[self.leg_width / 2, self.leg_width / 2, self.table_height / 2],
                rgba=[0.3, 0.3, 0.3, 1],
                friction=[0.7, 0.1, 0.1],
                contype=0,
                conaffinity=0,
            )

        # Floor plane
        self.spec.worldbody.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            pos=[0, 0, -0.01],
            size=[10, 10, 0.1],
            rgba=[0.79, 0.68, 0.4, 1],
            friction=[0.7, 0.1, 0.1],
            contype=0,
            conaffinity=0,
        )

        # Add target site
        self.spec.worldbody.add_site(
            name="target",
            # type=mujoco.mjtGeom.mjGEOM_SPHERE, # Removed type to use default sphere
            size=[0.03, 0.03, 0.03],
            rgba=[1, 0, 0, 1],
            pos=[0, 0, self.table_height + self.table_thickness / 2 + 0.01],
        )

    def spawn(self, other_spec) -> None:
        # Create a spawn site at the specified position (on top of the table)
        spawn_site = self.spec.worldbody.add_site(
            pos=[0, 0, self.table_height + self.table_thickness / 2 + 0.01],  # Slightly above table top
            quat=[1, 0, 0, 0],
        )

        # Manually copy meshes to the world spec with the expected prefix
        # Note: In a real scenario, we might want to automate this or handle it in construct_lynx
        mesh_files = [
            str(CWD / "src/ariel/body_phenotypes/Lynx_Arm/models/clamp_0205.stl"),
            str(CWD / "src/ariel/body_phenotypes/Lynx_Arm/models/clamp_0226.stl"),
            str(CWD / "src/ariel/body_phenotypes/Lynx_Arm/models/clamp_0227.stl"),
            str(CWD / "src/ariel/body_phenotypes/Lynx_Arm/models/clamp_1216.stl"),
        ]
        for mesh_file in mesh_files:
            mesh_base_name = str(mesh_file).split("/")[-1].replace(".stl", "")
            for i in range(1, 6):
                self.spec.add_mesh(
                    name=f"robot_tube{i}_{mesh_base_name}",
                    file=str(mesh_file),
                    scale=[0.001, 0.001, 0.001],
                )

        spawn_site.attach_body(
            body=other_spec.worldbody,
            prefix="robot_",
        )


def table_terrain_mjx(env_mjcf) -> None:
    """Builds a table terrain for the simulation. Includes a square table with 4 legs and a light source."""
    # Add the same lighting as the plane terrain
    env_mjcf.worldbody.add(
        "light",
        cutoff=100,
        diffuse=[1, 1, 1],
        dir=[0, 0, -1.3],
        directional=True,
        exponent=1,
        pos=[0, 0, 1.3],
        specular=[0.1, 0.1, 0.1],
        castshadow=False,
    )

    # Table dimensions
    table_width = 1.0
    table_depth = 1.0
    table_height = 0.8
    table_thickness = 0.05
    leg_width = 0.1

    # Table top (square)
    env_mjcf.worldbody.add(
        "geom",
        friction=[0.7, 0.1, 0.1],
        conaffinity=0,  # Disable collision
        contype=0,
        condim=3,
        name="table_top",
        pos=[0, 0, table_height],
        rgba=[0.6, 0.4, 0.2, 1],  # Brown wood color
        size=[table_width / 2, table_depth / 2, table_thickness / 2],  # Revert to original box size
        type="box",  # Revert to box type
    )

    # Table legs (4 corners)
    leg_positions = [
        [table_width / 2 - leg_width / 2, table_depth / 2 - leg_width / 2],   # Front right
        [-table_width / 2 + leg_width / 2, table_depth / 2 - leg_width / 2],  # Front left
        [table_width / 2 - leg_width / 2, -table_depth / 2 + leg_width / 2],  # Back right
        [-table_width / 2 + leg_width / 2, -table_depth / 2 + leg_width / 2],  # Back left
    ]

    for i, (x, y) in enumerate(leg_positions):
        env_mjcf.worldbody.add(
            "geom",
            friction=[0.7, 0.1, 0.1],
            conaffinity=0,  # Disable collision
            contype=0,
            condim=3,
            name=f"table_leg_{i + 1}",
            pos=[x, y, table_height / 2],
            rgba=[0.5, 0.3, 0.1, 1],  # Darker brown for legs
            size=[leg_width / 2, leg_width / 2, table_height / 2],  # Revert to original box size
            type="box",  # Revert to box type
    )

    # Optional: Add a floor plane underneath
    #     env_mjcf.worldbody.add("geom", friction=[0.7, 0.1, 0.1], conaffinity=1, condim=3, name="floor", pos=[0,0,0], rgba=[0.8, 0.9, 0.8, 1], size=[40,40,40], type="plane", material="MatPlane")

    # env_mjcf.worldbody.add(
    #     "geom",
    #     friction=[0.7, 0.1, 0.1],
    #     conaffinity=1,
    #     condim=3,
    #     name="floor",
    #     pos=[0, 0, -0.01],
    #     rgba=[0.8, 0.9, 0.8, 1],
    #     size=[40, 40, 40],
    #     type="plane",
    #     material="MatPlane"
    # )

    # Original target site (commented out)
    # env_mjcf.worldbody.add(
    #     "site",
    #     type="sphere",
    #     size=[0.1],
    #     rgba=[1, 0, 0, 1],
    #     pos=[0, 0, table_height + table_thickness/2 + 0.01],  # Slightly above the table
    #     name="target"
    # )
    # Wrap the target site in a mocap body for dynamic control in MJX
    target_mocap_body = env_mjcf.worldbody.add(
        "body", name="target_mocap_body", mocap="true", pos=[0, 0, table_height + table_thickness / 2 + 0.01],
    )
    target_mocap_body.add(
        "site",
        type="sphere",
        size=[0.01],  # Smaller size for the site itself, as the body defines the main position
        rgba=[1, 0, 0, 1],
        name="target",
        pos=[0, 0, 0],  # Position relative to the mocap body
    )


def table_terrain(env_mjcf) -> None:
    """Builds a table terrain for the simulation. Includes a square table with 4 legs and a light source."""
    # Add the same lighting as the plane terrain
    env_mjcf.worldbody.add(
        "light",
        cutoff=100,
        diffuse=[1, 1, 1],
        dir=[0, 0, -1.3],
        directional=True,
        exponent=1,
        pos=[0, 0, 1.3],
        specular=[0.1, 0.1, 0.1],
        castshadow=False,
    )

    # Table dimensions
    table_width = 1.0
    table_depth = 0.8
    table_height = 0.6
    table_thickness = 0.018
    leg_width = 0.045

    # Table top (square)
    env_mjcf.worldbody.add(
        "geom",
        friction=[0.7, 0.1, 0.1],
        # group=GROUP_ENV,
        # contype=1 << GROUP_ENV,
        # conaffinity=(1 << GROUP_TUBE) | (1 << GROUP_JOINT),
        contype=0,
        conaffinity=0,
        name="table_top",
        pos=[0, 0, table_height + table_thickness / 2 + 1e-6],
        rgba=[0.89, 0.87, 0.81, 1],  # [0.6, 0.4, 0.2, 1],  # Brown wood color
        size=[table_width / 2, table_depth / 2, table_thickness / 2],
        type="box",
    )

    # Table legs (4 corners)
    leg_positions = [
        [table_width / 2 - leg_width / 2, table_depth / 2 - leg_width / 2],   # Front right
        [-table_width / 2 + leg_width / 2, table_depth / 2 - leg_width / 2],  # Front left
        [table_width / 2 - leg_width / 2, -table_depth / 2 + leg_width / 2],  # Back right
        [-table_width / 2 + leg_width / 2, -table_depth / 2 + leg_width / 2],  # Back left
    ]

    for i, (x, y) in enumerate(leg_positions):
        env_mjcf.worldbody.add(
            "geom",
            friction=[0.7, 0.1, 0.1],
            # group=GROUP_ENV,
            # contype=1 << GROUP_ENV,
            # conaffinity=(1 << GROUP_TUBE) | (1 << GROUP_JOINT),
            contype=0,
            conaffinity=0,
            name=f"table_leg_{i + 1}",
            pos=[x, y, table_height / 2],
            rgba=[0.3, 0.3, 0.3, 1],
            size=[leg_width / 2, leg_width / 2, table_height / 2],
            type="box",
    )

    # Optional: Add a floor plane underneath
    #     env_mjcf.worldbody.add("geom", friction=[0.7, 0.1, 0.1], conaffinity=1, condim=3, name="floor", pos=[0,0,0], rgba=[0.8, 0.9, 0.8, 1], size=[40,40,40], type="plane", material="MatPlane")

    env_mjcf.worldbody.add(
        "geom",
        friction=[0.7, 0.1, 0.1],
        # group=GROUP_ENV,
        # contype=1 << GROUP_ENV,
        # conaffinity=(1 << GROUP_TUBE) | (1 << GROUP_JOINT),
        contype=0,
        conaffinity=0,
        name="floor",
        pos=[0, 0, -0.01],
        rgba=[0.79, 0.72, 0.45, 1],  # 0.8, 0.9, 0.8, 1],
        size=[10, 10, 0.1],
        type="plane",
        # material="MatPlane"
    )

    # Add target site
    env_mjcf.worldbody.add(
        "site",
        type="sphere",
        size=[0.03],
        rgba=[1, 0, 0, 1],
        pos=[0, 0, table_height + table_thickness / 2 + 0.01],  # Slightly above the table
        name="target",
    )

    # sky = env_mjcf.asset.add(
    #     "texture",
    #     name="sky",
    #     type="skybox",
    #     width=2,
    #     height=2,
    #     rgb1=[0.5, 0.5, 0.5], # Light gray sky
    #     )

    # # env_mjcf.visual.map.texture = 0
    # # env_mjcf.visual.map.rgb.clear = [0.5, 0.5, 0.5, 1]
    # # print(f"+++++++++++++++++ map: {env_mjcf.visual}")
    # env_mjcf.visual.map.skybox = sky
