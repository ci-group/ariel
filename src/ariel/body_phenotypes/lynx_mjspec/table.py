"""Table environment for MjSpec Lynx arm."""
from dataclasses import dataclass
import mujoco


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
                diffuse=[0.3, 0.3, 0.3],
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
            rgba=[0.95, 0.88, 0.7, 1],
            friction=[0.7, 0.1, 0.1],
            contype=1,
            conaffinity=1,
        )

        # Table legs (4 corners)
        leg_positions = [
            [self.table_width / 2 - self.leg_width / 2, self.table_depth / 2 - self.leg_width / 2],
            [-self.table_width / 2 + self.leg_width / 2, self.table_depth / 2 - self.leg_width / 2],
            [self.table_width / 2 - self.leg_width / 2, -self.table_depth / 2 + self.leg_width / 2],
            [-self.table_width / 2 + self.leg_width / 2, -self.table_depth / 2 + self.leg_width / 2],
        ]

        for i, (x, y) in enumerate(leg_positions):
            self.spec.worldbody.add_geom(
                name=f"table_leg_{i + 1}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[x, y, self.table_height / 2],
                size=[self.leg_width / 2, self.leg_width / 2, self.table_height / 2],
                rgba=[0.3, 0.3, 0.3, 1],
                friction=[0.7, 0.1, 0.1],
                contype=1,
                conaffinity=1,
            )

        # Floor plane
        self.spec.worldbody.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            pos=[0, 0, -0.01],
            size=[10, 10, 0.1],
            rgba=[0.79, 0.68, 0.4, 1],
            friction=[0.7, 0.1, 0.1],
            contype=1,
            conaffinity=1,
        )

        # Add target site
        self.spec.worldbody.add_site(
            name="target",
            size=[0.03, 0.03, 0.03],
            rgba=[1, 0, 0, 1],
            pos=[0, 0, self.table_height + self.table_thickness / 2 + 0.01],
        )

    def spawn(self, other_spec) -> None:
        """Spawn robot body flush on the table surface."""
        # Robot base is flush with the table surface
        table_top_z = self.table_height + self.table_thickness / 2
        spawn_site = self.spec.worldbody.add_site(
            pos=[0, 0, table_top_z],
            quat=[1, 0, 0, 0],
        )

        # Attach the arm spec to the spawn site
        spawn_site.attach_body(
            body=other_spec.worldbody,
            prefix="robot_",
        )
