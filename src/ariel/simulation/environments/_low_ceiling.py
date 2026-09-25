"""MuJoCo world with a low overhead clearance."""

from dataclasses import dataclass

import mujoco

from ariel.simulation.environments._base_world import BaseWorld


@dataclass
class LowCeilingWorld(BaseWorld):
    """Flat corridor with configurable overhead clearance."""

    name: str = "low-ceiling-world"

    corridor_length: float = 4.0
    corridor_width: float = 2.0
    clearance_height: float = 0.30
    ceiling_thickness: float = 0.05

    load_precompiled: bool = False

    def __post_init__(self) -> None:
        super().__init__(
            name=self.name,
            load_precompiled=self.load_precompiled,
        )

        if self.is_precompiled:
            return

        self._expand_spec()

    def _expand_spec(self) -> None:
        floor = self.spec.worldbody.add_body(
            name="floor",
            pos=[
                0,
                0,
                -0.025,
            ],
        )

        floor.add_geom(
            name="floor-geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[
                self.corridor_length / 2,
                self.corridor_width / 2,
                0.025,
            ],
            rgba=[
                0.3,
                0.3,
                0.3,
                1.0,
            ],
        )

        ceiling = self.spec.worldbody.add_body(
            name="ceiling",
            pos=[
                0,
                0,
                self.clearance_height
                + self.ceiling_thickness / 2,
            ],
        )

        ceiling.add_geom(
            name="ceiling-geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[
                self.corridor_length / 2,
                self.corridor_width / 2,
                self.ceiling_thickness / 2,
            ],
            rgba=[
                0.5,
                0.5,
                0.5,
                1.0,
            ],
        )