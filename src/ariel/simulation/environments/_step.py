"""MuJoCo world with a single raised step."""

from dataclasses import dataclass

import mujoco

from ariel.simulation.environments._base_world import BaseWorld


@dataclass
class StepWorld(BaseWorld):
    """Two platforms separated by a vertical step."""

    name: str = "step-world"

    step_height: float = 0.15
    platform_length: float = 2.0
    platform_width: float = 2.0
    platform_thickness: float = 0.05

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
        half_length = self.platform_length / 2
        half_width = self.platform_width / 2
        half_thickness = self.platform_thickness / 2

        lower = self.spec.worldbody.add_body(
            name="lower-platform",
            pos=[
                -half_length,
                0,
                -half_thickness,
            ],
        )

        lower.add_geom(
            name="lower-platform-geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[
                half_length,
                half_width,
                half_thickness,
            ],
        )

        upper = self.spec.worldbody.add_body(
            name="upper-platform",
            pos=[
                half_length,
                0,
                self.step_height - half_thickness,
            ],
        )

        upper.add_geom(
            name="upper-platform-geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[
                half_length,
                half_width,
                half_thickness,
            ],
        )