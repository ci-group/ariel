"""MuJoCo world with a gap between two platforms."""

from dataclasses import dataclass

import mujoco

from ariel.simulation.environments._base_world import BaseWorld


@dataclass
class GapWorld(BaseWorld):
    """Two flat platforms separated by a configurable gap."""

    name: str = "gap-world"

    gap_width: float = 0.30
    platform_length: float = 2.0
    platform_width: float = 2.0
    platform_height: float = 0.05

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
        half_height = self.platform_height / 2

        left_center_x = -(
            self.gap_width / 2
            + half_length
        )

        right_center_x = (
            self.gap_width / 2
            + half_length
        )

        left_platform = self.spec.worldbody.add_body(
            name="left-platform",
            pos=[
                left_center_x,
                0,
                -half_height,
            ],
        )

        left_platform.add_geom(
            name="left-platform-geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[
                half_length,
                half_width,
                half_height,
            ],
            rgba=[
                0.3,
                0.3,
                0.3,
                1.0,
            ],
        )

        right_platform = self.spec.worldbody.add_body(
            name="right-platform",
            pos=[
                right_center_x,
                0,
                -half_height,
            ],
        )

        right_platform.add_geom(
            name="right-platform-geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[
                half_length,
                half_width,
                half_height,
            ],
            rgba=[
                0.3,
                0.3,
                0.3,
                1.0,
            ],
        )