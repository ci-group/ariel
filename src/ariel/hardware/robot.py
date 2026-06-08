"""Lightweight robot descriptor for hardware deployment.

No MuJoCo required.  Use :class:`HardwareRobot` instead of the full
robogen-lite module tree when running on the physical robot.

Example
-------
    from ariel.hardware.robot import HardwareRobot

    # Spider: 8 joints wired to Robohat servo channels 0-7
    robot = HardwareRobot(n=8, servo_map=list(range(8)))
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HardwareRobot:
    """Minimal robot descriptor that the hardware runner needs.

    Parameters
    ----------
    n : int
        Number of actuated joints (must equal CPG oscillator count).
    servo_map : list[int]
        Maps joint index ``i`` to the physical Robohat servo channel number.
        ``servo_map[i]`` must be a valid channel (0-15 for one assembly board,
        0-31 for two).  Defaults to identity ``[0, 1, ..., n-1]``.
    """

    n: int
    servo_map: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.servo_map:
            self.servo_map = list(range(self.n))
        if len(self.servo_map) != self.n:
            raise ValueError(
                f"servo_map length {len(self.servo_map)} does not match n={self.n}."
            )
